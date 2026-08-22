"""Unit tests for the router tick. Fake provider + fake submitter: no RPC, no
shell, no real queue file except a temp round-trip."""

import os
import tempfile
import unittest

from google3.experimental.users.qiaos.tpu_utils import avail_provider as AP
from google3.experimental.users.qiaos.tpu_utils import route_check as RC
from google3.experimental.users.qiaos.tpu_utils import route_lib as R


def _entry(job_id='j1', power='v7-32', archs=('v7',), **kw):
  return R.QueueEntry(job_id=job_id, power=power, allowed_archs=list(archs), **kw)


def _avail(cell, arch, free, oversold=False, price=20.0, metro=''):
  return R.CellAvail(cell=cell, arch=arch, free_chips=free, oversold=oversold,
                     price=price, metro=metro or cell)


class _FakeProvider:
  """Stands in for AvailabilityProvider.fetch()."""

  def __init__(self, avail_by_cell, arch_price=None, arch_pool=None):
    self._a = avail_by_cell
    self._p = arch_price or {}
    self._pool = arch_pool or {}

  def fetch(self):
    return self._a, self._p, self._pool


class _FakeSubmitter:
  """Records argv, returns a scripted xid (or None to simulate a dead launch)."""

  def __init__(self, xid='555001'):
    self.calls = []
    self.cwds = []
    self.cancels = []
    self._xid = xid

  def submit(self, argv, cwd=''):
    self.calls.append(argv)
    self.cwds.append(cwd)
    return self._xid, f'Launched experiment {self._xid}' if self._xid else 'no line'

  def cancel(self, xid):
    self.cancels.append(xid)
    return True, 'stopped'


class QueuePersistenceTest(unittest.TestCase):

  def test_round_trip(self):
    path = tempfile.mkstemp(suffix='.json')[1]
    self.addCleanup(os.remove, path)
    entries = [
        _entry('a', power='v7-32', archs=('v7', 'v6p'), priority=3,
               launch_kwargs={'config': 'x.py', 'force': True}),
        _entry('b', power='v6e-32', archs=('v6e',), topology_locked=True),
    ]
    RC.save_queue(path, entries)
    back = RC.load_queue(path)
    self.assertEqual([e.job_id for e in back], ['a', 'b'])
    self.assertEqual(back[0].priority, 3)
    self.assertEqual(back[0].launch_kwargs, {'config': 'x.py', 'force': True})
    self.assertTrue(back[1].topology_locked)
    self.assertEqual(back[0].state, R.JobState.QUEUED)

  def test_missing_file_is_empty(self):
    self.assertEqual(RC.load_queue('/no/such/queue.json'), [])


class BuildCmdTest(unittest.TestCase):

  def test_basic_shape(self):
    e = _entry('j1', tier='PROD')
    p = R.Placement(job_id='j1', arch='v7', chips=32, cell='yutulpz',
                    price=20.0, reason='r')
    argv = RC.build_tpu_queue_cmd(p, e, group='9')
    self.assertEqual(argv[:2], ['tpu', 'queue'])
    self.assertIn('--tpu_type=v7-32', argv)
    self.assertIn('--group=9', argv)
    self.assertIn('--cell=yutulpz', argv)
    self.assertIn('--tier=PROD', argv)

  def test_launch_kwargs_forms(self):
    e = _entry('j1', tier='', launch_kwargs={
        'config': 'cfg.py',        # --config=cfg.py
        'force': True,             # --force  (bare)
        'skip_preflight': None,    # --skip_preflight (bare)
        'disabled': False,         # omitted
        '--already_dashed': 'v',   # kept as-is
    })
    p = R.Placement(job_id='j1', arch='v6p', chips=32, cell='nk',
                    price=9.0, reason='r')
    argv = RC.build_tpu_queue_cmd(p, e)
    self.assertIn('--config=cfg.py', argv)
    self.assertIn('--force', argv)
    self.assertIn('--skip_preflight', argv)
    self.assertNotIn('--disabled', argv)
    self.assertNotIn('--disabled=False', argv)
    self.assertIn('--already_dashed=v', argv)
    self.assertNotIn('--tier=', ' '.join(argv))   # empty tier -> no flag


class ExtractXidTest(unittest.TestCase):

  def test_launched_line(self):
    self.assertEqual(RC.extract_xid('foo\nLaunched experiment 12345\nbar'),
                     '12345')

  def test_resume_workunit_line(self):
    self.assertEqual(
        RC.extract_xid('Added 1 work unit(s) to experiment 67890'), '67890')

  def test_ansi_colorized_id(self):
    self.assertEqual(
        RC.extract_xid('Launched experiment \x1b[1m\x1b[34m99999\x1b[0m'),
        '99999')

  def test_no_line(self):
    self.assertIsNone(RC.extract_xid('build failed, SIGBUS'))


class RunTickTest(unittest.TestCase):

  def test_no_queued_jobs(self):
    e = _entry('j1')
    e.state = R.JobState.RUNNING
    prov = _FakeProvider({})
    out, log = RC.run_tick([e], prov, now=0.0)
    self.assertTrue(any('nothing to do' in l for l in log))

  def test_dry_run_does_not_submit(self):
    e = _entry('j1', power='v7-32', archs=('v7',))
    prov = _FakeProvider({'yutulpz|v7': _avail('yutulpz', 'v7', 320)},
                         arch_price={'v7': 20.0}, arch_pool={'v7': 320})
    sub = _FakeSubmitter()
    out, log = RC.run_tick([e], prov, now=0.0, submitter=sub, dry_run=True)
    self.assertEqual(sub.calls, [])                       # nothing submitted
    self.assertEqual(e.state, R.JobState.QUEUED)          # unchanged
    self.assertTrue(any(l.startswith('[DRY]') for l in log))

  def test_live_submit_marks_submitted(self):
    e = _entry('j1', power='v7-32', archs=('v7',))
    prov = _FakeProvider({'yutulpz|v7': _avail('yutulpz', 'v7', 320)},
                         arch_price={'v7': 20.0}, arch_pool={'v7': 320})
    sub = _FakeSubmitter(xid='777')
    out, log = RC.run_tick([e], prov, now=100.0, submitter=sub, dry_run=False)
    self.assertEqual(len(sub.calls), 1)
    self.assertEqual(e.state, R.JobState.SUBMITTED)
    self.assertEqual(e.xid, '777')
    self.assertEqual(e.cell, 'yutulpz')
    self.assertEqual(e.arch, 'v7')
    self.assertEqual(e.chips, 32)
    self.assertEqual(e.submitted_at, 100.0)

  def test_workdir_is_passed_to_submitter_as_cwd(self):
    # REGRESSION (monitor v21 field report): the router must package `tpu queue`
    # from the job's OWN checkout, or a run whose config lives in a snapshot dir
    # (not via --config) ships the wrong source. workdir must reach submit(cwd=).
    e = _entry('j1', power='v7-32', archs=('v7',))
    e.workdir = '/some/checkout/dir'
    prov = _FakeProvider({'yutulpz|v7': _avail('yutulpz', 'v7', 320)},
                         arch_price={'v7': 20.0}, arch_pool={'v7': 320})
    sub = _FakeSubmitter(xid='777')
    RC.run_tick([e], prov, now=0.0, submitter=sub, dry_run=False)
    self.assertEqual(sub.cwds, ['/some/checkout/dir'])

  def test_no_workdir_passes_empty_cwd(self):
    e = _entry('j1', power='v7-32', archs=('v7',))   # workdir defaults to ''
    prov = _FakeProvider({'yutulpz|v7': _avail('yutulpz', 'v7', 320)},
                         arch_price={'v7': 20.0}, arch_pool={'v7': 320})
    sub = _FakeSubmitter(xid='777')
    RC.run_tick([e], prov, now=0.0, submitter=sub, dry_run=False)
    self.assertEqual(sub.cwds, [''])                  # inherit router CWD

  def test_live_submit_no_xid_marks_failed_attempt(self):
    e = _entry('j1', power='v7-32', archs=('v7',))
    prov = _FakeProvider({'yutulpz|v7': _avail('yutulpz', 'v7', 320)},
                         arch_price={'v7': 20.0}, arch_pool={'v7': 320})
    sub = _FakeSubmitter(xid=None)                        # dead launch
    out, log = RC.run_tick([e], prov, now=0.0, submitter=sub, dry_run=False)
    self.assertEqual(e.state, R.JobState.QUEUED)          # stays queued
    self.assertEqual(e.attempts, 1)
    self.assertIsNone(e.xid)

  def test_nothing_placeable_keeps_queued(self):
    e = _entry('j1', power='v7-32', archs=('v7',))
    # only an oversold cell -> no placement
    prov = _FakeProvider(
        {'yulpptr|v7': _avail('yulpptr', 'v7', 320, oversold=True)},
        arch_price={'v7': 20.0}, arch_pool={'v7': 0})
    sub = _FakeSubmitter()
    out, log = RC.run_tick([e], prov, now=0.0, submitter=sub, dry_run=False)
    self.assertEqual(sub.calls, [])
    self.assertEqual(e.state, R.JobState.QUEUED)
    self.assertTrue(any('nothing placeable' in l for l in log))

  def test_topology_locked_freezes_geometry_on_live_submit(self):
    e = _entry('j1', power='v6p-32', archs=('v7', 'v6p'), topology_locked=True)
    prov = _FakeProvider({'c|v7': _avail('c', 'v7', 320)},
                         arch_price={'v7': 20.0}, arch_pool={'v7': 320})
    sub = _FakeSubmitter(xid='888')
    RC.run_tick([e], prov, now=0.0, submitter=sub, dry_run=False)
    self.assertEqual(e.locked_geometry, '2x4x4')          # frozen from v7-32


class _FakeProbe:
  """Returns a scripted STATUS_* per xid."""

  def __init__(self, by_xid):
    self._by_xid = by_xid

  def status(self, xid):
    return self._by_xid.get(xid, RC.STATUS_UNKNOWN)


def _submitted(job_id, xid, cell, submitted_at, **kw):
  e = _entry(job_id, **kw)
  e.state = R.JobState.SUBMITTED
  e.xid = xid
  e.cell = cell
  e.arch = 'v7'
  e.chips = 32
  e.submitted_at = submitted_at
  return e


class ClassifyStateTest(unittest.TestCase):

  def test_pending_only_when_pending(self):
    self.assertEqual(RC.classify_wu_states(True, False, False), RC.STATUS_PENDING)

  def test_running_wins(self):
    self.assertEqual(RC.classify_wu_states(False, True, False), RC.STATUS_RUNNING)

  def test_terminal_wins_over_all(self):
    self.assertEqual(RC.classify_wu_states(True, True, True), RC.STATUS_TERMINAL)

  def test_coming_up_is_running(self):
    # preparing/starting: not pending, not running, not terminal -> RUNNING
    self.assertEqual(RC.classify_wu_states(False, False, False), RC.STATUS_RUNNING)


class RunRerouteTest(unittest.TestCase):

  def test_no_candidate_when_within_deadline(self):
    e = _submitted('j1', '111', 'yulpptr', submitted_at=100.0)
    probe = _FakeProbe({'111': RC.STATUS_PENDING})
    sub = _FakeSubmitter()
    _, log = RC.run_reroute([e], now=300.0, probe=probe, submitter=sub,
                            reroute_after_s=600.0, dry_run=False)
    self.assertEqual(sub.cancels, [])                     # 200s < 600s
    self.assertEqual(e.state, R.JobState.SUBMITTED)
    self.assertTrue(any('no SUBMITTED job past' in l for l in log))

  def test_pending_past_deadline_cancels_and_requeues(self):
    e = _submitted('j1', '111', 'yulpptr', submitted_at=0.0)
    probe = _FakeProbe({'111': RC.STATUS_PENDING})
    sub = _FakeSubmitter()
    RC.run_reroute([e], now=700.0, probe=probe, submitter=sub,
                   reroute_after_s=600.0, cooldown_s=1800.0, dry_run=False)
    self.assertEqual(sub.cancels, ['111'])               # cancelled
    self.assertEqual(e.state, R.JobState.QUEUED)         # back to queue
    self.assertIsNone(e.xid)
    self.assertGreater(e.cooldown_cells.get('yulpptr', 0), 700.0)  # cell cooled

  def test_dry_run_does_not_cancel(self):
    e = _submitted('j1', '111', 'yulpptr', submitted_at=0.0)
    probe = _FakeProbe({'111': RC.STATUS_PENDING})
    sub = _FakeSubmitter()
    _, log = RC.run_reroute([e], now=700.0, probe=probe, submitter=sub,
                            reroute_after_s=600.0, dry_run=True)
    self.assertEqual(sub.cancels, [])
    self.assertEqual(e.state, R.JobState.SUBMITTED)      # untouched
    self.assertTrue(any('[DRY][reroute]' in l for l in log))

  def test_running_now_is_promoted_not_cancelled(self):
    e = _submitted('j1', '111', 'yukulwh', submitted_at=0.0)
    probe = _FakeProbe({'111': RC.STATUS_RUNNING})
    sub = _FakeSubmitter()
    RC.run_reroute([e], now=700.0, probe=probe, submitter=sub,
                   reroute_after_s=600.0, dry_run=False)
    self.assertEqual(sub.cancels, [])
    self.assertEqual(e.state, R.JobState.RUNNING)

  def test_unknown_status_never_cancels(self):
    e = _submitted('j1', '111', 'yulpptr', submitted_at=0.0)
    probe = _FakeProbe({'111': RC.STATUS_UNKNOWN})
    sub = _FakeSubmitter()
    RC.run_reroute([e], now=700.0, probe=probe, submitter=sub,
                   reroute_after_s=600.0, dry_run=False)
    self.assertEqual(sub.cancels, [])                     # never cancel blind
    self.assertEqual(e.state, R.JobState.SUBMITTED)

  def test_terminal_marks_failed(self):
    e = _submitted('j1', '111', 'yulpptr', submitted_at=0.0)
    probe = _FakeProbe({'111': RC.STATUS_TERMINAL})
    sub = _FakeSubmitter()
    RC.run_reroute([e], now=700.0, probe=probe, submitter=sub,
                   reroute_after_s=600.0, dry_run=False)
    self.assertEqual(sub.cancels, [])
    self.assertEqual(e.state, R.JobState.FAILED)

  def test_cancel_failure_leaves_submitted(self):
    e = _submitted('j1', '111', 'yulpptr', submitted_at=0.0)
    probe = _FakeProbe({'111': RC.STATUS_PENDING})
    class _FailCancel(_FakeSubmitter):
      def cancel(self, xid):
        self.cancels.append(xid)
        return False, 'xmanager stop failed'
    sub = _FailCancel()
    RC.run_reroute([e], now=700.0, probe=probe, submitter=sub,
                   reroute_after_s=600.0, dry_run=False)
    self.assertEqual(sub.cancels, ['111'])
    self.assertEqual(e.state, R.JobState.SUBMITTED)      # not re-queued on failure

  def test_queued_job_is_not_a_candidate(self):
    e = _entry('j1')                                      # QUEUED, never submitted
    probe = _FakeProbe({})
    _, log = RC.run_reroute([e], now=10000.0, probe=probe,
                            submitter=_FakeSubmitter(), dry_run=False)
    self.assertTrue(any('no SUBMITTED job past' in l for l in log))


class SubmitterCwdTest(unittest.TestCase):
  """The real Submitter, exercised only on its input-validation path (no shell):
  a non-existent workdir must be refused BEFORE any packaging happens."""

  def test_nonexistent_workdir_refused(self):
    sub = RC.Submitter()
    xid, out = sub.submit(['tpu', 'queue', '--tpu_type=v7-32'],
                          cwd='/no/such/checkout/dir')
    self.assertIsNone(xid)
    self.assertIn('workdir does not exist', out)


if __name__ == '__main__':
  unittest.main()
