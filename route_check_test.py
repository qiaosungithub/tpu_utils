"""Unit tests for the router tick. Fake provider + fake submitter: no RPC, no
shell, no real queue file except a temp round-trip."""

import os
import tempfile
import json
import subprocess
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


class ConcurrentWriteTest(unittest.TestCase):
  """The concurrency bug: a route tick that load...RPC...save'd the WHOLE queue
  clobbered rows enqueued during the RPC window. These tests pin the fix
  (merge_and_save_touched + the queue lock) and include a negative control that
  the naive whole-overwrite still loses the row -- so the test can actually fail.
  """

  def _queue_path(self):
    path = tempfile.mkstemp(suffix='.json')[1]
    self.addCleanup(os.remove, path)
    # Clean up the sidecar lockfile too.
    self.addCleanup(lambda: os.path.exists(path + '.lock') and
                    os.remove(path + '.lock'))
    return path

  def test_merge_preserves_concurrently_enqueued_row(self):
    # The exact scenario: a tick snapshots the queue (holding [a]), an enqueue
    # lands [a, b] during the RPC window, then the tick writes back its touched
    # copy of [a]. With merge_and_save_touched, b MUST survive.
    path = self._queue_path()
    RC.save_queue(path, [_entry('a', priority=6)])
    snapshot = RC.load_queue(path)            # tick's in-memory snapshot: [a]
    # ... concurrent enqueue lands during the (simulated) RPC window ...
    with RC.with_queue_lock(path):
      live = RC.load_queue(path)
      live.append(_entry('b', priority=0))
      RC.save_queue(path, live)               # queue now [a, b]
    # ... tick finishes and merge-writes its touched snapshot ([a] mutated) ...
    snapshot[0].last_reason = 'routed this tick'
    RC.merge_and_save_touched(path, snapshot)
    back = {e.job_id: e for e in RC.load_queue(path)}
    self.assertIn('b', back, 'concurrently enqueued row was clobbered')
    self.assertIn('a', back)
    self.assertEqual(back['a'].last_reason, 'routed this tick',
                     'touched row must carry the tick mutation')
    self.assertEqual(back['b'].priority, 0)

  def test_negative_control_whole_overwrite_loses_row(self):
    # Prove the test is real: the OLD pattern (whole-queue save of the stale
    # snapshot) DOES drop the concurrently enqueued row.
    path = self._queue_path()
    RC.save_queue(path, [_entry('a', priority=6)])
    snapshot = RC.load_queue(path)            # [a]
    with RC.with_queue_lock(path):
      live = RC.load_queue(path)
      live.append(_entry('b', priority=0))
      RC.save_queue(path, live)               # [a, b]
    RC.save_queue(path, snapshot)             # OLD BUG: overwrite with stale [a]
    back = {e.job_id: e for e in RC.load_queue(path)}
    self.assertNotIn('b', back,
                     'negative control should reproduce the clobber')

  def test_merge_drops_removed_ids(self):
    path = self._queue_path()
    RC.save_queue(path, [_entry('a'), _entry('b'), _entry('c')])
    entries = RC.load_queue(path)
    RC.merge_and_save_touched(path, entries, dropped_job_ids={'b'})
    self.assertEqual([e.job_id for e in RC.load_queue(path)], ['a', 'c'])

  def test_concurrent_enqueue_processes_lose_nothing(self):
    # The real acceptance test: fork N processes that each enqueue a distinct id
    # concurrently, while a route-tick-style merge runs in the parent. Every
    # enqueued id must be present at the end -- 0 lost.
    import multiprocessing
    path = self._queue_path()
    RC.save_queue(path, [_entry('seed', priority=9)])

    n = 12

    def _enqueuer(i):
      with RC.with_queue_lock(path):
        entries = RC.load_queue(path)
        entries.append(_entry(f'job{i}', priority=0))
        RC.save_queue(path, entries)

    procs = [multiprocessing.Process(target=_enqueuer, args=(i,))
             for i in range(n)]
    for p in procs:
      p.start()
    # Meanwhile the parent runs several route-tick-style merge-writes on stale
    # snapshots -- exactly the operation that used to clobber enqueues.
    for _ in range(5):
      snap = RC.load_queue(path)
      for e in snap:
        e.last_reason = 'tick'
      RC.merge_and_save_touched(path, snap)
    for p in procs:
      p.join(timeout=30)
    got = {e.job_id for e in RC.load_queue(path)}
    missing = {f'job{i}' for i in range(n)} - got
    self.assertEqual(missing, set(),
                     f'{len(missing)} concurrently enqueued rows were lost')
    self.assertIn('seed', got)


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


class _BudgetRefusedSubmitter:
  """submit() returns no xid but WITH the budget marker (over-bar refusal)."""

  def __init__(self):
    self.calls = []
    self.cwds = []

  def submit(self, argv, cwd=''):
    self.calls.append(argv)
    self.cwds.append(cwd)
    return None, ('[budget check] total projected: 9999 (Limit: 2228)\n'
                  '[[BUDGET_DEFERRED]]\n'
                  '[budget check] ERROR: Budget exceeded for tpu check!')

  def cancel(self, xid):
    """Part of the _Submitter protocol; never exercised by these tests."""
    raise AssertionError(f'cancel({xid}) must not be called in this test')


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
                   reroute_after_s=600.0, cooldown_s=1800.0, dry_run=False,
                   sleep_fn=lambda _: None)
    self.assertEqual(sub.cancels, ['111'])               # cancelled
    self.assertEqual(e.state, R.JobState.QUEUED)         # back to queue
    self.assertIsNone(e.xid)
    self.assertGreater(e.cooldown_cells.get('yulpptr', 0), 700.0)  # cell cooled

  def test_dry_run_does_not_cancel(self):
    e = _submitted('j1', '111', 'yulpptr', submitted_at=0.0)
    probe = _FakeProbe({'111': RC.STATUS_PENDING})
    sub = _FakeSubmitter()
    _, log = RC.run_reroute([e], now=700.0, probe=probe, submitter=sub,
                            reroute_after_s=600.0, dry_run=True,
                            sleep_fn=lambda _: None)
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
                   reroute_after_s=600.0, dry_run=False, sleep_fn=lambda _: None)
    self.assertEqual(sub.cancels, ['111'])
    self.assertEqual(e.state, R.JobState.SUBMITTED)      # not re-queued on failure

  def test_queued_job_is_not_a_candidate(self):
    e = _entry('j1')                                      # QUEUED, never submitted
    probe = _FakeProbe({})
    _, log = RC.run_reroute([e], now=10000.0, probe=probe,
                            submitter=_FakeSubmitter(), dry_run=False)
    self.assertTrue(any('no SUBMITTED job past' in l for l in log))


class _SeqProbe:
  """Returns a scripted SEQUENCE of STATUS_* per xid, one per call -- so a test
  can make the first probe PENDING and the second RUNNING (the shadow gap)."""

  def __init__(self, seq_by_xid):
    self._seq = {k: list(v) for k, v in seq_by_xid.items()}
    self.calls = {}

  def status(self, xid):
    self.calls[xid] = self.calls.get(xid, 0) + 1
    seq = self._seq.get(xid, [])
    if not seq:
      return RC.STATUS_UNKNOWN
    return seq.pop(0) if len(seq) > 1 else seq[0]  # last value sticks


class _FakeOutputProbe:
  """Returns a scripted latest_mtime per xid (or None = no disk evidence)."""

  def __init__(self, mtime_by_xid):
    self._by_xid = mtime_by_xid

  def latest_mtime(self, entry):
    return self._by_xid.get(entry.xid)


class RerouteHardeningTest(unittest.TestCase):
  """The 2026-08-24 guards: a single PENDING snapshot must not cancel a job that
  is actually alive (BATCH EMA shadow-WU gap -- xid 282605596)."""

  def _no_sleep(self, _):
    pass

  def test_shadow_gap_second_probe_running_is_not_rerouted(self):
    # First probe PENDING (caught in a shadow gap), second probe RUNNING.
    e = _submitted('j1', '111', 'yutulpz', submitted_at=0.0)
    probe = _SeqProbe({'111': [RC.STATUS_PENDING, RC.STATUS_RUNNING]})
    sub = _FakeSubmitter()
    _, log = RC.run_reroute([e], now=700.0, probe=probe, submitter=sub,
                            reroute_after_s=600.0, dry_run=False,
                            output_probe=_FakeOutputProbe({}),  # no disk evidence
                            confirm_gap_s=15.0, sleep_fn=self._no_sleep)
    self.assertEqual(sub.cancels, [])                    # NOT cancelled
    self.assertEqual(e.state, R.JobState.SUBMITTED)
    self.assertEqual(probe.calls['111'], 2)              # took the 2nd sample
    self.assertTrue(any('2nd probe' in l for l in log))

  def test_fresh_output_is_not_rerouted_even_if_pending(self):
    # Both probes would say PENDING, but disk shows a write 60s ago -> alive.
    e = _submitted('j1', '222', 'yutulpz', submitted_at=0.0)
    probe = _SeqProbe({'222': [RC.STATUS_PENDING, RC.STATUS_PENDING]})
    sub = _FakeSubmitter()
    _, log = RC.run_reroute([e], now=700.0, probe=probe, submitter=sub,
                            reroute_after_s=600.0, dry_run=False,
                            output_probe=_FakeOutputProbe({'222': 640.0}),  # 60s ago
                            fresh_output_s=1200.0, sleep_fn=self._no_sleep)
    self.assertEqual(sub.cancels, [])                    # NOT cancelled
    self.assertEqual(e.state, R.JobState.SUBMITTED)
    self.assertEqual(probe.calls['222'], 1)              # short-circuited before 2nd probe
    self.assertTrue(any('FRESH output' in l for l in log))

  def test_both_pending_no_fresh_output_is_rerouted(self):
    # The genuine stuck case: two PENDING samples, stale output -> DO reroute.
    e = _submitted('j1', '333', 'yutulpz', submitted_at=0.0)
    probe = _SeqProbe({'333': [RC.STATUS_PENDING, RC.STATUS_PENDING]})
    sub = _FakeSubmitter()
    RC.run_reroute([e], now=700.0, probe=probe, submitter=sub,
                   reroute_after_s=600.0, cooldown_s=1800.0, dry_run=False,
                   output_probe=_FakeOutputProbe({'333': None}),  # no output
                   confirm_gap_s=15.0, sleep_fn=self._no_sleep)
    self.assertEqual(sub.cancels, ['333'])               # cancelled (correct)
    self.assertEqual(e.state, R.JobState.QUEUED)
    self.assertIsNone(e.xid)

  def test_boundary_stale_output_plus_second_pending_still_reroutes(self):
    # v26 edge: output mtime EXACTLY at the freshness boundary (=stale) AND the
    # second probe also PENDING -> must STILL reroute (guards near-miss, not a
    # permanent shield). fresh_output_s=1200, write was exactly 1200s ago.
    e = _submitted('j1', '444', 'yutulpz', submitted_at=0.0)
    probe = _SeqProbe({'444': [RC.STATUS_PENDING, RC.STATUS_PENDING]})
    sub = _FakeSubmitter()
    RC.run_reroute([e], now=2000.0, probe=probe, submitter=sub,
                   reroute_after_s=600.0, cooldown_s=1800.0, dry_run=False,
                   output_probe=_FakeOutputProbe({'444': 800.0}),  # 1200s ago == boundary
                   fresh_output_s=1200.0, confirm_gap_s=15.0,
                   sleep_fn=self._no_sleep)
    self.assertEqual(sub.cancels, ['444'])               # boundary=stale -> reroute
    self.assertEqual(e.state, R.JobState.QUEUED)

  def test_terminal_zombie_cleanup_unaffected_by_hardening(self):
    # v26 req 2: TERMINAL->FAILED path must NOT go through the new guards.
    e = _submitted('j1', '555', 'yutulpz', submitted_at=0.0)
    probe = _SeqProbe({'555': [RC.STATUS_TERMINAL]})
    sub = _FakeSubmitter()
    RC.run_reroute([e], now=700.0, probe=probe, submitter=sub,
                   reroute_after_s=600.0, dry_run=False,
                   output_probe=_FakeOutputProbe({'555': 690.0}),  # even fresh output
                   sleep_fn=self._no_sleep)
    self.assertEqual(sub.cancels, [])
    self.assertEqual(e.state, R.JobState.FAILED)         # zombie still cleaned
    self.assertEqual(probe.calls['555'], 1)              # no 2nd probe for TERMINAL


class SubmitterCwdTest(unittest.TestCase):
  """The real Submitter, exercised only on its input-validation path (no shell):
  a non-existent workdir must be refused BEFORE any packaging happens."""

  def test_nonexistent_workdir_refused(self):
    sub = RC.Submitter()
    xid, out = sub.submit(['tpu', 'queue', '--tpu_type=v7-32'],
                          cwd='/no/such/checkout/dir')
    self.assertIsNone(xid)
    self.assertIn('workdir does not exist', out)


class _StaleProbe:
  """Scripted srcfs failure counter for the mode-2 brake test."""

  def __init__(self, counts):
    self._counts = list(counts)
    self._i = 0

  def failure_count(self):
    v = self._counts[min(self._i, len(self._counts) - 1)]
    self._i += 1
    return v


class SerialWorkerTest(unittest.TestCase):

  def setUp(self):
    self.path = tempfile.mkstemp(suffix='.json')[1]
    self.addCleanup(lambda: os.path.exists(self.path) and os.remove(self.path))
    lock = self.path + '.lock'
    self.addCleanup(lambda: os.path.exists(lock) and os.remove(lock))

  def _seed(self, entries):
    RC.save_queue(self.path, entries)

  def _load(self):
    return RC.load_queue(self.path)

  def _byid(self, jid):
    return {e.job_id: e for e in self._load()}[jid]

  def _prov(self, cell='yutulpz', arch='v7', free=320):
    return _FakeProvider({f'{cell}|{arch}': _avail(cell, arch, free)},
                         arch_price={arch: 20.0}, arch_pool={arch: free})

  def test_claims_one_and_submits(self):
    self._seed([_entry('a', power='v7-32', archs=('v7',))])
    sub = _FakeSubmitter(xid='900')
    outcome, log, _ = RC.run_worker_once(
        self.path, self._prov(), sub, now=100.0, worker_id='w1')
    self.assertEqual(outcome, 'submitted')
    e = self._byid('a')
    self.assertEqual(e.state, R.JobState.SUBMITTED)
    self.assertEqual(e.xid, '900')
    self.assertIsNone(e.build_started_at)         # slot released

  def test_single_build_invariant_blocks_second_claim(self):
    # one already BUILDING (live) + one QUEUED -> worker must NOT start a 2nd
    e_bld = _entry('bld', power='v7-32', archs=('v7',))
    e_bld.state = R.JobState.BUILDING
    e_bld.build_started_at = 99.0
    e_q = _entry('q', power='v7-32', archs=('v7',))
    self._seed([e_bld, e_q])
    sub = _FakeSubmitter(xid='901')
    outcome, log, _ = RC.run_worker_once(
        self.path, self._prov(), sub, now=100.0, worker_id='w2',
        build_stale_s=1800.0)
    self.assertEqual(outcome, 'busy')
    self.assertEqual(sub.calls, [])               # nothing built
    self.assertEqual(self._byid('q').state, R.JobState.QUEUED)  # still queued

  def test_stale_building_is_reclaimed_then_claimed(self):
    e_bld = _entry('old', power='v7-32', archs=('v7',))
    e_bld.state = R.JobState.BUILDING
    e_bld.build_started_at = 0.0                   # ancient
    self._seed([e_bld])
    sub = _FakeSubmitter(xid='902')
    outcome, _, _ = RC.run_worker_once(
        self.path, self._prov(), sub, now=5000.0, worker_id='w3',
        build_stale_s=1800.0)
    # stale claim reclaimed -> then the same entry (now QUEUED) is built
    self.assertEqual(outcome, 'submitted')
    self.assertEqual(self._byid('old').state, R.JobState.SUBMITTED)

  def test_no_xid_requeues_not_submitted(self):
    self._seed([_entry('z', power='v7-32', archs=('v7',))])
    sub = _FakeSubmitter(xid=None)                 # found[]/build crash
    outcome, log, _ = RC.run_worker_once(
        self.path, self._prov(), sub, now=100.0, worker_id='w4')
    self.assertEqual(outcome, 'requeued')
    e = self._byid('z')
    self.assertEqual(e.state, R.JobState.QUEUED)   # NOT submitted
    self.assertEqual(e.attempts, 1)
    self.assertIsNone(e.build_started_at)          # slot released

  def test_nothing_placeable_requeues(self):
    self._seed([_entry('p', power='v7-32', archs=('v7',))])
    prov = _FakeProvider({'x|v7': _avail('x', 'v7', 320, oversold=True)},
                         arch_price={'v7': 20.0}, arch_pool={'v7': 0})
    sub = _FakeSubmitter(xid='903')
    outcome, _, _ = RC.run_worker_once(
        self.path, prov, sub, now=100.0, worker_id='w5')
    self.assertEqual(outcome, 'requeued')
    self.assertEqual(sub.calls, [])
    self.assertEqual(self._byid('p').state, R.JobState.QUEUED)

  def test_idle_when_empty(self):
    self._seed([])
    outcome, _, _ = RC.run_worker_once(
        self.path, self._prov(), _FakeSubmitter(), now=0.0, worker_id='w6')
    self.assertEqual(outcome, 'idle')

  def test_srcfs_brake_skips_when_failures_spike(self):
    self._seed([_entry('b', power='v7-32', archs=('v7',))])
    sub = _FakeSubmitter(xid='904')
    probe = _StaleProbe([100, 130])   # +30 between polls (>= 20 brake)
    # first poll establishes baseline (100), no brake, builds
    o1, _, fc1 = RC.run_worker_once(
        self.path, self._prov(), sub, now=100.0, worker_id='w7',
        stage_probe=probe, srcfs_fail_brake=20, last_fail_count=None)
    self.assertEqual(o1, 'submitted')
    # second poll sees +30 -> brake
    self._seed([_entry('b2', power='v7-32', archs=('v7',))])
    o2, log2, _ = RC.run_worker_once(
        self.path, self._prov(), sub, now=200.0, worker_id='w7',
        stage_probe=probe, srcfs_fail_brake=20, last_fail_count=fc1)
    self.assertEqual(o2, 'braked')
    self.assertTrue(any('BRAKE' in l for l in log2))

  def test_worker_loop_bounded(self):
    self._seed([_entry('a', power='v7-32', archs=('v7',)),
                _entry('b', power='v7-32', archs=('v7',))])
    sub = _FakeSubmitter(xid='905')
    RC.run_worker_loop(
        self.path, provider_factory=lambda: self._prov(),
        submitter=sub, worker_id='wl', poll_s=0.0, max_iterations=2)
    # both drained to SUBMITTED across 2 iterations (serial)
    states = {e.job_id: e.state for e in self._load()}
    self.assertEqual(states['a'], R.JobState.SUBMITTED)
    self.assertEqual(states['b'], R.JobState.SUBMITTED)

  def test_nonexistent_workdir_is_HELD_not_churned(self):
    e = _entry('bad', power='v7-32', archs=('v7',))
    e.workdir = '/no/such/checkout/xyz'
    self._seed([e])
    sub = _FakeSubmitter(xid='906')
    outcome, log, _ = RC.run_worker_once(
        self.path, self._prov(), sub, now=100.0, worker_id='w')
    self.assertEqual(outcome, 'held')
    self.assertEqual(sub.calls, [])                 # never built
    self.assertEqual(self._byid('bad').state, R.JobState.HELD)

  def test_HELD_job_is_not_claimed_again(self):
    # a HELD job must be skipped; a QUEUED sibling is built instead
    e_held = _entry('held', power='v7-32', archs=('v7',))
    e_held.state = R.JobState.HELD
    e_ok = _entry('ok', power='v7-32', archs=('v7',))
    self._seed([e_held, e_ok])
    sub = _FakeSubmitter(xid='907')
    outcome, _, _ = RC.run_worker_once(
        self.path, self._prov(), sub, now=100.0, worker_id='w')
    self.assertEqual(outcome, 'submitted')
    self.assertEqual(self._byid('ok').state, R.JobState.SUBMITTED)
    self.assertEqual(self._byid('held').state, R.JobState.HELD)  # untouched

  def test_max_attempts_moves_to_HELD_not_infinite_requeue(self):
    e = _entry('z', power='v7-32', archs=('v7',))
    self._seed([e])
    sub = _FakeSubmitter(xid=None)                  # every build fails (no XID)
    # attempts: 0->1 (requeue), 1->2 (requeue), 2->3 (>=3 -> HELD)
    for expected in ('requeued', 'requeued', 'held'):
      o, _, _ = RC.run_worker_once(
          self.path, self._prov(), sub, now=100.0, worker_id='w',
          max_build_attempts=3)
      self.assertEqual(o, expected)
    self.assertEqual(self._byid('z').state, R.JobState.HELD)
    self.assertEqual(self._byid('z').attempts, 3)

  def test_requeue_held_via_helper(self):
    e = _entry('h', power='v7-32', archs=('v7',))
    e.state = R.JobState.HELD
    e.attempts = 5
    R.requeue_held(e)
    self.assertEqual(e.state, R.JobState.QUEUED)
    self.assertEqual(e.attempts, 0)

  # --- Step3 R2: budget refusal is NOT a build failure ---
  def test_budget_deferral_marker_parks_not_attempts(self):
    # submit returns NO xid but WITH the [[BUDGET_DEFERRED]] marker -> the job
    # must land BUDGET_DEFERRED with attempts UNCHANGED (not treated as a
    # build failure). This is the R2 fix.
    e = _entry('bd', power='v7-32', archs=('v7',))
    self._seed([e])
    sub = _BudgetRefusedSubmitter()
    outcome, log, _ = RC.run_worker_once(
        self.path, self._prov(), sub, now=100.0, worker_id='w')
    self.assertEqual(outcome, 'budget_deferred')
    got = self._byid('bd')
    self.assertEqual(got.state, R.JobState.BUDGET_DEFERRED)
    self.assertEqual(got.attempts, 0)              # NOT incremented
    self.assertIsNone(got.build_started_at)         # slot released

  def test_budget_deferral_never_becomes_held(self):
    # Even after many over-budget rounds, a budget-refused job never accrues
    # attempts toward HELD (contrast test_max_attempts_moves_to_HELD).
    e = _entry('bd2', power='v7-32', archs=('v7',))
    self._seed([e])
    sub = _BudgetRefusedSubmitter()
    for _ in range(5):
      # promote back to QUEUED (as the top-of-round would) then re-run
      cur = self._byid('bd2')
      if cur.state == R.JobState.BUDGET_DEFERRED:
        R.promote_deferred([cur]); RC.save_queue(self.path, [cur])
      o, _, _ = RC.run_worker_once(
          self.path, self._prov(), sub, now=100.0, worker_id='w',
          max_build_attempts=3)
      self.assertEqual(o, 'budget_deferred')
    self.assertEqual(self._byid('bd2').attempts, 0)
    self.assertNotEqual(self._byid('bd2').state, R.JobState.HELD)

  def test_real_build_failure_still_attempts(self):
    # a no-XID WITHOUT the marker is still a real failure -> attempts++ (the
    # existing MODE-1 GUARD path is unchanged by the R2 fix).
    e = _entry('rf', power='v7-32', archs=('v7',))
    self._seed([e])
    sub = _FakeSubmitter(xid=None)                  # no marker, no xid
    outcome, _, _ = RC.run_worker_once(
        self.path, self._prov(), sub, now=100.0, worker_id='w')
    self.assertEqual(outcome, 'requeued')
    self.assertEqual(self._byid('rf').attempts, 1)  # real failure counts


class IsBudgetDeferralTest(unittest.TestCase):
  """Step3 R2: the [[BUDGET_DEFERRED]] marker parser."""

  def test_plain_marker(self):
    self.assertTrue(RC.is_budget_deferral('foo\n[[BUDGET_DEFERRED]]\nbar'))

  def test_ansi_wrapped_marker(self):
    self.assertTrue(RC.is_budget_deferral('\x1b[31m[[BUDGET_DEFERRED]]\x1b[0m'))

  def test_absent(self):
    self.assertFalse(RC.is_budget_deferral('Launched experiment 123'))
    self.assertFalse(RC.is_budget_deferral(''))
    self.assertFalse(RC.is_budget_deferral(''))  # None-ish input, typed as str

  def test_substring_not_matched(self):
    # must be its OWN line, not embedded in prose (avoid false positives).
    self.assertFalse(RC.is_budget_deferral('note: [[BUDGET_DEFERRED]] was seen'))


class RunDispatchTest(unittest.TestCase):
  """Step3: run_dispatch_once (router half) -- promote/backpressure/greedy."""

  def setUp(self):
    self.path = tempfile.mkstemp(suffix='.json')[1]
    self.addCleanup(lambda: os.path.exists(self.path) and os.remove(self.path))
    lock = self.path + '.lock'
    self.addCleanup(lambda: os.path.exists(lock) and os.remove(lock))

  def _seed(self, entries):
    RC.save_queue(self.path, entries)

  def _byid(self, jid):
    return {e.job_id: e for e in RC.load_queue(self.path)}[jid]

  def _budget(self, headroom, per_cost, exempt_types=()):
    """Fake budget_query_fn: fixed headroom; new_cost from per_cost by type."""
    def fn(tpu_type, tier='PROD', lo='', group=''):
      return {'income': 1000.0, 'bar': 100.0, 'current': 100.0 - headroom,
              'headroom': headroom, 'new_cost': per_cost.get(tpu_type, 0.0),
              'exempt': tpu_type in exempt_types, 'fits': True}
    return fn

  def _q(self, jid, power='v7-32', archs=('v7',), priority=0):
    e = _entry(jid, power=power, archs=archs)
    e.state = R.JobState.QUEUED
    e.priority = priority
    return e

  def test_promote_then_dispatch(self):
    d = self._q('d'); d.state = R.JobState.BUDGET_DEFERRED
    self._seed([d])
    out, log = RC.run_dispatch_once(
        self.path, now=100.0, budget_query_fn=self._budget(1000.0, {'v7-32': 10.0}),
        dry_run=False)
    self.assertEqual(out, 'dispatched')
    self.assertEqual(self._byid('d').state, R.JobState.BUILD_REQUESTED)

  def test_backpressure_skips_dispatch(self):
    br = self._q('br'); br.state = R.JobState.BUILD_REQUESTED
    q = self._q('q')
    self._seed([br, q])
    out, log = RC.run_dispatch_once(
        self.path, now=100.0, budget_query_fn=self._budget(1000.0, {}),
        dry_run=False)
    self.assertEqual(out, 'backpressure')
    self.assertEqual(self._byid('q').state, R.JobState.QUEUED)  # not dispatched

  def test_greedy_marks_fit_and_defer(self):
    a = self._q('a', priority=2); b = self._q('b', priority=1)
    self._seed([a, b])
    # headroom 100, each costs 60 -> a fits (60), b deferred (60 > 40 left)
    out, log = RC.run_dispatch_once(
        self.path, now=100.0,
        budget_query_fn=self._budget(100.0, {'v7-32': 60.0}), dry_run=False)
    self.assertEqual(self._byid('a').state, R.JobState.BUILD_REQUESTED)
    self.assertEqual(self._byid('b').state, R.JobState.BUDGET_DEFERRED)

  def test_no_budget_fails_safe(self):
    self._seed([self._q('a')])
    out, log = RC.run_dispatch_once(
        self.path, now=100.0, budget_query_fn=lambda *a, **k: None,
        dry_run=False)
    self.assertEqual(out, 'no-budget')
    self.assertEqual(self._byid('a').state, R.JobState.QUEUED)  # untouched, safe

  def test_dry_run_does_not_mutate(self):
    self._seed([self._q('a')])
    out, log = RC.run_dispatch_once(
        self.path, now=100.0, budget_query_fn=self._budget(1000.0, {'v7-32': 1.0}),
        dry_run=True)
    self.assertEqual(self._byid('a').state, R.JobState.QUEUED)  # unchanged
    self.assertTrue(any('DRY' in l for l in log))

  def test_idle_when_no_queued(self):
    r = self._q('r'); r.state = R.JobState.RUNNING
    self._seed([r])
    out, log = RC.run_dispatch_once(
        self.path, now=100.0, budget_query_fn=self._budget(1000.0, {}),
        dry_run=False)
    self.assertEqual(out, 'idle')


class GroupOrderTest(unittest.TestCase):
  """The place pass tries groups in preference order (e.g. vqfree g5 then g9).

  The loop in _run applies run_tick once per group, each with that group's own
  availability; a job placed by an earlier group is SUBMITTED and the next
  group's tick only sees the QUEUED remainder. These tests pin that composition
  invariant (which is what --group_order relies on) at the run_tick level.
  """

  def _run_group_order(self, entries, providers_by_group, group_order):
    """Mirror _run's place loop: sequential run_tick per group, live submit."""
    submitters = {}
    updated = entries
    for grp in group_order:
      remaining = [e for e in updated if e.state == R.JobState.QUEUED]
      if not remaining:
        break
      sub = _FakeSubmitter(xid=f'xid-{grp}')
      submitters[grp] = sub
      updated, _ = RC.run_tick(
          updated, providers_by_group[grp], now=100.0, submitter=sub,
          dry_run=False, group=grp)
    return updated, submitters

  def test_prefers_first_group_when_it_can_place(self):
    # A job placeable in BOTH g5 and g9 must be taken by g5 (tried first); g9
    # must never see it.
    e = _entry('j1', power='v7-32', archs=('v7',))
    prov5 = _FakeProvider({'yutulpz|v7': _avail('yutulpz', 'v7', 320)},
                          arch_price={'v7': 20.0}, arch_pool={'v7': 320})
    prov9 = _FakeProvider({'yudfwra|v7': _avail('yudfwra', 'v7', 320)},
                          arch_price={'v7': 20.0}, arch_pool={'v7': 320})
    _out, subs = self._run_group_order(
        [e], {'5': prov5, '9': prov9}, ['5', '9'])
    self.assertEqual(len(subs['5'].calls), 1, 'g5 should place the job')
    self.assertNotIn('9', subs, 'g9 tick should be skipped: nothing left QUEUED')
    self.assertEqual(e.state, R.JobState.SUBMITTED)
    self.assertEqual(e.xid, 'xid-5')

  def test_falls_back_to_second_group_when_first_cannot_place(self):
    # g5 has NO availability; the job must fall through to g9.
    e = _entry('j1', power='v7-32', archs=('v7',))
    prov5 = _FakeProvider({})  # vqfree empty this tick
    prov9 = _FakeProvider({'yudfwra|v7': _avail('yudfwra', 'v7', 320)},
                          arch_price={'v7': 20.0}, arch_pool={'v7': 320})
    _out, subs = self._run_group_order(
        [e], {'5': prov5, '9': prov9}, ['5', '9'])
    self.assertEqual(subs['5'].calls, [], 'g5 cannot place (no avail)')
    self.assertEqual(len(subs['9'].calls), 1, 'g9 should place the fallback')
    self.assertEqual(e.state, R.JobState.SUBMITTED)
    self.assertEqual(e.xid, 'xid-9')

  def test_split_batch_partly_g5_partly_g9(self):
    # Two jobs, g5 can seat only one slice; the other must fall to g9. Neither
    # is lost, neither double-placed.
    e1 = _entry('j1', power='v7-32', archs=('v7',))
    e2 = _entry('j2', power='v7-32', archs=('v7',))
    prov5 = _FakeProvider({'yutulpz|v7': _avail('yutulpz', 'v7', 32)},
                          arch_price={'v7': 20.0}, arch_pool={'v7': 32})
    prov9 = _FakeProvider({'yudfwra|v7': _avail('yudfwra', 'v7', 320)},
                          arch_price={'v7': 20.0}, arch_pool={'v7': 320})
    out, subs = self._run_group_order(
        [e1, e2], {'5': prov5, '9': prov9}, ['5', '9'])
    placed_g5 = sum(len(subs[g].calls) for g in subs if g == '5')
    placed_g9 = sum(len(subs[g].calls) for g in subs if g == '9')
    self.assertEqual(placed_g5, 1, 'g5 seats exactly one slice')
    self.assertEqual(placed_g9, 1, 'the other falls to g9')
    self.assertTrue(all(e.state == R.JobState.SUBMITTED for e in out))
    self.assertEqual({e.xid for e in out}, {'xid-5', 'xid-9'})


class RunReconcileTest(unittest.TestCase):
  """Step2: XM-truth reconcile pass (run_reconcile) -- R3 zombie cleanup."""

  def _running(self, job_id, xid):
    e = _submitted(job_id, xid, 'yulpptr', submitted_at=0.0)
    e.state = R.JobState.RUNNING
    return e

  def test_zombie_running_marked_failed(self):
    # local RUNNING but XM says terminal -> the 91%-zombie case.
    e = self._running('z1', '111')
    probe = _FakeProbe({'111': RC.STATUS_TERMINAL})
    out, log = RC.run_reconcile([e], now=100.0, probe=probe, dry_run=False)
    self.assertEqual(e.state, R.JobState.FAILED)
    self.assertTrue(any('zombie' in l for l in log))

  def test_submitted_promoted_to_running(self):
    e = _submitted('p1', '222', 'yulpptr', submitted_at=0.0)
    probe = _FakeProbe({'222': RC.STATUS_RUNNING})
    RC.run_reconcile([e], now=100.0, probe=probe, dry_run=False)
    self.assertEqual(e.state, R.JobState.RUNNING)

  def test_unknown_never_acts(self):
    # THE safety rule: a probe hiccup must never mark a live job dead.
    e = self._running('u1', '333')
    probe = _FakeProbe({'333': RC.STATUS_UNKNOWN})
    RC.run_reconcile([e], now=100.0, probe=probe, dry_run=False)
    self.assertEqual(e.state, R.JobState.RUNNING)

  def test_pending_left_for_reroute(self):
    # reconcile leaves genuinely-pending SUBMITTED for the reroute step.
    e = _submitted('q1', '444', 'yulpptr', submitted_at=0.0)
    probe = _FakeProbe({'444': RC.STATUS_PENDING})
    RC.run_reconcile([e], now=100.0, probe=probe, dry_run=False)
    self.assertEqual(e.state, R.JobState.SUBMITTED)

  def test_dry_run_does_not_mutate(self):
    e = self._running('z2', '555')
    probe = _FakeProbe({'555': RC.STATUS_TERMINAL})
    _, log = RC.run_reconcile([e], now=100.0, probe=probe, dry_run=True)
    self.assertEqual(e.state, R.JobState.RUNNING)          # untouched
    self.assertTrue(any('would set' in l for l in log))

  def test_no_xid_skipped(self):
    e = _entry('b1')
    e.state = R.JobState.BUILDING            # BUILDING with no xid
    e.xid = None
    probe = _FakeProbe({})
    RC.run_reconcile([e], now=100.0, probe=probe, dry_run=False)
    self.assertEqual(e.state, R.JobState.BUILDING)         # no XM identity

  def test_terminal_entries_not_touched(self):
    # QUEUED/HELD/DONE/FAILED are not in RECONCILABLE_STATES -> never probed.
    q = _entry('q'); q.state = R.JobState.QUEUED
    h = _entry('h'); h.state = R.JobState.HELD
    probe = _FakeProbe({})
    out, log = RC.run_reconcile([q, h], now=100.0, probe=probe, dry_run=False)
    self.assertEqual(q.state, R.JobState.QUEUED)
    self.assertEqual(h.state, R.JobState.HELD)
    self.assertTrue(any('no non-terminal entries' in l for l in log))

  def test_summary_counts(self):
    zombie = self._running('z', 'z1')
    promo = _submitted('p', 'p1', 'c', submitted_at=0.0)
    unk = self._running('u', 'u1')
    probe = _FakeProbe({'z1': RC.STATUS_TERMINAL, 'p1': RC.STATUS_RUNNING,
                        'u1': RC.STATUS_UNKNOWN})
    _, log = RC.run_reconcile([zombie, promo, unk], now=100.0, probe=probe,
                              dry_run=False)
    summary = [l for l in log if 'checked:' in l]
    self.assertEqual(len(summary), 1)
    self.assertIn('1 zombie->FAILED', summary[0])
    self.assertIn('1 promoted->RUNNING', summary[0])
    self.assertIn('1 UNKNOWN', summary[0])



# --- submit timeout: local failure is not remote absence -------------------
# ★The bug this covers cost real money: `tpu queue` timed out at 1800s, the
# submitter reported "no XID", the worker counted a failed attempt and
# RESUBMITTED -- while the first experiment was already running. Two copies
# billed the same quota, and the orphan had no local row at all.
_PARTIAL_WITH_XID = 'Launched experiment 284946261\nstill building...'


class _FakeTimeout(subprocess.TimeoutExpired):
  def __init__(self, stdout=b'', stderr=b''):
    super().__init__(cmd='tpu queue', timeout=1800.0)
    self.stdout = stdout
    self.stderr = stderr


class SubmitTimeoutRecoveryTest(unittest.TestCase):

  def _submitter(self, name_lookup=None):
    s = RC.Submitter()
    if name_lookup is not None:
      s.find_xid_by_name = name_lookup
    return s

  def test_xid_recovered_from_partial_output(self):
    """Cheapest probe: the id was already printed before the timeout."""
    s = self._submitter(lambda n, **kw: (None, 'should not be reached'))
    xid, out = s._recover_timed_out_xid(
        _FakeTimeout(stdout=_PARTIAL_WITH_XID.encode()),
        ['tpu', 'queue', '--exp_name=job_a'])
    self.assertEqual(xid, '284946261')
    self.assertIn('WAS created', out)

  def test_xid_recovered_from_xm_by_name(self):
    """Timeout landed before the id flushed -> ask XManager by name."""
    s = self._submitter(lambda n, **kw: ('284999999', 'XM lookup matched 1'))
    xid, out = s._recover_timed_out_xid(
        _FakeTimeout(), ['tpu', 'queue', '--exp_name=job_a'])
    self.assertEqual(xid, '284999999')
    self.assertIn('Adopting it', out)

  def test_NC_genuinely_absent_still_reports_no_xid(self):
    """★Negative control: when the job really was not submitted, we must still
    say so -- the fix must not fabricate an XID and strand a QUEUED row."""
    s = self._submitter(lambda n, **kw: (None, 'XM lookup ran and found no exact-name match'))
    xid, out = s._recover_timed_out_xid(
        _FakeTimeout(), ['tpu', 'queue', '--exp_name=job_a'])
    self.assertIsNone(xid)
    self.assertIn('Treating as not-submitted', out)

  def test_NC_unknown_remote_state_is_named_not_guessed(self):
    """No --exp_name -> we cannot check; say UNKNOWN rather than imply failure."""
    s = self._submitter()
    xid, out = s._recover_timed_out_xid(_FakeTimeout(), ['tpu', 'queue'])
    self.assertIsNone(xid)
    self.assertIn('UNKNOWN', out)

  def test_NC_xm_lookup_failure_does_not_claim_absence(self):
    """If the lookup itself failed, that is not evidence the job is absent."""
    s = self._submitter(lambda n, **kw: (None, 'XM lookup itself timed out; remote state UNKNOWN'))
    xid, out = s._recover_timed_out_xid(
        _FakeTimeout(), ['tpu', 'queue', '--exp_name=job_a'])
    self.assertIsNone(xid)
    self.assertIn('UNKNOWN', out)


  def test_submit_ITSELF_recovers_on_timeout(self):
    """★End-to-end through submit(), with a REAL timeout -- no monkeypatching.

    NEGATIVE-CONTROL GAP THIS CLOSES: testing `_recover_timed_out_xid` alone
    still passed when the recovery was ripped out of `submit()`, because
    nothing asserted that submit() actually CALLS it. Here the wrapper is a
    throwaway script that prints the XID and then hangs past the deadline --
    exactly the real shape (experiment created early, build still running).
    """
    d = tempfile.mkdtemp()
    wrapper = os.path.join(d, 'fake_wrapper.sh')
    with open(wrapper, 'w') as f:
      f.write('tpu() { echo "Launched experiment 284946261"; sleep 30; }\n')
    self.addCleanup(lambda: os.path.exists(wrapper) and os.remove(wrapper))

    s = RC.Submitter(wrapper_path=wrapper, timeout_s=1.0)
    xid, out = s.submit(['tpu', 'queue', '--exp_name=job_a'])
    self.assertEqual(xid, '284946261',
                     'submit() must adopt the already-created experiment')
    self.assertIn('WAS created', out)

  def test_exp_name_parsing(self):
    self.assertEqual(RC._exp_name_of(['tpu', '--exp_name=abc']), 'abc')
    self.assertIsNone(RC._exp_name_of(['tpu', 'queue']))


class PriorXidsTest(unittest.TestCase):
  """A resubmitted row must keep its earlier XIDs findable."""

  def _entry(self):
    return R.QueueEntry(job_id='j1', power='h100-8', allowed_archs=['h100'])

  def _placement(self):
    return R.Placement(job_id='j1', cell='sh', arch='h100', chips=8,
                       price=4.0, reason='test', geometry=None)

  def test_superseded_xid_is_preserved(self):
    e = self._entry()
    R.apply_placement(e, self._placement(), xid='111', now=0.0)
    self.assertEqual(e.prior_xids, [])
    R.apply_placement(e, self._placement(), xid='222', now=1.0)
    self.assertEqual(e.xid, '222')
    self.assertEqual(e.prior_xids, ['111'], 'the first XID must remain findable')

  def test_resubmitting_same_xid_is_not_recorded_twice(self):
    e = self._entry()
    R.apply_placement(e, self._placement(), xid='111', now=0.0)
    R.apply_placement(e, self._placement(), xid='111', now=1.0)
    self.assertEqual(e.prior_xids, [])

  def test_history_survives_a_round_trip_through_json(self):
    e = self._entry()
    R.apply_placement(e, self._placement(), xid='111', now=0.0)
    R.apply_placement(e, self._placement(), xid='222', now=1.0)
    back = R.QueueEntry.from_dict(json.loads(json.dumps(e.to_dict())))
    self.assertEqual(back.prior_xids, ['111'])

  def test_old_rows_without_the_field_still_load(self):
    """Backward compatibility: a queue written before this field must load."""
    d = self._entry().to_dict()
    d.pop('prior_xids', None)
    back = R.QueueEntry.from_dict(d)
    self.assertEqual(back.prior_xids, [])

if __name__ == '__main__':
  unittest.main()
