"""Tests for the migrator. The rule under test is "archive by default": a row is carried
forward only if it is both live and intact, and the three output lists always sum to input.
"""

import json
import os
import tempfile
import unittest

try:
  from google3.experimental.users.qiaos.tpu_utils import jobchain as jc
  from google3.experimental.users.qiaos.tpu_utils import jobmigrate as jm
  from google3.experimental.users.qiaos.tpu_utils import jobstore as js
except ImportError:  # pragma: no cover - flat-layout fallback
  import jobchain as jc  # type: ignore
  import jobmigrate as jm  # type: ignore
  import jobstore as js  # type: ignore



def _row(**kw):
  d = dict(job_id='j1', state='QUEUED', workdir='/w',
           # a bucket is required by the poisoned-personal-quota gate
           launch_kwargs={'config': 'c', 'bucket': '/cns/oi-d/home/qiaos/eqr_data'},
           attempts=0, power='v6p-32', allowed_metros=['tul'])
  d.update(kw)
  return d


class Classify(unittest.TestCase):
  def test_enforcer_shell_is_archived(self):
    """15 rows had exactly this shape: a requeue rebuilt launch_kwargs from nothing, so the
    job would cold-start AND be priced at a tenth of reality."""
    v, why = jm.classify(_row(launch_kwargs={'resume_xid': '123'}))
    self.assertEqual(v, jm.Verdict.ARCHIVE)
    self.assertIn('resume_xid', why)

  def test_tmp_workdir_is_archived(self):
    v, _ = jm.classify(_row(workdir='/tmp'))
    self.assertEqual(v, jm.Verdict.ARCHIVE)

  def test_terminal_is_archived(self):
    for st in ('FAILED', 'CANCELLED', 'COMPLETED'):
      self.assertEqual(jm.classify(_row(state=st))[0], jm.Verdict.ARCHIVE)

  def test_xid_without_xm_truth_is_archived_not_guessed(self):
    """★12 SUBMITTED rows were five days dead with their experiments purged. 'Has an xid' is
    not 'is alive' -- refusing to guess is the whole point."""
    v, why = jm.classify(_row(state='SUBMITTED', xid='282431624'), live_xids=None)
    self.assertEqual(v, jm.Verdict.ARCHIVE)
    self.assertIn('refusing to guess', why)

  def test_xid_confirmed_live_is_migrated(self):
    v, _ = jm.classify(_row(state='RUNNING', xid='284380582'),
                       live_xids={'284380582'})
    self.assertEqual(v, jm.Verdict.MIGRATE)

  def test_xid_confirmed_dead_is_archived(self):
    v, why = jm.classify(_row(state='SUBMITTED', xid='999'), live_xids={'284380582'})
    self.assertEqual(v, jm.Verdict.ARCHIVE)
    self.assertIn('not live in XM', why)


class IdentityRecovery(unittest.TestCase):
  """The fingerprint comes from the job's OWN checkout, not from a shared stagedir -- the
  stagedir copy is the one that gets backfilled from a global default and cross-wires lines."""

  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()

  def tearDown(self):
    self.tmp.cleanup()

  def test_reads_fingerprint_from_workdir_config(self):
    open(os.path.join(self.tmp.name, 'config.sh'), 'w').write(
        'export PROJECT_NAME="codi-torch"\nexport TARGET_LABEL="//x/codi:main"\n')
    self.assertEqual(jm.read_identity(self.tmp.name), ('codi-torch', '//x/codi:main'))

  def test_missing_config_yields_no_guess(self):
    self.assertEqual(jm.read_identity(self.tmp.name), (None, None))

  def test_unrecoverable_identity_is_rejected_by_the_gate(self):
    job = jm.to_job(_row(workdir=self.tmp.name))
    self.assertEqual(job.target_label, jc.UNKNOWN)
    with self.assertRaises(jc.RejectedAtEnqueue):
      jc.validate_enqueue(job)


class ToJob(unittest.TestCase):
  def test_attempts_are_not_carried_across(self):
    """★An att=65 row was scored under a taxonomy that counted budget refusals as build
    defects. Carrying the number forward would import the bug's verdict into the fix."""
    j = jm.to_job(_row(attempts=65, state='HELD'))
    self.assertEqual(j.build_attempts, 0)
    self.assertIn('att=65', j.last_reason)      # the history is stated, not silently dropped

  def test_launch_kwargs_and_metros_are_carried_verbatim(self):
    lk = {'config': 'remote_run_config', 'bucket': '/cns/is-d/x'}
    j = jm.to_job(_row(launch_kwargs=lk, allowed_metros=['cbf']))
    self.assertEqual(j.launch_kwargs, lk)
    self.assertEqual(j.allowed_metros, ['cbf'])

  def test_chain_starts_empty(self):
    j = jm.to_job(_row())
    self.assertEqual(j.nodes, [])
    self.assertEqual(j.version, 0)


class Accounting(unittest.TestCase):
  """Every input row lands in exactly one output list. A row that vanishes between formats is
  the failure this rewrite exists to stop, so the sum is asserted rather than assumed."""

  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.wd = os.path.join(self.tmp.name, 'proj')
    os.makedirs(self.wd)
    open(os.path.join(self.wd, 'config.sh'), 'w').write(
        'export PROJECT_NAME="p"\nexport TARGET_LABEL="//p:main"\n')

  def tearDown(self):
    self.tmp.cleanup()

  def _run(self, rows, dry_run=True):
    src = os.path.join(self.tmp.name, 'old.json')
    json.dump({'entries': rows}, open(src, 'w'))
    store = js.JobStore(os.path.join(self.tmp.name, 'new.json'))
    return jm.migrate(src, store, os.path.join(self.tmp.name, 'arch.json'),
                      dry_run=dry_run), store

  def test_lists_sum_to_input(self):
    rows = [_row(job_id='a', workdir=self.wd),
            _row(job_id='b', state='FAILED'),
            _row(job_id='c', launch_kwargs={'resume_xid': '1'}),
            _row(job_id='d', workdir='/google/src/cloud/u/ws/google3')]
    r, _ = self._run(rows)
    self.assertEqual(r['total'], 4)
    self.assertEqual(len(r['migrated']) + len(r['archived']) + len(r['rejected']), 4)

  def test_dry_run_writes_nothing(self):
    r, store = self._run([_row(job_id='a', workdir=self.wd)], dry_run=True)
    self.assertEqual(len(r['migrated']), 1)
    self.assertEqual(store.load(), {})           # nothing landed

  def test_real_run_populates_the_store(self):
    r, store = self._run([_row(job_id='a', workdir=self.wd)], dry_run=False)
    got = store.load()
    self.assertIn('a', got)
    self.assertEqual(got['a'].project_name, 'p')
    self.assertEqual(got['a'].version, 0)

  def test_forbidden_arch_is_rejected_not_migrated(self):
    r, _ = self._run([_row(job_id='a', workdir=self.wd, power='gb200-8')])
    self.assertEqual(len(r['rejected']), 1)
    self.assertIn('gb200', r['rejected'][0][1])


class BucketPrefixRecovery(unittest.TestCase):
  """★These exist because a leading empty segment made `[:5]` drop the project directory, and
  the obvious assertions did NOT catch it:

    - `ckpt.startswith(bucket + '/')` passes for ANY shorter prefix.
    - self-reflexivity, `rewrite_prefix(ckpt, b, b) == ckpt`, holds for BOTH the right and the
      wrong prefix -- swapping a prefix for itself is a no-op at any depth.

  What does catch it is asking whether the swap preserved the SHAPE: a cross-metro rewrite
  must change the cell and nothing else, so the segment count is invariant and the tail is
  identical. A test has to be able to fail on the actual defect, not merely be present.
  """

  import jobplace as _jp

  def test_prefix_keeps_the_project_segment(self):
    self.assertEqual(
        jm._bucket_prefix_of('/cns/oi-d/home/qiaos/eqr_data/logs/x/step_9'),
        '/cns/oi-d/home/qiaos/eqr_data')

  def test_cross_metro_swap_preserves_segment_count_and_tail(self):
    """The assertion that actually fails on the bug."""
    for ckpt, dst in (
        ('/cns/oi-d/home/qiaos/eqr_data/logs/E/xid_1_a/step_20000',
         '/cns/is-d/home/qiaos/eqr_data'),
        ('/cns/is-d/home/qiaos/lyy_parcae_runs/logs/P/xid_2_b/step_500.pt',
         '/cns/oi-d/home/qiaos/lyy_parcae_runs'),
        ('/cns/oi-d/home/qiaos/eqr_data/r/step_9/state',
         '/cns/li-d/home/qiaos/eqr_data'),
    ):
      with self.subTest(ckpt=ckpt):
        src = jm._bucket_prefix_of(ckpt)
        out = self._jp.rewrite_prefix(ckpt, src, dst)
        self.assertEqual(len(out.split('/')), len(ckpt.split('/')),
                         f'segment count changed: {out}')
        self.assertEqual(out.split('/')[5:], ckpt.split('/')[5:],
                         f'tail was altered: {out}')
        self.assertTrue(out.startswith(dst + '/'))

  def test_no_segment_is_duplicated_by_the_swap(self):
    ckpt = '/cns/oi-d/home/qiaos/eqr_data/logs/E/step_9'
    out = self._jp.rewrite_prefix(ckpt, jm._bucket_prefix_of(ckpt),
                                  '/cns/is-d/home/qiaos/eqr_data')
    self.assertNotIn('/eqr_data/eqr_data/', out)

  def test_unrecoverable_bucket_is_UNKNOWN_not_a_short_guess(self):
    """UNKNOWN makes a cross-metro resume refuse loudly; a short guess corrupts it silently."""
    for bad in ('/cns/oi-d/home/qiaos', 'gs://bucket/logs/x', '', 'relative/path'):
      with self.subTest(bad=bad):
        self.assertEqual(jm._bucket_prefix_of(bad), jc.UNKNOWN)

  def test_every_live_checkpoint_round_trips(self):
    """Against the REAL queue: every declared checkpoint must survive a swap unchanged except
    for its cell."""
    import json
    rows = json.load(open('/usr/local/google/home/qiaos/.tpu_local_queue.json'))['entries']
    n = 0
    for row in rows:
      ckpt = jm.declared_checkpoint(row.get('launch_kwargs') or {})
      if not ckpt or not ckpt.startswith('/cns/'):
        continue
      src = jm._bucket_prefix_of(ckpt)
      self.assertNotEqual(src, jc.UNKNOWN, f'{row.get("job_id")}: {ckpt}')
      dst = src.replace('/oi-d/', '/is-d/').replace('/is-d/', '/li-d/', 1) \
          if '/oi-d/' in src else src.replace('/is-d/', '/oi-d/')
      out = self._jp.rewrite_prefix(ckpt, src, dst)
      self.assertEqual(len(out.split('/')), len(ckpt.split('/')), ckpt)
      n += 1
    self.assertGreater(n, 20, 'expected the live queue to exercise this')
