"""Tests for the dispatch loop.

The gates are tested by their CONSEQUENCE, not their return value: what matters is whether a
refusal charged the job a strike, and whether a launched job can ever be un-launched.
"""

import os
import sys
import tempfile
import unittest

try:
  from google3.experimental.users.qiaos.tpu_utils import jobchain as jc
  from google3.experimental.users.qiaos.tpu_utils import jobdispatch as jd
  from google3.experimental.users.qiaos.tpu_utils import jobplace as jp
  from google3.experimental.users.qiaos.tpu_utils import jobstore as js
except ImportError:  # pragma: no cover - flat-layout fallback
  import jobchain as jc  # type: ignore
  import jobdispatch as jd  # type: ignore
  import jobplace as jp  # type: ignore
  import jobstore as js  # type: ignore

sys.path.insert(0, '/google/src/cloud/qiaos/run_amply_workspace')


class Base(unittest.TestCase):
  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.wd = os.path.join(self.tmp.name, 'proj')
    os.makedirs(self.wd)
    self.store = js.JobStore(os.path.join(self.tmp.name, 'q.json'))
    self.logs: list[str] = []
    self.built: list[tuple[str, dict]] = []
    self.copied: list[tuple[str, str]] = []
    self.exists: set[str] = set()
    self.cells = ['yutulpz']
    self.headroom = 1e9
    self.xid_to_return = 'x-new'
    self.identity = None                    # None => identity gate disabled

  def tearDown(self):
    self.tmp.cleanup()

  def _job(self, job_id='j1', **kw):
    d = dict(job_id=job_id, workdir=self.wd, target_label='//p:main', project_name='p',
             allowed_metros=['tul'], power='v6p-32', tier='PROD',
             launch_kwargs={'bucket': '/cns/oi-d/home/qiaos/eqr_data', 'config': 'c'})
    d.update(kw)
    return self.store.create(jc.Job(**d))

  def _builder(self, job, placement, env):
    self.built.append((job.job_id, dict(env)))
    return self.xid_to_return, 'build log tail'

  def _dispatcher(self, **kw):
    d = dict(store=self.store, builder=self._builder,
             available_cells_fn=lambda: self.cells,
             headroom_fn=lambda: self.headroom,
             copy_fn=lambda s, t: (self.copied.append((s, t)), self.exists.add(t))[0],
             exists_fn=lambda p: p in self.exists,
             identity_fn=self.identity,
             log=self.logs.append)
    d.update(kw)
    return jd.Dispatcher(**d)


class HappyPath(Base):
  def test_cold_start_submits_without_load_from(self):
    self._job()
    r = self._dispatcher().round()
    self.assertEqual([x.action for x in r], ['submitted'])
    self.assertNotIn('LOAD_FROM', self.built[0][1])
    self.assertEqual(self.built[0][1]['CHECKPOINT_BUCKET'],
                     '/cns/oi-d/home/qiaos/eqr_data')
    got = self.store.get('j1')
    self.assertEqual(got.cur_xid, 'x-new')
    self.assertTrue(got.ever_had_xid)
    self.assertEqual(got.build_attempts, 0)

  def test_resume_sets_load_from_and_keeps_context(self):
    lk = {'config': 'c', 'bucket': '/cns/oi-d/x'}
    j = self._job(launch_kwargs=lk)
    j = self.store.commit(jc.open_attempt(j, 'x-old', 'yutulpz', 'tul', 'na', '/cns/oi-d'))
    self.store.commit(jc.close_attempt(j, jc.Outcome.PREEMPTED,
                                       ckpt_path='/cns/oi-d/r/step_60000', step=60000))
    self._dispatcher().round()
    self.assertEqual(self.built[0][1]['LOAD_FROM'], '/cns/oi-d/r/step_60000')
    got = self.store.get('j1')
    self.assertEqual(got.launch_kwargs, lk)          # the 17/17 bug
    self.assertEqual(got.allowed_metros, ['tul'])    # I7
    self.assertEqual(len(got.nodes), 2)

  def test_cross_metro_resume_copies_first(self):
    """Operator's rule: the job must never read or write across a metro at runtime."""
    # ★The job runs in cbf, so its own bucket is the cbf one -- a tul bucket here would be
    # refused at the gate, and rightly: that is a job whose every write crosses a metro.
    # The CHECKPOINT it resumes from is the cross-metro part, and that is what gets copied.
    j = self._job(allowed_metros=['cbf'],
                  launch_kwargs={'bucket': '/cns/is-d/home/qiaos/eqr_data', 'config': 'c'})
    j = self.store.commit(jc.open_attempt(j, 'x-old', 'yutulpz', 'tul', 'na', '/cns/oi-d'))
    self.store.commit(jc.close_attempt(j, jc.Outcome.PREEMPTED,
                                       ckpt_path='/cns/oi-d/r/step_9'))
    self.cells = ['yucbfiv']
    self._dispatcher().round()
    self.assertEqual(len(self.copied), 1)
    self.assertTrue(self.built[0][1]['LOAD_FROM'].startswith('/cns/is-d/'))
    self.assertEqual(self.built[0][1]['CHECKPOINT_BUCKET'], '/cns/is-d/home/qiaos/eqr_data')


class AttemptsTaxonomy(Base):
  """★The operator's question, at the loop level: an environment refusal must not spend a
  strike, and every path must converge."""

  def _assert_env_refusal(self, job_id='j1'):
    got = self.store.get(job_id)
    self.assertEqual(got.build_attempts, 0)
    self.assertIs(got.state, jc.JobState.DEFERRED)

  def test_no_cell_defers_without_a_strike(self):
    self._job()
    self.cells = ['yucbfiv']                  # not in allowed_metros
    self.assertEqual(self._dispatcher().round()[0].action, 'deferred')
    self._assert_env_refusal()

  def test_over_budget_defers_without_a_strike(self):
    self._job()
    self.headroom = 0.0001
    self.assertEqual(self._dispatcher().round()[0].action, 'deferred')
    self._assert_env_refusal()

  def test_failed_checkpoint_copy_defers_without_a_strike(self):
    j = self._job(allowed_metros=['cbf'],
                  launch_kwargs={'bucket': '/cns/is-d/home/qiaos/eqr_data', 'config': 'c'})
    j = self.store.commit(jc.open_attempt(j, 'x-old', 'yutulpz', 'tul', 'na', '/cns/oi-d'))
    self.store.commit(jc.close_attempt(j, jc.Outcome.PREEMPTED,
                                       ckpt_path='/cns/oi-d/r/step_9'))
    self.cells = ['yucbfiv']
    d = self._dispatcher(copy_fn=lambda s, t: None)     # silent no-op copy
    self.assertEqual(d.round()[0].action, 'deferred')
    self._assert_env_refusal()
    self.assertEqual(self.built, [])                    # ★never launched against the remote

  def test_no_xid_is_a_defect_and_converges_to_held(self):
    self._job(job_id='j1')
    self.xid_to_return = None
    d = self._dispatcher()
    for _ in range(3):
      d.round()
      for job in self.store.load().values():
        if job.state is jc.JobState.DEFERRED:
          self.store.commit(jc.promote_deferred(job))
    got = self.store.get('j1')
    self.assertEqual(got.build_attempts, 3)
    self.assertIs(got.state, jc.JobState.HELD)          # I10: it CONVERGES
    self.assertIn('build log tail', got.failure_tail or '')

  def test_deferred_jobs_are_promoted_next_round(self):
    self._job()
    self.headroom = 0.0001
    self._dispatcher().round()
    self.assertIs(self.store.get('j1').state, jc.JobState.DEFERRED)
    self.headroom = 1e9
    self.assertEqual(self._dispatcher().round()[0].action, 'submitted')
    self.assertEqual(self.store.get('j1').build_attempts, 0)   # never charged


class IdentityGate(Base):
  """A wrong staged package runs another line's experiment with every check green."""

  def test_mismatch_is_a_defect_not_an_environment_refusal(self):
    self._job()
    self.identity = lambda job: ('//third_party/py/simple_diffusion:main_eqr', 'elt-jax-dit')
    self.assertEqual(self._dispatcher().round()[0].action, 'failed')
    got = self.store.get('j1')
    self.assertEqual(got.build_attempts, 1)             # this one IS the job's fault
    self.assertEqual(self.built, [])

  def test_match_proceeds(self):
    self._job()
    self.identity = lambda job: ('//p:main', 'p')
    self.assertEqual(self._dispatcher().round()[0].action, 'submitted')

  def test_unreadable_identity_defers_rather_than_blaming_the_job(self):
    def boom(job):
      raise OSError('stagedir unreadable')
    self._job()
    self.identity = boom
    self.assertEqual(self._dispatcher().round()[0].action, 'deferred')
    self.assertEqual(self.store.get('j1').build_attempts, 0)


class Serialisation(Base):
  def test_one_build_per_round_by_default(self):
    for i in range(3):
      self._job(job_id=f'j{i}')
    self.assertEqual(len(self._dispatcher().round()), 1)

  def test_priority_order(self):
    self._job(job_id='low', priority=0)
    self._job(job_id='high', priority=9)
    self.assertEqual(self._dispatcher().round()[0].job_id, 'high')

  def test_terminal_and_held_jobs_are_not_dispatched(self):
    j = self._job(job_id='held')
    self.store.commit(jc.record_failure(j, jc.FailureClass.JOB_DEFECT, 'x', max_attempts=1))
    self.assertEqual(self._dispatcher().round(), [])


class BeliefLine(Base):
  """C11: the process must state what it believes, so a stale view is visible from outside."""

  def test_belief_line_reports_store_counts_and_pid(self):
    self._job()
    line = self._dispatcher().belief_line()
    self.assertIn('QUEUED', line)
    self.assertIn('pid=', line)
    self.assertIn(self.store.path, line)


class DryRunLeavesNoTrace(unittest.TestCase):
  """★A dry run must not mutate the store.

  Found by monitor-v47 during the pre-canary observation window: it noticed the v2 store's
  mtime moving while nothing was supposed to be dispatched. The cause was that a dry run
  produced no XID for the same reason a crashed build does, so the dispatcher scored it as a
  job-intrinsic failure -- three observation rounds pushed a job that had never been built to
  HELD with attempts=3, a state only a human can clear.

  ★The damage outlived the observation, and by the second round the dry run was describing
  the marks it had made itself.
  """

  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.wd = os.path.join(self.tmp.name, 'proj')
    os.makedirs(self.wd)
    self.store = js.JobStore(os.path.join(self.tmp.name, 'q.json'))
    self.store.create(jc.Job(
        job_id='j1', workdir=self.wd, target_label='//p:main', project_name='p',
        allowed_metros=['tul'], power='v6p-32',
        launch_kwargs={'bucket': '/cns/oi-d/home/qiaos/eqr_data', 'config': 'c'}))

  def tearDown(self):
    self.tmp.cleanup()

  def _dispatcher(self):
    import jobbuild as _jb
    try:
      from google3.experimental.users.qiaos.tpu_utils import jobbuild as _jb  # noqa: F811
    except ImportError:
      pass
    return jd.Dispatcher(
        store=self.store, builder=_jb.TpuQueueBuilder(dry_run=True, log=lambda s: None),
        available_cells_fn=lambda: ['yutulpz'],
        headroom_fn=lambda: 1e9,
        copy_fn=lambda s, t: None, exists_fn=lambda p: True,
        log=lambda s: None)

  def test_three_rounds_change_nothing(self):
    before = self.store.get('j1')
    d = self._dispatcher()
    for _ in range(3):
      d.round()
    after = self.store.get('j1')
    self.assertEqual(after.version, before.version, 'dry run bumped the version')
    self.assertEqual(after.build_attempts, 0, 'dry run charged a strike')
    self.assertIs(after.state, jc.JobState.QUEUED, 'dry run changed the state')

  def test_the_result_says_what_it_would_have_done(self):
    """Reporting nothing would be safe but useless -- the point of a dry run is the plan."""
    r = self._dispatcher().round()[0]
    self.assertEqual(r.action, 'dry_run')
    self.assertIn('yutulpz', r.detail)


class EnqueueContractIsRecheckedAtDispatch(unittest.TestCase):
  """★A gate that runs only at the door protects the door, not the road.

  Found while choosing a canary: a row already in the store had no bucket at all.
  `validate_enqueue` refuses that shape, but the dispatcher never asked -- it would have
  submitted the job onto the poisoned personal quota, where writes fail while leaving a
  0-byte file behind. Rows outlive the gates that admitted them: a store predates a gate, a
  migration carries an old shape across, or the world moves under a row that was fine.
  """

  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.wd = os.path.join(self.tmp.name, 'proj')
    os.makedirs(self.wd)
    self.store = js.JobStore(os.path.join(self.tmp.name, 'q.json'))
    self.built = []

  def tearDown(self):
    self.tmp.cleanup()

  def _dispatcher(self):
    return jd.Dispatcher(
        store=self.store,
        builder=lambda j, p, e: (self.built.append(j.job_id), ('x1', 'ok'))[1],
        available_cells_fn=lambda: ['yutulpz'], headroom_fn=lambda: 1e9,
        copy_fn=lambda s, t: None, exists_fn=lambda p: True, log=lambda s: None)

  def _smuggle(self, **kw):
    """Put a row into the store bypassing create(), the way a migration or an older
    schema would have."""
    d = dict(job_id='j1', workdir=self.wd, target_label='//p:main', project_name='p',
             allowed_metros=['tul'], power='v6p-32',
             launch_kwargs={'bucket': '/cns/oi-d/home/qiaos/eqr_data', 'config': 'c'})
    d.update(kw)
    jobs = self.store.load()
    jobs['j1'] = jc.Job(**d)
    self.store._write_all(jobs)

  def test_a_row_with_no_bucket_is_held_not_submitted(self):
    self._smuggle(launch_kwargs={'config': 'c', 'exp_name': 'x'})
    r = self._dispatcher().round()[0]
    self.assertEqual(r.action, 'held')
    self.assertEqual(self.built, [])                       # ★never reached the builder
    self.assertIs(self.store.get('j1').state, jc.JobState.HELD)

  def test_a_row_on_the_poisoned_bucket_is_held(self):
    self._smuggle(launch_kwargs={'config': 'c',
                                 'bucket': '/cns/yutulpz-d/home/qiaos/eqr_data'})
    self.assertEqual(self._dispatcher().round()[0].action, 'held')
    self.assertEqual(self.built, [])

  def test_holding_does_not_charge_a_strike(self):
    """A malformed row did not FAIL; a counter is the wrong instrument for "fix me"."""
    self._smuggle(launch_kwargs={'config': 'c', 'exp_name': 'x'})
    self._dispatcher().round()
    self.assertEqual(self.store.get('j1').build_attempts, 0)

  def test_a_valid_row_still_dispatches(self):
    """Negative control: gate 0 must not refuse everything."""
    self._smuggle()
    self.assertEqual(self._dispatcher().round()[0].action, 'submitted')
    self.assertEqual(self.built, ['j1'])
