"""Tests for jobstore. The centrepiece is StaleSnapshotScenario, which replays the exact
sequence that corrupted the queue all night and asserts the new store refuses it.
"""

import os
import tempfile
import unittest

try:
  from google3.experimental.users.qiaos.tpu_utils import jobchain as jc
  from google3.experimental.users.qiaos.tpu_utils import jobstore as js
except ImportError:  # pragma: no cover - flat-layout fallback
  import jobchain as jc  # type: ignore
  import jobstore as js  # type: ignore



def _job(job_id='j1', **kw):
  d = dict(job_id=job_id, workdir='/home/u/proj', target_label='//p:main',
           project_name='proj', allowed_metros=['tul'], power='v6p-32',
           # required by the poisoned-personal-bucket gate; unrelated to what these
           # cases assert, so give them a valid group-billed destination.
           launch_kwargs={'bucket': '/cns/oi-d/home/qiaos/eqr_data', 'config': 'c'})
  d.update(kw)
  return jc.Job(**d)


class StoreBase(unittest.TestCase):
  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.store = js.JobStore(os.path.join(self.tmp.name, 'q.json'))
    self.workdir = os.path.join(self.tmp.name, 'proj')
    os.makedirs(self.workdir)

  def tearDown(self):
    self.tmp.cleanup()

  def _create(self, job_id='j1', **kw):
    kw.setdefault('workdir', self.workdir)
    return self.store.create(_job(job_id, **kw))


class StaleSnapshotScenario(StoreBase):
  """★THE test. Replays the incident, step for step.

  Old behaviour: a pass read the queue, worked for 20 minutes, wrote back what it had read,
  and silently undid everything that happened in between. Every step held the lock correctly.
  New behaviour: the write is REFUSED, loudly, and the stale writer must re-read.
  """

  def test_twenty_minute_old_write_is_refused(self):
    self._create()
    stale = self.store.get('j1')                      # t0: a pass reads the world
    self.store.patch('j1', stale.version, {'state': jc.JobState.HELD,
                                           'last_reason': 'human parked it'})
    with self.assertRaises(js.StaleWriteError) as e:  # t20: it writes back what it read
      self.store.patch('j1', stale.version, {'state': jc.JobState.QUEUED})
    self.assertEqual(e.exception.expected, stale.version)
    self.assertIs(self.store.get('j1').state, jc.JobState.HELD)   # the human's edit SURVIVES

  def test_untouched_fields_are_never_written(self):
    """110 of 125 rows were overwritten per pass because the writer returned everything it
    read. Here a writer literally cannot express an opinion about a field it did not name."""
    self._create(priority=7)
    j = self.store.get('j1')
    self.store.patch('j1', j.version, {'last_reason': 'only this'})
    self.assertEqual(self.store.get('j1').priority, 7)

  def test_other_jobs_are_never_touched(self):
    self._create('j1'); self._create('j2', priority=3)
    j1 = self.store.get('j1')
    self.store.patch('j1', j1.version, {'last_reason': 'x'})
    j2 = self.store.get('j2')
    self.assertEqual((j2.priority, j2.version), (3, 0))   # untouched, version not bumped

  def test_deletion_does_not_come_back(self):
    """codi-v6 deleted two rows under the lock with triple assertions; nine minutes later a
    pass holding an older snapshot resurrected them. Deletion cannot self-heal the way an
    edit can -- it removes the channel that would announce it."""
    self._create()
    stale = self.store.get('j1')
    self.store.remove('j1', stale.version)
    self.assertIsNone(self.store.get('j1'))
    with self.assertRaises(KeyError):                  # the stale writer cannot re-add it
      self.store.patch('j1', stale.version, {'state': jc.JobState.QUEUED})
    self.assertIsNone(self.store.get('j1'))

  def test_stale_delete_is_refused(self):
    self._create()
    stale = self.store.get('j1')
    self.store.patch('j1', stale.version, {'last_reason': 'moved on'})
    with self.assertRaises(js.StaleWriteError):
      self.store.remove('j1', stale.version)
    self.assertIsNotNone(self.store.get('j1'))


class DoubleSpendScenario(StoreBase):
  """gpu-survey-v3: one row produced five 8-card B200 jobs, ~20 min apart, because a stale
  writer kept resetting it to a pre-launch snapshot while earlier jobs still ran."""

  def test_rollback_to_prelaunch_is_refused(self):
    self._create()
    j = self.store.get('j1')
    pre = j.version
    launched = jc.open_attempt(j, '284380582', 'yutulpz', 'tul', 'na', '/cns/oi-d')
    self.store.commit(launched)
    with self.assertRaises(js.StaleWriteError):
      self.store.patch('j1', pre, {'state': jc.JobState.BUILD_REQUESTED, 'cur_xid': None})
    self.assertEqual(self.store.get('j1').cur_xid, '284380582')

  def test_even_at_the_right_version_the_invariant_holds(self):
    """Defence in depth: if a caller does re-read, I5 still refuses to un-launch a job."""
    self._create()
    launched = jc.open_attempt(self.store.get('j1'), 'x1', 'c', 'tul', 'na', '/cns/oi-d')
    self.store.commit(launched)
    with self.assertRaises(jc.InvariantError):
      self.store.patch('j1', launched.version,
                       {'state': jc.JobState.BUILD_REQUESTED, 'cur_xid': None})


class ResumeCarriesEverything(StoreBase):
  """The 17/17 bug: a requeue rebuilt launch_kwargs from scratch with one key, so jobs
  cold-started AND were priced at a tenth of reality, holding fleet budget at a fake price."""

  def test_full_chain_preserves_context(self):
    kwargs = {'config': 'prod_g3p5', 'bucket': '/cns/oi-d/x', 'exp_name': 'e1'}
    self._create(launch_kwargs=kwargs, allowed_metros=['tul'])
    j = self.store.get('j1')
    j = self.store.commit(jc.open_attempt(j, 'x1', 'yutulpz', 'tul', 'na', '/cns/oi-d'))
    j = self.store.commit(jc.close_attempt(j, jc.Outcome.PREEMPTED,
                                           ckpt_path='/cns/oi-d/r/step_60000', step=60000))
    j = self.store.commit(jc.open_attempt(j, 'x2', 'nl', 'tul', 'na', '/cns/oi-d'))
    got = self.store.get('j1')
    self.assertEqual(got.launch_kwargs, kwargs)           # verbatim
    self.assertEqual(got.allowed_metros, ['tul'])         # I7
    self.assertEqual(got.resume_source.ckpt_path, '/cns/oi-d/r/step_60000')
    self.assertEqual(len(got.nodes), 2)
    self.assertEqual(got.build_attempts, 0)               # a preemption is not a build defect

  def test_ckpt_path_is_stored_verbatim(self):
    """Four incompatible shapes exist; torch ports store a FILE (step_N.pt), not a directory.
    Any normalisation breaks at least one family."""
    for p in ('/cns/is-d/r/step_9/state', '/cns/is-d/r/step_9',
              '/cns/is-d/r/checkpoint_9', '/cns/is-d/r/step_9.pt'):
      with self.subTest(p=p):
        sid = 'j_' + p.replace('/', '_').replace('.', '_')
        self._create(sid)
        j = self.store.commit(jc.open_attempt(self.store.get(sid), 'x', 'c', 'tul', 'na', '/b'))
        j = self.store.commit(jc.close_attempt(j, jc.Outcome.PREEMPTED, ckpt_path=p))
        self.assertEqual(self.store.get(sid).resume_source.ckpt_path, p)


class EnqueueGate(StoreBase):
  def test_forbidden_arch_never_enters_the_store(self):
    with self.assertRaises(jc.RejectedAtEnqueue):
      self.store.create(_job(workdir=self.workdir, power='gb200-8'))
    self.assertEqual(self.store.load(), {})

  def test_depot_root_never_enters_the_store(self):
    with self.assertRaises(jc.RejectedAtEnqueue):
      self.store.create(_job(workdir='/google/src/cloud/qiaos/elt_jax/google3'))
    self.assertEqual(self.store.load(), {})

  def test_valid_job_does_enter(self):
    """Negative control: the gate must not refuse everything."""
    self._create()
    self.assertIn('j1', self.store.load())


class Durability(StoreBase):
  def test_no_tmp_files_left_behind(self):
    self._create()
    j = self.store.get('j1')
    self.store.patch('j1', j.version, {'last_reason': 'x'})
    leftovers = [f for f in os.listdir(self.tmp.name) if '.tmp.' in f]
    self.assertEqual(leftovers, [])

  def test_survives_reload(self):
    self._create(launch_kwargs={'a': 1, 'bucket': '/cns/oi-d/home/qiaos/eqr_data',
                                'config': 'c'})
    reopened = js.JobStore(self.store.path)
    self.assertEqual(reopened.get('j1').launch_kwargs,
                     {'a': 1, 'bucket': '/cns/oi-d/home/qiaos/eqr_data', 'config': 'c'})
