"""Tests for jobchain. Each case reproduces a MEASURED incident from 2026-08-28.

A test here is not "does the code do what it says"; it is "would this have stopped the thing
that actually happened". The incident is named in each docstring so that a future reader can
tell whether a change is a fix or a regression.
"""

import unittest

try:
  from google3.experimental.users.qiaos.tpu_utils import jobchain as jc
except ImportError:  # pragma: no cover - flat-layout fallback
  import jobchain as jc  # type: ignore



def _job(**kw):
  d = dict(job_id='j1', workdir='/home/u/proj', target_label='//p:main',
           project_name='proj', allowed_metros=['tul'])
  d.update(kw)
  return jc.Job(**d)


class TerminalIsOneWay(unittest.TestCase):
  """gpu-survey-v3: a FAILED row was flipped back to BUILD_REQUESTED within 30s of its guard
  stopping, and the dispatcher rebuilt it. 'Terminal' meant nothing while a stale writer
  could replay an older snapshot."""

  def test_failed_cannot_become_claimable(self):
    old = _job(state=jc.JobState.FAILED, version=5)
    new = jc.Job.from_dict(old.to_dict())
    new.version, new.state = 6, jc.JobState.BUILD_REQUESTED
    with self.assertRaises(jc.InvariantError) as e:
      jc.check_transition(old, new)
    self.assertIn('I4', str(e.exception))

  def test_completed_cannot_be_reopened(self):
    old = _job(state=jc.JobState.COMPLETED, version=5)
    new = jc.Job.from_dict(old.to_dict()); new.version, new.state = 6, jc.JobState.QUEUED
    with self.assertRaises(jc.InvariantError):
      jc.check_transition(old, new)


class AttemptsAreMonotonic(unittest.TestCase):
  """codi-torch measured att 1 -> 0 at 12:49Z, with last_reason reverting verbatim to an
  older snapshot. A counter that only ever increments cannot decrease for any legitimate
  reason, which makes it the hardest available evidence of an overwrite."""

  def test_decrease_is_refused(self):
    old = _job(build_attempts=3, version=1)
    new = jc.Job.from_dict(old.to_dict()); new.version, new.build_attempts = 2, 0
    with self.assertRaises(jc.InvariantError) as e:
      jc.check_transition(old, new)
    self.assertIn('I3', str(e.exception))


class XidIsNeverForgotten(unittest.TestCase):
  """trm-torch-port-v2: a row holding RUNNING xid=284366800 was replaced wholesale with
  BUILD_REQUESTED/xid=None while the real job kept running -- so the row looked never-launched
  and a second dispatch would have doubled the spend."""

  def test_launched_job_cannot_become_claimable_while_attempt_is_open(self):
    """The row is reset to a pre-launch snapshot while the attempt is STILL OPEN -- the real
    job keeps running, the row looks never-launched, and the next dispatch doubles the spend."""
    old = jc.open_attempt(_job(), '284366800', 'yucbfad', 'cbf', 'na', '/cns/is-d')
    old.state = jc.JobState.RUNNING
    new = jc.Job.from_dict(old.to_dict())
    new.version, new.state, new.cur_xid = old.version + 1, jc.JobState.BUILD_REQUESTED, None
    with self.assertRaises(jc.InvariantError) as e:
      jc.check_transition(old, new)
    self.assertIn('I5', str(e.exception))

  def test_becoming_claimable_after_closing_the_attempt_is_ALLOWED(self):
    """★The negative control for I5, and the one that matters: a resume IS a launched job
    becoming claimable again. An invariant that blocked this too would look like extra safety
    while making resume impossible -- which is exactly what my first version did, until this
    test caught it."""
    j = jc.open_attempt(_job(), 'x1', 'yucbfad', 'cbf', 'na', '/cns/is-d')
    closed = jc.close_attempt(j, jc.Outcome.PREEMPTED, ckpt_path='/cns/is-d/r/step_9', step=9)
    jc.check_transition(j, closed)                 # must NOT raise
    self.assertIs(closed.state, jc.JobState.QUEUED)
    self.assertTrue(closed.ever_had_xid)
    self.assertIsNone(closed.cur_xid)

  def test_ever_had_xid_cannot_be_unset(self):
    old = _job(ever_had_xid=True, version=1)
    new = jc.Job.from_dict(old.to_dict()); new.version, new.ever_had_xid = 2, False
    with self.assertRaises(jc.InvariantError):
      jc.check_transition(old, new)


class OneLiveXidPerJob(unittest.TestCase):
  """gpu-survey-v3 built five 8-card B200 jobs from a single row, ~20 min apart, because the
  row kept being reset to a pre-launch snapshot while earlier jobs were still running."""

  def test_second_open_attempt_is_refused(self):
    j = jc.open_attempt(_job(), 'x1', 'yutulpz', 'tul', 'na', '/cns/oi-d')
    with self.assertRaises(jc.InvariantError) as e:
      jc.open_attempt(j, 'x2', 'yutulpz', 'tul', 'na', '/cns/oi-d')
    self.assertIn('I6', str(e.exception))

  def test_close_then_reopen_is_fine(self):
    j = jc.open_attempt(_job(), 'x1', 'yutulpz', 'tul', 'na', '/cns/oi-d')
    j = jc.close_attempt(j, jc.Outcome.PREEMPTED, ckpt_path='/cns/oi-d/r/step_100', step=100)
    j2 = jc.open_attempt(j, 'x2', 'yutulpz', 'tul', 'na', '/cns/oi-d')
    self.assertEqual(len(j2.nodes), 2)
    self.assertEqual(j2.cur_xid, 'x2')


class MetrosSurvive(unittest.TestCase):
  """Seven lines independently demanded this. Losing allowed_metros does not fail loudly:
  the job builds, runs, and is then silently deleted by the WIM pruner (XID 284145906)."""

  def test_metros_cannot_change(self):
    old = _job(allowed_metros=['tul'], version=1)
    new = jc.Job.from_dict(old.to_dict()); new.version, new.allowed_metros = 2, None
    with self.assertRaises(jc.InvariantError) as e:
      jc.check_transition(old, new)
    self.assertIn('I7', str(e.exception))

  def test_metros_survive_a_full_resume_cycle(self):
    j = _job(allowed_metros=['cbf'], launch_kwargs={'config': 'c', 'bucket': '/cns/is-d'})
    j = jc.open_attempt(j, 'x1', 'yucbfiv', 'cbf', 'na', '/cns/is-d')
    j = jc.close_attempt(j, jc.Outcome.PREEMPTED, ckpt_path='/cns/is-d/r/step_9', step=9)
    j = jc.open_attempt(j, 'x2', 'yucbful', 'cbf', 'na', '/cns/is-d')
    self.assertEqual(j.allowed_metros, ['cbf'])
    self.assertEqual(j.launch_kwargs, {'config': 'c', 'bucket': '/cns/is-d'})


class ChainIsAppendOnly(unittest.TestCase):
  """codi-reproduction-v6 deleted two rows under the sidecar lock with triple assertions;
  nine minutes later they were fully resurrected from an older snapshot."""

  def test_nodes_cannot_shrink(self):
    """Truncating the chain is refused. Note the job is left in a NON-claimable state here so
    that I8 is what fires: with a claimable state, I5 catches it first (also correct, but a
    test that cannot tell which guard fired cannot detect one of them silently breaking)."""
    j = jc.open_attempt(_job(), 'x1', 'c', 'tul', 'na', '/cns/oi-d')
    j = jc.close_attempt(j, jc.Outcome.FAILED)
    new = jc.Job.from_dict(j.to_dict())
    new.version, new.nodes = j.version + 1, []
    new.state = jc.JobState.HELD          # keep out of CLAIMABLE so I5 does not pre-empt I8
    with self.assertRaises(jc.InvariantError) as e:
      jc.check_transition(j, new)
    self.assertIn('I8', str(e.exception))

  def test_truncation_while_claimable_is_still_caught(self):
    """Truncating the chain is refused regardless of the resulting state. I8 is the guard that
    fires here; I5 deliberately does NOT, because a closed attempt plus a claimable state is
    a legitimate resume (see test_becoming_claimable_after_closing_the_attempt_is_ALLOWED)."""
    j = jc.open_attempt(_job(), 'x1', 'c', 'tul', 'na', '/cns/oi-d')
    j = jc.close_attempt(j, jc.Outcome.FAILED)
    new = jc.Job.from_dict(j.to_dict())
    new.version, new.nodes = j.version + 1, []
    new.state = jc.JobState.QUEUED
    with self.assertRaises(jc.InvariantError) as e:
      jc.check_transition(j, new)
    self.assertIn('I8', str(e.exception))

  def test_closed_node_cannot_reopen(self):
    j = jc.open_attempt(_job(), 'x1', 'c', 'tul', 'na', '/cns/oi-d')
    j = jc.close_attempt(j, jc.Outcome.FAILED)
    new = jc.Job.from_dict(j.to_dict()); new.version += 1; new.nodes[0].ended_at = None
    with self.assertRaises(jc.InvariantError):
      jc.check_transition(j, new)


class AttemptsTaxonomy(unittest.TestCase):
  """The operator's question: why would a build attempt care about credit headroom?

  It should not. Budget/quota/capacity refusals drove entries to att=65, and the real root
  cause was worse than the mislabelling: route_check.py:804 incremented attempts with NO
  convergence path at all, so a job could sit in HELD and keep counting (measured 4 -> 8)."""

  def test_environment_refusal_does_not_count(self):
    j = _job(build_attempts=1)
    out = jc.record_failure(j, jc.FailureClass.ENVIRONMENT, 'over the credit bar')
    self.assertEqual(out.build_attempts, 1)          # unchanged
    self.assertIs(out.state, jc.JobState.DEFERRED)   # and it CONVERGES

  def test_job_defect_counts_and_converges_to_held(self):
    j = _job(build_attempts=2)
    out = jc.record_failure(j, jc.FailureClass.JOB_DEFECT, 'BUILD failed', max_attempts=3)
    self.assertEqual(out.build_attempts, 3)
    self.assertIs(out.state, jc.JobState.HELD)       # I10: there IS an exit

  def test_every_failure_path_converges(self):
    """I10 as a property: no class/count combination leaves a job in a counting-forever state."""
    for cls in jc.FailureClass:
      for att in range(0, 12):
        out = jc.record_failure(_job(build_attempts=att), cls, 'r', max_attempts=3)
        self.assertIn(out.state, (jc.JobState.DEFERRED, jc.JobState.QUEUED, jc.JobState.HELD))
        if cls is jc.FailureClass.ENVIRONMENT:
          self.assertEqual(out.build_attempts, att)

  def test_failure_tail_is_persisted(self):
    """One line spent 12h and 9 builds unable to get a single line of error text."""
    out = jc.record_failure(_job(), jc.FailureClass.JOB_DEFECT, 'no XID', tail='ERROR: boom')
    self.assertEqual(out.failure_tail, 'ERROR: boom')


class ForbiddenArchs(unittest.TestCase):
  """Operator directive: GB200/GB300 must not be used.

  ★Refused at enqueue rather than deleted from the price table: `gb200) echo "20"` is a
  limit-price CAP, and the caller reads an empty cap as 'no policy: leave the job uncapped',
  so deleting those rows would RELAX the constraint while looking like a removal."""

  def test_explicit_request_is_refused(self):
    with self.assertRaises(jc.RejectedAtEnqueue) as e:
      jc.validate_enqueue(_job(power='gb200-8', workdir='/tmp/x'))
    self.assertIn('gb200', str(e.exception))

  def test_hiding_in_allowed_archs_is_refused(self):
    with self.assertRaises(jc.RejectedAtEnqueue):
      jc.validate_enqueue(_job(power='h100-8', allowed_archs=['h100', 'gb300'],
                               workdir='/tmp/x'))

  def test_permitted_arch_is_not_blocked_by_this_check(self):
    """Negative control: the gate must not refuse everything."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
      jc.validate_enqueue(_job(power='h100-8', allowed_archs=['h100'], workdir=d,
                               allowed_metros=['cbf'],
                               launch_kwargs={'bucket': '/cns/is-d/home/qiaos/eqr_data',
                                              'config': 'c'}))


class WorkdirValidation(unittest.TestCase):
  """The depot-root workdir produced 76.1% of one day's 91,437 CreateSnapshot failures."""

  def test_depot_root_by_shape(self):
    with self.assertRaises(jc.RejectedAtEnqueue) as e:
      jc.validate_enqueue(_job(workdir='/google/src/cloud/qiaos/elt_jax/google3'))
    self.assertIn('depot root', str(e.exception))

  def test_depot_root_by_count_predicate(self):
    """Scanning by PREDICATE, not by matching the one bad string we already knew:
    that difference turned 4 known rows into 20 actual ones."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
      with self.assertRaises(jc.RejectedAtEnqueue):
        jc.validate_enqueue(_job(workdir=d), top_level_count=417)

  def test_tmp_is_refused(self):
    with self.assertRaises(jc.RejectedAtEnqueue) as e:
      jc.validate_enqueue(_job(workdir='/tmp'))
    self.assertIn('/tmp', str(e.exception))

  def test_stagedir_inside_workdir_is_refused(self):
    import os, tempfile
    with tempfile.TemporaryDirectory() as d:
      sub = os.path.join(d, 'stages'); os.makedirs(sub)
      with self.assertRaises(jc.RejectedAtEnqueue) as e:
        jc.validate_enqueue(_job(workdir=d), stagedir_root=sub)
      self.assertIn('INSIDE', str(e.exception))


class IdentityVerification(unittest.TestCase):
  """parcae-torch XID 284387576 and codi-torch XID 284359695 both silently ran elt's target.
  Every structural check was green: config.sh present, BUILD present, 415 files."""

  def test_mismatch_is_refused(self):
    j = _job(target_label='//parcae:main', project_name='parcae-torch')
    with self.assertRaises(jc.RejectedAtEnqueue) as e:
      jc.verify_identity(j, '//third_party/py/simple_diffusion:main_eqr', 'elt-jax-dit')
    self.assertIn('IDENTITY MISMATCH', str(e.exception))

  def test_match_passes(self):
    j = _job(target_label='//parcae:main', project_name='parcae-torch')
    jc.verify_identity(j, '//parcae:main', 'parcae-torch')   # must not raise

  def test_missing_fingerprint_refused_at_enqueue(self):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
      with self.assertRaises(jc.RejectedAtEnqueue):
        jc.validate_enqueue(_job(workdir=d, target_label=jc.UNKNOWN))


class BatchIsEvalOnly(unittest.TestCase):
  """★These four cases did not exist when the gate was written, and that is why the gate was
  wrong in BOTH directions for 19 of 19 live BATCH rows.

  The original check read `'eval' in job_id`. job_id is machine-generated as
  `<power>-<6 hex>` (queue_cli._new_job_id), so the word can never appear: every BATCH job was
  refused, real evals included. And a hand-written id containing "eval" admitted a TRAINING
  job -- the single thing the rule exists to prevent. Semantics belong in a field.
  """

  def setUp(self):
    self.tmp = __import__('tempfile').TemporaryDirectory()
    self.wd = self.tmp.name

  def tearDown(self):
    self.tmp.cleanup()

  def _j(self, **kw):
    d = dict(job_id='v6p-16-a04a6b', workdir=self.wd, target_label='//p:main',
             project_name='p', power='v6p-16', allowed_metros=['tul'],
             # A bucket is required by an unrelated gate; these cases are about BATCH, so
             # give them a valid one rather than letting a second refusal mask the first.
             launch_kwargs={'bucket': '/cns/oi-d/home/qiaos/eqr_data', 'config': 'c'})
    d.update(kw)
    if 'launch_kwargs' in kw:
      d['launch_kwargs'] = {'bucket': '/cns/oi-d/home/qiaos/eqr_data', **kw['launch_kwargs']}
    return jc.Job(**d)

  def test_real_eval_on_batch_is_ACCEPTED(self):
    """The false-positive half: a genuine eval, with a real generated id, must pass."""
    jc.validate_enqueue(self._j(tier='BATCH', is_eval=True,
                                launch_kwargs={'config': 'eval_arc1_pass2_3ema'}))

  def test_training_on_batch_is_REFUSED(self):
    with self.assertRaises(jc.RejectedAtEnqueue) as e:
      jc.validate_enqueue(self._j(tier='BATCH', is_eval=False,
                                  launch_kwargs={'config': 'maze64_train_lr8e-4'}))
    self.assertIn('eval-only', str(e.exception))

  def test_a_job_id_spelling_eval_does_NOT_grant_batch(self):
    """The false-negative half: renaming must not be a way past the gate."""
    with self.assertRaises(jc.RejectedAtEnqueue):
      jc.validate_enqueue(self._j(job_id='eval-v6p-16-deadbe', tier='BATCH', is_eval=False,
                                  launch_kwargs={'config': 'maze64_train_lr8e-4'}))

  def test_prod_is_unaffected_by_the_eval_flag(self):
    jc.validate_enqueue(self._j(tier='PROD', is_eval=False))
    jc.validate_enqueue(self._j(tier='PROD', is_eval=True))


class BucketGuard(unittest.TestCase):
  """★The personal Colossus quota is full and its handle is poisoned: a write fails AND leaves
  a 0-byte file, so the loss presents as "the file is there". One run lost 5000 steps that way.
  The launcher's own default points at that bucket, so declaring nothing is the dangerous case.
  """

  def setUp(self):
    self.tmp = __import__('tempfile').TemporaryDirectory()

  def tearDown(self):
    self.tmp.cleanup()

  def _j(self, **kw):
    d = dict(job_id='v6p-32-abc123', workdir=self.tmp.name, target_label='//p:main',
             project_name='p', power='v6p-32', allowed_metros=['tul', 'cbf'],
             launch_kwargs={'bucket': '/cns/oi-d/home/qiaos/eqr_data', 'config': 'c'})
    d.update(kw)
    return jc.Job(**d)

  def test_silence_is_refused_because_the_default_is_poisoned(self):
    with self.assertRaises(jc.RejectedAtEnqueue) as e:
      jc.validate_enqueue(self._j(launch_kwargs={'config': 'c'}))
    self.assertIn('no bucket declared', str(e.exception))

  def test_explicit_personal_bucket_is_refused(self):
    with self.assertRaises(jc.RejectedAtEnqueue) as e:
      jc.validate_enqueue(self._j(
          launch_kwargs={'bucket': '/cns/yutulpz-d/home/qiaos/eqr_data'}))
    self.assertIn('personal Colossus quota', str(e.exception))

  def test_group_bucket_passes(self):
    """Negative control: the guard must not refuse the correct destination."""
    jc.validate_enqueue(self._j(launch_kwargs={'bucket': '/cns/oi-d/home/qiaos/eqr_data', 'config': 'c'}))

  def test_alternative_spellings_are_all_honoured(self):
    for key in ('bucket', 'CHECKPOINT_BUCKET', '--bucket'):
      with self.subTest(key=key):
        # config too: an unrelated gate requires it, and this case is about the bucket
        # key's spelling, not about which refusal happens to fire first.
        jc.validate_enqueue(self._j(launch_kwargs={key: '/cns/is-d/home/qiaos/x',
                                                   'config': 'c'}))


class LaunchKwargsShape(unittest.TestCase):
  """★Both shapes come from the old requeue path, and both fail SILENTLY today: a dotted
  pseudo-key is passed through as `--config.load_from=` which no launcher accepts (the job
  cold-starts from step 0), and an exp_name-only row lost its config to repeated `--launch=`
  arguments overwriting each other. Reported by maze128 against its own four rows."""

  def setUp(self):
    self.tmp = __import__('tempfile').TemporaryDirectory()

  def tearDown(self):
    self.tmp.cleanup()

  def _j(self, lk):
    return jc.Job(job_id='v7-32-abc123', workdir=self.tmp.name, target_label='//p:main',
                  project_name='p', power='v7-32', allowed_metros=['tul', 'cbf'],
                  launch_kwargs=lk)

  def test_a_launcher_swallowed_key_is_refused(self):
    with self.assertRaises(jc.RejectedAtEnqueue) as e:
      jc.validate_enqueue(self._j({'bucket': '/cns/oi-d/x', 'config': 'c',
                                   'config.load_from': '/cns/oi-d/r/step_9'}))
    self.assertIn('config.load_from', str(e.exception))

  def test_a_legitimate_dotted_override_is_ALLOWED(self):
    """★The negative control that my first version of this gate failed.

    `--config.*` is a normal ml_collections override; the launcher forwards it and the binary
    parses it, and some entry points REQUIRE the dotted form (a bare `--eval_ckpt=` is
    rejected upstream). A blanket dotted-key refusal made parcae's official eval recipe
    unsubmittable with no way around it -- caught by parcae-v6 before canary, from its real
    config rather than from a constructed example.
    """
    jc.validate_enqueue(self._j({
        'bucket': '/cns/is-d/home/qiaos/lyy_parcae_runs',
        'config': 'parcae_140m_strict_nsgram',
        'config.eval_ckpt': '/cns/is-d/official/bare_ckpt',
        'config.val_data.data_dir': '/cns/is-d/official/val',
        'exp_name': 'parcae-140m-OFFICIAL-eval',
    }))

  def test_bare_load_from_is_also_refused(self):
    """The undotted spelling is swallowed too -- the discriminator is the NAME, not the dot."""
    with self.assertRaises(jc.RejectedAtEnqueue):
      jc.validate_enqueue(self._j({'bucket': '/cns/oi-d/x', 'config': 'c',
                                   'load_from': '/cns/oi-d/r/step_9'}))

  def test_a_sanity_job_with_no_config_file_is_ALLOWED(self):
    """★A torch sanity job carries an entry-point flag and no config file, and is perfectly
    well-formed. My first version demanded a `config` key and refused it -- reported by
    trm-torch-v3 against its only row, before canary."""
    jc.validate_enqueue(self._j({'group': '9',
                                 'bucket': '/cns/is-d/home/qiaos/eqr_data',
                                 'exp_name': 'trm_arc1_torch_h100_sanity',
                                 'app.sanity_only': 'true'}))

  def test_exp_name_only_shell_is_refused(self):
    # A bucket is supplied so that the bucket gate does not fire first and mask which
    # refusal is under test -- a test that cannot tell two guards apart cannot detect one
    # of them silently breaking.
    with self.assertRaises(jc.RejectedAtEnqueue) as e:
      jc.validate_enqueue(self._j({'exp_name': 'maze128-row98',
                                   'bucket': '/cns/oi-d/home/qiaos/eqr_data'}))
    self.assertIn('says nothing about what to RUN', str(e.exception))

  def test_the_real_maze128_shells_are_all_refused(self):
    """Against the live store: the four rows maze128 reported must not be dispatchable."""
    import json
    store = json.load(open('/usr/local/google/home/qiaos/.tpu_jobs_v2.json'))
    hit = 0
    for raw in store['jobs']:
      if raw['job_id'] not in ('v7-32-9e7d95', 'v7-32-6ec303', 'v7-32-817127',
                               'v7-32-b36772'):
        continue
      hit += 1
      with self.assertRaises(jc.RejectedAtEnqueue, msg=raw['job_id']):
        jc.validate_enqueue(jc.Job.from_dict(raw))
    self.assertEqual(hit, 4, 'expected all four rows to be present in the store')

  def test_a_well_formed_row_still_passes(self):
    """Negative control: the gate must not refuse a correct submission."""
    # ★No load_from here: a checkpoint belongs in the CHAIN (nodes[].ckpt_path), from which
    # the dispatcher sets LOAD_FROM at launch. Putting it in launch_kwargs is the shape the
    # launcher swallows, which is why the gate above refuses it.
    jc.validate_enqueue(self._j({'bucket': '/cns/oi-d/home/qiaos/eqr_data', 'config': 'c',
                                 'exp_name': 'x'}))


class BucketMustBeReachable(unittest.TestCase):
  """★A bucket alone is not enough -- it must be in a metro the job can be placed in.

  Caught by monitor-v47: its scan of the store checked bucket/metro co-location while mine
  checked only that a bucket existed. A tul bucket with no metro constraint can land in cbf,
  and then every checkpoint write crosses a metro: ~94x slower, duty cycle under the 0.20
  floor, and the pruner deletes the job mid-run -- no preemption notice, no crash.
  """

  def setUp(self):
    self.tmp = __import__('tempfile').TemporaryDirectory()

  def tearDown(self):
    self.tmp.cleanup()

  def _j(self, bucket, metros):
    return jc.Job(job_id='v6p-32-abc123', workdir=self.tmp.name, target_label='//p:main',
                  project_name='p', power='v6p-32', allowed_metros=metros,
                  launch_kwargs={'bucket': bucket, 'config': 'c'})

  def test_co_located_passes(self):
    jc.validate_enqueue(self._j('/cns/oi-d/home/qiaos/eqr_data', ['tul']))
    jc.validate_enqueue(self._j('/cns/is-d/home/qiaos/eqr_data', ['cbf']))

  def test_cross_metro_bucket_is_refused(self):
    with self.assertRaises(jc.RejectedAtEnqueue) as e:
      jc.validate_enqueue(self._j('/cns/oi-d/home/qiaos/eqr_data', ['cbf']))
    self.assertIn('not in allowed_metros', str(e.exception))

  def test_a_cns_bucket_with_no_metros_is_refused(self):
    """Nothing keeps the job in the bucket's metro, so 'it has a bucket' proves nothing."""
    with self.assertRaises(jc.RejectedAtEnqueue) as e:
      jc.validate_enqueue(self._j('/cns/oi-d/home/qiaos/eqr_data', None))
    self.assertIn('allowed_metros is empty', str(e.exception))

  def test_multi_metro_passes_when_the_bucket_is_in_one_of_them(self):
    jc.validate_enqueue(self._j('/cns/oi-d/home/qiaos/eqr_data', ['tul', 'cbf']))


class EveryTransitionStampsTheTime(unittest.TestCase):
  """★A record that changes without moving `updated_at` says "nothing happened here" while
  something did.

  The dispatcher's hold path built a record by hand and omitted the timestamp; the store then
  had a row at version N+1 in a new state, whose updated_at still pointed at an earlier
  event. monitor-v48's sentinel correctly saw a file rewritten with no entry newer than the
  baseline, and had to ask whether a third-party writer existed. The fix is not to remember
  the field -- it is to stop hand-building records.
  """

  def test_all_transitions_advance_updated_at(self):
    import time as _t
    base = _job()
    base.updated_at = 1000.0
    now = 2000.0
    cases = {
        'hold': lambda j: jc.hold(j, 'r', now=now),
        'cancel': lambda j: jc.cancel(j, 'r', now=now),
        'record_failure': lambda j: jc.record_failure(
            j, jc.FailureClass.JOB_DEFECT, 'r', now=now),
        'open_attempt': lambda j: jc.open_attempt(j, 'x', 'c', 'tul', 'na', '/b', now=now),
    }
    for name, fn in cases.items():
      with self.subTest(transition=name):
        out = fn(jc.Job.from_dict(base.to_dict()))
        self.assertEqual(out.updated_at, now, f'{name} left updated_at at {out.updated_at}')
        self.assertEqual(out.version, base.version + 1, f'{name} did not bump version')

  def test_hold_does_not_charge_a_strike(self):
    out = jc.hold(_job(build_attempts=1), 'malformed')
    self.assertEqual(out.build_attempts, 1)
    self.assertIs(out.state, jc.JobState.HELD)


class AnOpenAttemptIsNotAbandoned(unittest.TestCase):
  """★A node's outcome defaulted to ABANDONED, so a job that was RUNNING right now -- verified
  in XM, writing to its bucket -- carried a node saying ABANDONED. Caught by monitor-v48 on
  the canary's own record: two fields on one record saying opposite things.

  ABANDONED means "we stopped tracking this and cannot know". That is a real finding, and it
  must not be produced by merely not having reached the end yet."""

  def test_a_fresh_attempt_is_in_flight(self):
    j = jc.open_attempt(_job(), 'x1', 'if', 'cbf', 'na', '/cns/is-d/x')
    self.assertIs(j.nodes[-1].outcome, jc.Outcome.IN_FLIGHT)
    self.assertIsNone(j.nodes[-1].ended_at)

  def test_closing_sets_the_real_outcome(self):
    j = jc.open_attempt(_job(), 'x1', 'if', 'cbf', 'na', '/cns/is-d/x')
    for oc in (jc.Outcome.COMPLETED, jc.Outcome.FAILED, jc.Outcome.PREEMPTED):
      with self.subTest(outcome=oc):
        closed = jc.close_attempt(jc.Job.from_dict(j.to_dict()), oc)
        self.assertIs(closed.nodes[-1].outcome, oc)
        self.assertIsNotNone(closed.nodes[-1].ended_at)

  def test_open_and_ended_agree(self):
    """The two ways of asking "is this attempt over" must never disagree.

    ★The predicate is "is there a CREDIBLE end time", not "is ended_at None". A 0.0 satisfies
    `is not None` and fails truthiness, so it slips through the gap between those two
    questions -- which is exactly how one slipped past this test and into the live store.
    """
    _ended = jc.attempt_has_ended   # ★the one definition, not a re-spelling of it

    j = jc.open_attempt(_job(), 'x1', 'if', 'cbf', 'na', '/cns/is-d/x')
    for n in j.nodes:
      self.assertEqual(_ended(n), n.outcome is not jc.Outcome.IN_FLIGHT)
    closed = jc.close_attempt(j, jc.Outcome.COMPLETED)
    for n in closed.nodes:
      self.assertEqual(_ended(n), n.outcome is not jc.Outcome.IN_FLIGHT)

  def test_a_zero_timestamp_never_reads_as_a_real_time(self):
    """★0.0 formats as 1970-01-01, an ordinary-looking date nobody re-reads. The sentinel
    must be a value no formatter can turn into a plausible answer."""
    self.assertLess(jc.UNKNOWN_TIME, 0)
    for f in ('started_at',):
      self.assertEqual(getattr(jc.Node(xid='x'), f), jc.UNKNOWN_TIME)
    for f in ('enqueued_at', 'updated_at'):
      self.assertEqual(getattr(jc.Job(job_id='j'), f), jc.UNKNOWN_TIME)


class TheWholeStoreMustBeConsistent(unittest.TestCase):
  """★Transition tests only see records this process just built. The store is full of records
  written by earlier code, and that is where the disagreements live: monitor-v48 and I scanned
  the same file for the same inconsistency and got 1 versus 0, because a node carried
  `ended_at = 0.0` -- which passes `is None` and fails a truthiness test."""

  def test_ended_at_an_unknown_time_is_LEGITIMATE(self):
    """★A seed node knows THAT an attempt ended and not WHEN. Demanding a timestamp would
    force the migrator to invent one -- which is precisely how the 0.0 arrived."""
    j = _job()
    j.nodes = [jc.Node(xid='x1', outcome=jc.Outcome.COMPLETED,
                       started_at=jc.UNKNOWN_TIME, ended_at=jc.UNKNOWN_TIME)]
    self.assertEqual(jc.check_node_consistency(j), [])

  def test_a_zero_timestamp_is_reported(self):
    j = _job()
    j.nodes = [jc.Node(xid='x1', outcome=jc.Outcome.COMPLETED, started_at=0.0, ended_at=0.0)]
    problems = jc.check_node_consistency(j)
    self.assertTrue(any('reads as both' in p for p in problems), problems)

  def test_completed_without_an_end_time_is_reported(self):
    j = _job()
    j.nodes = [jc.Node(xid='x1', outcome=jc.Outcome.COMPLETED, ended_at=None)]
    self.assertTrue(any('answers differently' in p for p in jc.check_node_consistency(j)))

  def test_in_flight_with_an_end_time_is_reported(self):
    j = _job()
    j.nodes = [jc.Node(xid='x1', outcome=jc.Outcome.IN_FLIGHT, ended_at=123.0)]
    self.assertTrue(any('answers differently' in p for p in jc.check_node_consistency(j)))

  def test_cur_xid_with_no_matching_open_node_is_reported(self):
    """A row claiming a live job its own chain does not record -- the shape that let a real
    running job look never-launched."""
    j = _job(cur_xid='284525515')
    j.nodes = [jc.Node(xid='other', outcome=jc.Outcome.FAILED, ended_at=1.0)]
    self.assertTrue(any('no IN_FLIGHT node carries it' in p
                        for p in jc.check_node_consistency(j)))

  def test_a_healthy_chain_reports_nothing(self):
    """Negative control: a checker that flags everything is as useless as one that flags
    nothing."""
    j = jc.open_attempt(_job(), 'x1', 'if', 'cbf', 'na', '/cns/is-d/x')
    self.assertEqual(jc.check_node_consistency(j), [])
    closed = jc.close_attempt(j, jc.Outcome.PREEMPTED)
    self.assertEqual(jc.check_node_consistency(closed), [])

  def test_the_LIVE_store_is_clean(self):
    """★Runs against the real file, so an inconsistency written by earlier code cannot hide
    behind tests that only exercise fresh records."""
    import json
    import os
    path = os.path.expanduser('~/.tpu_jobs_v2.json')
    if not os.path.exists(path):
      self.skipTest('no live store')
    raw = json.load(open(path))
    all_problems = []
    for r in raw['jobs']:
      all_problems += [f"{r['job_id']}: {p}"
                       for p in jc.check_node_consistency(jc.Job.from_dict(r))]
    self.assertEqual(all_problems, [], '\n'.join(all_problems))


class SentinelsNeverRenderAsDates(unittest.TestCase):
  """★A sentinel is only safe while every reader knows it. -1 strftimes to 1969-12-31 and 0.0
  to 1970-01-01 -- ordinary-looking dates that land in a report and are never re-read.
  monitor-v48 caught exactly this: my checker recognised the sentinel, but nothing stopped a
  report from printing 1969."""

  def test_sentinels_render_as_words_not_dates(self):
    self.assertEqual(jc.fmt_time(None), '<not set>')
    self.assertEqual(jc.fmt_time(jc.UNKNOWN_TIME), '<unknown>')
    self.assertEqual(jc.fmt_time(0.0), '<unknown>')
    for bad in ('1969', '1970'):
      self.assertNotIn(bad, jc.fmt_time(jc.UNKNOWN_TIME) + jc.fmt_time(0.0))

  def test_a_real_time_still_renders(self):
    """Negative control: a formatter that hides everything is useless."""
    self.assertIn('2026', jc.fmt_time(1787966305.0))

  def test_end_time_distinguishes_the_three_cases(self):
    running = jc.Node(xid='x', outcome=jc.Outcome.IN_FLIGHT)
    seeded = jc.Node(xid='x', outcome=jc.Outcome.COMPLETED, ended_at=jc.UNKNOWN_TIME)
    done = jc.Node(xid='x', outcome=jc.Outcome.COMPLETED, ended_at=1787966305.0)
    self.assertEqual(jc.fmt_end_time(running), '<still running>')
    self.assertEqual(jc.fmt_end_time(seeded), '<ended, time unknown>')
    self.assertIn('2026', jc.fmt_end_time(done))
