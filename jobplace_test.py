"""Tests for placement + checkpoint co-location."""

import sys
import unittest

try:
  from google3.experimental.users.qiaos.tpu_utils import jobchain as jc
  from google3.experimental.users.qiaos.tpu_utils import jobplace as jp
except ImportError:  # pragma: no cover - flat-layout fallback
  import jobchain as jc  # type: ignore
  import jobplace as jp  # type: ignore

sys.path.insert(0, '/google/src/cloud/qiaos/run_amply_workspace')


def _job(**kw):
  d = dict(job_id='j1', workdir='/w', target_label='//p:main', project_name='p',
           allowed_metros=['tul'], power='v6p-32')
  d.update(kw)
  return jc.Job(**d)


def _with_ckpt(ckpt, metro, bucket, **kw):
  j = _job(**kw)
  j = jc.open_attempt(j, 'x1', 'somecell', metro, 'na', bucket)
  return jc.close_attempt(j, jc.Outcome.PREEMPTED, ckpt_path=ckpt)


class Choose(unittest.TestCase):
  """RESELECT means 'anywhere I declared', never 'anywhere'."""

  def test_reselect_honours_allowed_metros(self):
    p = jp.choose(_job(allowed_metros=['tul']), ['yucbfiv', 'yutulpz', 'yulpptr'])
    self.assertEqual(p.metro, 'tul')
    self.assertEqual(p.bucket, '/cns/oi-d/home/qiaos/eqr_data')

  def test_reselect_refuses_rather_than_widening(self):
    """A job outside its declared metros is deleted mid-run by the pruner, which is far more
    expensive than not starting."""
    with self.assertRaises(jp.PlacementError) as e:
      jp.choose(_job(allowed_metros=['tul']), ['yucbfiv', 'yulpptr'])
    self.assertIn('allowed_metros', str(e.exception))

  def test_unknown_cell_is_never_a_candidate(self):
    with self.assertRaises(jp.PlacementError):
      jp.choose(_job(allowed_metros=None), ['zzz_not_a_real_cell'])

  def test_no_metros_declared_means_any_KNOWN_cell(self):
    p = jp.choose(_job(allowed_metros=None), ['yucbfiv'])
    self.assertEqual(p.metro, 'cbf')

  def test_pin_reuses_the_previous_cell(self):
    j = _with_ckpt('/cns/oi-d/r/step_9', 'tul', '/cns/oi-d', placement_policy='PIN')
    j.nodes[-1].cell = 'yutulpz'
    p = jp.choose(j, ['yucbfiv', 'yutulpz'])
    self.assertEqual(p.cell, 'yutulpz')

  def test_pin_waits_rather_than_going_elsewhere(self):
    j = _with_ckpt('/cns/oi-d/r/step_9', 'tul', '/cns/oi-d', placement_policy='PIN')
    j.nodes[-1].cell = 'yutulpz'
    with self.assertRaises(jp.PlacementError) as e:
      jp.choose(j, ['nl', 'nk'])
    self.assertIn('not currently available', str(e.exception))

  def test_pin_without_history_is_refused(self):
    """A migrated job has no recorded cell -- the old registry had no such field."""
    with self.assertRaises(jp.PlacementError) as e:
      jp.choose(_job(placement_policy='PIN'), ['yutulpz'])
    self.assertIn('no previous cell', str(e.exception))


class PrefixRewrite(unittest.TestCase):
  """The tail is never parsed: four incompatible shapes exist."""

  def test_all_four_shapes_survive_verbatim(self):
    for tail in ('r/step_9/state', 'r/step_9', 'r/checkpoint_9', 'r/step_9.pt'):
      with self.subTest(tail=tail):
        got = jp.rewrite_prefix(f'/cns/oi-d/{tail}', '/cns/oi-d', '/cns/is-d')
        self.assertEqual(got, f'/cns/is-d/{tail}')

  def test_unrelated_prefix_is_refused_not_guessed(self):
    with self.assertRaises(jp.CheckpointCopyError):
      jp.rewrite_prefix('/cns/xx-d/r/step_9', '/cns/oi-d', '/cns/is-d')


class Colocate(unittest.TestCase):
  def setUp(self):
    self.copied = []
    self.exists = set()

  def _copy(self, src, dst):
    self.copied.append((src, dst))
    self.exists.add(dst)

  def test_same_metro_does_not_copy(self):
    j = _with_ckpt('/cns/oi-d/r/step_9', 'tul', '/cns/oi-d')
    p = jp.choose(j, ['yutulpz'])
    out = jp.colocate_checkpoint(j, p, copy_fn=self._copy,
                                 exists_fn=lambda x: x in self.exists)
    self.assertEqual(out, '/cns/oi-d/r/step_9')
    self.assertEqual(self.copied, [])            # no copy, no risk

  def test_cross_metro_copies_and_points_local(self):
    j = _with_ckpt('/cns/oi-d/r/step_9', 'tul', '/cns/oi-d', allowed_metros=['cbf'])
    p = jp.choose(j, ['yucbfiv'])
    out = jp.colocate_checkpoint(j, p, copy_fn=self._copy,
                                 exists_fn=lambda x: x in self.exists)
    self.assertEqual(out, '/cns/is-d/home/qiaos/eqr_data/r/step_9')
    self.assertEqual(len(self.copied), 1)

  def test_copy_is_idempotent(self):
    j = _with_ckpt('/cns/oi-d/r/step_9', 'tul', '/cns/oi-d', allowed_metros=['cbf'])
    p = jp.choose(j, ['yucbfiv'])
    self.exists.add('/cns/is-d/home/qiaos/eqr_data/r/step_9')
    jp.colocate_checkpoint(j, p, copy_fn=self._copy, exists_fn=lambda x: x in self.exists)
    self.assertEqual(self.copied, [])            # already there

  def test_cold_start_returns_none(self):
    p = jp.choose(_job(), ['yutulpz'])
    self.assertIsNone(jp.colocate_checkpoint(_job(), p, copy_fn=self._copy,
                                             exists_fn=lambda x: False))

  def test_silent_copy_failure_is_fail_closed(self):
    """A copy that reports success but wrote nothing must stop the launch, not fall back."""
    j = _with_ckpt('/cns/oi-d/r/step_9', 'tul', '/cns/oi-d', allowed_metros=['cbf'])
    p = jp.choose(j, ['yucbfiv'])
    with self.assertRaises(jp.CheckpointCopyError):
      jp.colocate_checkpoint(j, p, copy_fn=lambda s, d: None, exists_fn=lambda x: False)

  def test_ghost_write_caught_by_the_DELAYED_reread(self):
    """★rc=0, immediate read-back correct, file gone seconds later. Only the delayed re-read
    tells a real write from a ghost one."""
    j = _with_ckpt('/cns/oi-d/r/step_9', 'tul', '/cns/oi-d', allowed_metros=['cbf'])
    p = jp.choose(j, ['yucbfiv'])
    with self.assertRaises(jp.CheckpointCopyError) as e:
      jp.colocate_checkpoint(j, p, copy_fn=self._copy,
                             exists_fn=lambda x: x in self.exists,
                             verify_fn=lambda x: False)
    self.assertIn('DELAYED', str(e.exception))


class LaunchEnv(unittest.TestCase):
  def test_resume_sets_load_from(self):
    p = jp.choose(_job(), ['yutulpz'])
    env = jp.launch_env(_job(), p, '/cns/oi-d/r/step_9')
    self.assertEqual(env['LOAD_FROM'], '/cns/oi-d/r/step_9')
    self.assertEqual(env['CHECKPOINT_BUCKET'], '/cns/oi-d/home/qiaos/eqr_data')

  def test_cold_start_omits_load_from(self):
    """★Not empty-string: ABSENT. A pinned LOAD_FROM overrides the job's own auto-resume
    forever (measured: step 380k falling back to 298k every preemption)."""
    p = jp.choose(_job(), ['yutulpz'])
    env = jp.launch_env(_job(), p, None)
    self.assertNotIn('LOAD_FROM', env)

  def test_checkpoint_bucket_follows_the_compute_cell(self):
    for cell, want in (('yutulpz', '/cns/oi-d/home/qiaos/eqr_data'),
                       ('yucbfiv', '/cns/is-d/home/qiaos/eqr_data')):
      with self.subTest(cell=cell):
        p = jp.choose(_job(allowed_metros=None), [cell])
        self.assertEqual(jp.launch_env(_job(), p, None)['CHECKPOINT_BUCKET'], want)


class DeclaredBucketWins(unittest.TestCase):
  """★A job that says where it writes must be written there.

  `placement.bucket` is derived from a project-agnostic suffix, which is the right answer
  only for a job that never declared one. Found while verifying the canary: parcae declares
  `/cns/is-d/home/qiaos/lyy_parcae_runs` and the dispatcher was about to hand it
  `/cns/is-d/home/qiaos/eqr_data` -- correct metro, ANOTHER LINE'S DIRECTORY. The job would
  have run fine and its owner would have found an empty output dir.
  """

  def test_declared_bucket_is_used(self):
    j = _job(launch_kwargs={'bucket': '/cns/is-d/home/qiaos/lyy_parcae_runs', 'config': 'c'},
             allowed_metros=['cbf'])
    p = jp.choose(j, ['yucbfiv'])
    self.assertEqual(jp.launch_env(j, p, None)['CHECKPOINT_BUCKET'],
                     '/cns/is-d/home/qiaos/lyy_parcae_runs')

  def test_undeclared_falls_back_to_the_placement_default(self):
    j = _job(launch_kwargs={'config': 'c'}, allowed_metros=['cbf'])
    p = jp.choose(j, ['yucbfiv'])
    self.assertEqual(jp.launch_env(j, p, None)['CHECKPOINT_BUCKET'],
                     '/cns/is-d/home/qiaos/eqr_data')

  def test_all_three_spellings_are_honoured(self):
    for key in ('bucket', 'CHECKPOINT_BUCKET', '--bucket'):
      with self.subTest(key=key):
        j = _job(launch_kwargs={key: '/cns/is-d/mine', 'config': 'c'},
                 allowed_metros=['cbf'])
        p = jp.choose(j, ['yucbfiv'])
        self.assertEqual(jp.launch_env(j, p, None)['CHECKPOINT_BUCKET'], '/cns/is-d/mine')
