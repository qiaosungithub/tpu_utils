"""Tests for the real builder. No `tpu queue` is ever executed here: the runner is injected,
so every outcome (success, budget refusal, timeout, crash) is reproducible offline."""

import os
import sys
import tempfile
import unittest

try:
  from google3.experimental.users.qiaos.tpu_utils import jobbuild as jb
  from google3.experimental.users.qiaos.tpu_utils import jobchain as jc
  from google3.experimental.users.qiaos.tpu_utils import jobplace as jp
except ImportError:  # pragma: no cover - flat-layout fallback
  import jobbuild as jb  # type: ignore
  import jobchain as jc  # type: ignore
  import jobplace as jp  # type: ignore

sys.path.insert(0, '/google/src/cloud/qiaos/run_amply_workspace')


def _job(**kw):
  d = dict(job_id='j1', workdir='/w', target_label='//p:main', project_name='p',
           power='v6p-32', tier='PROD', allowed_metros=['tul'])
  d.update(kw)
  return jc.Job(**d)


_PLACE = jp.Placement(cell='yutulpz', metro='tul', continent='na',
                      bucket='/cns/oi-d/home/qiaos/eqr_data', reason='test')


class Argv(unittest.TestCase):
  def test_launch_kwargs_pass_through_verbatim(self):
    """Rebuilding this dict instead of carrying it cost 17 jobs their checkpoints and let
    them hold budget at a tenth of the real price."""
    argv = jb.build_argv(_job(launch_kwargs={'config': 'prod_g3p5',
                                             'bucket': '/cns/is-d/x',
                                             'skip-preflight': True,
                                             'never': False}), _PLACE)
    self.assertIn('--config=prod_g3p5', argv)
    self.assertIn('--bucket=/cns/is-d/x', argv)
    self.assertIn('--skip-preflight', argv)          # True -> bare flag
    self.assertNotIn('--never', argv)                # False -> dropped

  def test_cell_and_tier_are_pinned(self):
    argv = jb.build_argv(_job(), _PLACE)
    self.assertIn('--cell=yutulpz', argv)
    self.assertIn('--tier=PROD', argv)
    self.assertIn('--tpu_type=v6p-32', argv)


class Outcomes(unittest.TestCase):
  def _build(self, out, rc=0, timed_out=False):
    b = jb.TpuQueueBuilder(runner=lambda *a: (out, rc, timed_out))
    return b(_job(), _PLACE, {'CHECKPOINT_BUCKET': '/cns/oi-d/x'})

  def test_xid_is_extracted(self):
    xid, _ = self._build('... Created experiment 284380582 ...')
    self.assertEqual(xid, '284380582')

  def test_resume_line_also_yields_an_xid(self):
    xid, _ = self._build('Created 1 work unit(s) in experiment 284364771')
    self.assertEqual(xid, '284364771')

  def test_budget_marker_is_recognised(self):
    xid, tail = self._build('blah\n[[BUDGET_DEFERRED]]\nERROR: Budget exceeded ...', rc=1)
    self.assertIsNone(xid)
    self.assertIn('[[BUDGET_DEFERRED]]', tail)

  def test_budget_SENTENCE_alone_is_also_recognised(self):
    """★Older records carry only the English sentence, and any future refusal path that
    forgets the marker would otherwise be misclassified as a job defect -- the exact mistake
    that drove counters to 65."""
    self.assertTrue(jb.is_budget_refusal(
        '\x1b[31m[budget check] ERROR: Budget exceeded for tpu check!\x1b[0m'))

  def test_a_real_failure_is_not_mistaken_for_a_refusal(self):
    """Negative control: the classifier must not call everything a budget refusal."""
    self.assertFalse(jb.is_budget_refusal('FATAL: config.sh not found'))
    xid, tail = self._build('FATAL: config.sh not found', rc=1)
    self.assertIsNone(xid)
    self.assertNotIn('[[BUDGET_DEFERRED]]', tail)
    self.assertIn('rc=1', tail)

  def test_timeout_is_reported_with_its_bound(self):
    """The rule was in the daemon's own comments and applied to one lane but not the two
    that mattered; an unbounded one-shot then lived 2.45 hours holding a stale snapshot."""
    xid, tail = self._build('partial output', timed_out=True)
    self.assertIsNone(xid)
    self.assertIn('TIMED OUT', tail)

  def test_the_tail_keeps_the_END_of_the_output(self):
    """One line spent 12h and 9 builds unable to obtain a single sentence of error text.
    The error is at the bottom; a head-truncated log preserves only the banner."""
    out = 'BANNER\n' + ('x' * 5000) + '\nTHE ACTUAL ERROR'
    _, tail = self._build(out, rc=1)
    self.assertIn('THE ACTUAL ERROR', tail)

  def test_dry_run_executes_nothing(self):
    calls = []
    b = jb.TpuQueueBuilder(dry_run=True, runner=lambda *a: calls.append(a) or ('', 0, False))
    xid, tail = b(_job(), _PLACE, {})
    self.assertIsNone(xid)
    self.assertEqual(calls, [])
    self.assertIn(jb.DRY_RUN_MARKER, tail)   # a marker the dispatcher can match on,
                                             # not prose it would have to guess at


class StagedIdentity(unittest.TestCase):
  """Reads the STAGEDIR copy on purpose -- the one a wrapper backfills from a global default,
  which is how two lines silently ran a third line's target."""

  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()

  def tearDown(self):
    self.tmp.cleanup()

  def _write(self, project, target):
    open(os.path.join(self.tmp.name, 'config.sh'), 'w').write(
        f'export PROJECT_NAME="{project}"\nexport TARGET_LABEL="{target}"\n')

  def test_reads_the_staged_fingerprint(self):
    self._write('parcae-torch', '//x/parcae:main')
    tl, pn = jb.staged_identity(_job(workdir=self.tmp.name))
    self.assertEqual((tl, pn), ('//x/parcae:main', 'parcae-torch'))

  def test_crosswired_package_is_caught_by_the_gate(self):
    self._write('elt-jax-dit', '//third_party/py/simple_diffusion:main_eqr')
    job = _job(workdir=self.tmp.name, target_label='//x/parcae:main',
               project_name='parcae-torch')
    tl, pn = jb.staged_identity(job)
    with self.assertRaises(jc.RejectedAtEnqueue):
      jc.verify_identity(job, tl, pn)

  def test_missing_config_raises_rather_than_returning_a_guess(self):
    with self.assertRaises(OSError):
      jb.staged_identity(_job(workdir=self.tmp.name))


class TheTpuFunctionMustBeSourced(unittest.TestCase):
  """★`tpu` is a shell function, not an executable. Running argv directly fails with
  `[Errno 2] No such file or directory: 'tpu'` -- observed on the first real dispatch.
  Pointing at an absolute path is not a fix either: there is no binary to point at."""

  def test_the_command_sources_the_wrapper(self):
    seen = {}

    def fake_run(argv, cwd=None, env=None, capture_output=None, text=None, timeout=None):
      seen['argv'] = argv
      class R:
        stdout, stderr, returncode = 'Created experiment 123456', '', 0
      return R()

    import subprocess as _sp
    orig, _sp.run = _sp.run, fake_run
    try:
      b = jb.TpuQueueBuilder()
      xid, _ = b(_job(workdir='/'), _PLACE, {})
    finally:
      _sp.run = orig
    self.assertEqual(xid, '123456')
    self.assertEqual(seen['argv'][:2], ['bash', '-c'])
    self.assertIn('source ', seen['argv'][2])
    self.assertIn('tpu queue', seen['argv'][2])

  def test_a_missing_workdir_is_refused_before_running_anything(self):
    """cwd is what the rsync packages; an absent one would ship whatever was inherited."""
    b = jb.TpuQueueBuilder()
    xid, tail = b(_job(workdir='/no/such/dir/anywhere'), _PLACE, {})
    self.assertIsNone(xid)
    self.assertIn('workdir does not exist', tail)


class ALaunchMustNotBeMissed(unittest.TestCase):
  """★The canary launched XID 284525515 and the scheduler recorded "build produced no XID",
  because `tpu queue` says "Launched experiment" while the regex knew only "Created
  experiment". Missing a launch is not merely lost information: the dispatcher retries,
  builds again, launches a SECOND real job, and fails to recognise that one too."""

  def test_all_known_phrasings_are_recognised(self):
    for text, want in (
        ('... Created experiment 284380582 ...', '284380582'),
        ('Launched experiment 284525515 "parcae-140m-OFFICIAL"', '284525515'),
        ('Successfully registered XID 284525515 in ~/.tpu_jobs.json', '284525515'),
        ('Created 1 work unit(s) in experiment 284364771', '284364771'),
    ):
      with self.subTest(text=text[:40]):
        self.assertEqual(jb.extract_xid(text), want)

  def test_a_launch_the_regex_misses_is_still_caught_by_the_registry(self):
    """★The regex will always be one phrasing behind. Reading the artefact is not."""
    import json as _json
    import tempfile as _tf
    import time as _t
    with _tf.NamedTemporaryFile('w', suffix='.json', delete=False) as fh:
      _json.dump({'284525515': {'exp_name': 'my-eval', 'submitted_at': _t.time()}}, fh)
      path = fh.name
    orig, jb._REGISTRY_PATH = jb._REGISTRY_PATH, path
    try:
      self.assertEqual(jb.confirm_xid_from_registry('my-eval', _t.time() - 60), '284525515')
    finally:
      jb._REGISTRY_PATH = orig
      os.unlink(path)

  def test_an_older_run_of_the_same_experiment_is_not_claimed(self):
    """★Five entries share this canary's exp_name; only one is tonight's. Matching on name
    alone would have adopted a 10-hour-old job as the launch that just happened."""
    import json as _json
    import tempfile as _tf
    import time as _t
    now = _t.time()
    with _tf.NamedTemporaryFile('w', suffix='.json', delete=False) as fh:
      _json.dump({'111111': {'exp_name': 'my-eval', 'submitted_at': now - 36000}}, fh)
      path = fh.name
    orig, jb._REGISTRY_PATH = jb._REGISTRY_PATH, path
    try:
      self.assertIsNone(jb.confirm_xid_from_registry('my-eval', now - 60))
    finally:
      jb._REGISTRY_PATH = orig
      os.unlink(path)
