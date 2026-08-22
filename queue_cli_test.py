"""Unit tests for the local-queue CLI. Drives the subcommands through their flag
objects against a temp queue file; no RPC (status' live probe is not exercised)."""

import os
import tempfile
import unittest

from absl import flags

from google3.experimental.users.qiaos.tpu_utils import queue_cli as Q
from google3.experimental.users.qiaos.tpu_utils import route_check as RC
from google3.experimental.users.qiaos.tpu_utils import route_lib as R


def setUpModule():
  # The subcommands read flag .value directly; without app.run the flags are
  # "unparsed" and absl refuses access. Mark them parsed once for the module.
  flags.FLAGS.mark_as_parsed()


class ParseLaunchTest(unittest.TestCase):

  def test_kv_and_bare(self):
    got = Q._parse_launch_kwargs(['config=cfg.py', 'force', 'steps=1000'])
    self.assertEqual(got, {'config': 'cfg.py', 'force': True, 'steps': '1000'})

  def test_empty(self):
    self.assertEqual(Q._parse_launch_kwargs(None), {})
    self.assertEqual(Q._parse_launch_kwargs([]), {})

  def test_value_with_equals(self):
    # only the first '=' splits, so a value may contain '='
    self.assertEqual(Q._parse_launch_kwargs(['x=a=b']), {'x': 'a=b'})


class _FlagCtx:
  """Set absl flags for the duration of a with-block, restore after."""

  def __init__(self, **kw):
    self._kw = kw
    self._saved = {}

  def __enter__(self):
    for name, val in self._kw.items():
      flag_name = getattr(Q, name).name          # FlagHolder -> registered name
      self._saved[flag_name] = getattr(flags.FLAGS, flag_name)
      setattr(flags.FLAGS, flag_name, val)
    return self

  def __exit__(self, *a):
    for flag_name, val in self._saved.items():
      setattr(flags.FLAGS, flag_name, val)


class EnqueueDequeueTest(unittest.TestCase):

  def setUp(self):
    self.path = tempfile.mkstemp(suffix='.json')[1]
    self.addCleanup(lambda: os.path.exists(self.path) and os.remove(self.path))

  def _load(self):
    return RC.load_queue(self.path)

  def test_enqueue_requires_power_and_archs(self):
    with _FlagCtx(_QUEUE_FILE=self.path, _POWER=None, _ARCHS=None):
      rc = Q._cmd_enqueue(['queue_cli'])
    self.assertEqual(rc, 2)
    self.assertEqual(self._load(), [])

  def test_enqueue_writes_entry(self):
    with _FlagCtx(_QUEUE_FILE=self.path, _POWER='v7-32', _ARCHS=['v7', 'v6p'],
                  _TIER='PROD', _PRIORITY=5, _JOB_ID='j-a',
                  _LAUNCH=['config=cfg.py', 'force'], _METROS=None,
                  _MAX_PRICE=None, _POWER_TOL=0.5, _TOPOLOGY_LOCKED=False):
      rc = Q._cmd_enqueue(['queue_cli'])
    self.assertEqual(rc, 0)
    q = self._load()
    self.assertEqual(len(q), 1)
    e = q[0]
    self.assertEqual(e.job_id, 'j-a')
    self.assertEqual(e.power, 'v7-32')
    self.assertEqual(e.allowed_archs, ['v7', 'v6p'])
    self.assertEqual(e.priority, 5)
    self.assertEqual(e.launch_kwargs, {'config': 'cfg.py', 'force': True})
    self.assertEqual(e.state, R.JobState.QUEUED)

  def test_enqueue_rejects_duplicate_job_id(self):
    common = dict(_QUEUE_FILE=self.path, _POWER='v7-32', _ARCHS=['v7'],
                  _TIER='PROD', _PRIORITY=0, _JOB_ID='dup', _LAUNCH=None,
                  _METROS=None, _MAX_PRICE=None, _POWER_TOL=0.5,
                  _TOPOLOGY_LOCKED=False)
    with _FlagCtx(**common):
      self.assertEqual(Q._cmd_enqueue(['queue_cli']), 0)
      self.assertEqual(Q._cmd_enqueue(['queue_cli']), 1)   # duplicate
    self.assertEqual(len(self._load()), 1)

  def test_enqueue_autogenerates_job_id(self):
    with _FlagCtx(_QUEUE_FILE=self.path, _POWER='v6e-32', _ARCHS=['v6e'],
                  _TIER='PROD', _PRIORITY=0, _JOB_ID=None, _LAUNCH=None,
                  _METROS=None, _MAX_PRICE=None, _POWER_TOL=0.5,
                  _TOPOLOGY_LOCKED=False):
      self.assertEqual(Q._cmd_enqueue(['queue_cli']), 0)
    q = self._load()
    self.assertEqual(len(q), 1)
    self.assertTrue(q[0].job_id.startswith('v6e-32-'))

  def test_dequeue_removes_by_id(self):
    # seed two
    for jid in ('a', 'b'):
      with _FlagCtx(_QUEUE_FILE=self.path, _POWER='v7-32', _ARCHS=['v7'],
                    _TIER='PROD', _PRIORITY=0, _JOB_ID=jid, _LAUNCH=None,
                    _METROS=None, _MAX_PRICE=None, _POWER_TOL=0.5,
                    _TOPOLOGY_LOCKED=False):
        Q._cmd_enqueue(['queue_cli'])
    with _FlagCtx(_QUEUE_FILE=self.path, _JOB_ID=None):
      rc = Q._cmd_dequeue(['queue_cli', 'a'])   # positional id
    self.assertEqual(rc, 0)
    self.assertEqual([e.job_id for e in self._load()], ['b'])

  def test_dequeue_missing_id_is_error(self):
    with _FlagCtx(_QUEUE_FILE=self.path, _JOB_ID=None):
      rc = Q._cmd_dequeue(['queue_cli', 'nope'])
    self.assertEqual(rc, 1)

  def test_dequeue_no_id_usage_error(self):
    with _FlagCtx(_QUEUE_FILE=self.path, _JOB_ID=None):
      rc = Q._cmd_dequeue(['queue_cli'])
    self.assertEqual(rc, 2)


class MainDispatchTest(unittest.TestCase):

  def test_unknown_subcommand(self):
    self.assertEqual(Q.main(['queue_cli', 'bogus']), 2)

  def test_no_subcommand(self):
    self.assertEqual(Q.main(['queue_cli']), 2)


if __name__ == '__main__':
  unittest.main()
