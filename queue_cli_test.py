"""Unit tests for the local-queue CLI. Drives the subcommands through their flag
objects against a temp queue file; no RPC (status' live probe is not exercised)."""

import os
import tempfile
import time
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
                  _TIER='PROD', _PRIORITY=5, _JOB_NAME='j-a',
                  _LAUNCH=['config=cfg.py', 'force'], _METROS=None,
                  _MAX_PRICE=None, _POWER_TOL=0.5, _TOPOLOGY_LOCKED=False,
                  _WORKDIR='/my/checkout'):
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
    self.assertEqual(e.workdir, '/my/checkout')       # explicit workdir kept
    self.assertEqual(e.state, R.JobState.QUEUED)

  def test_enqueue_defaults_workdir_to_cwd(self):
    # monitor v21 fix: enqueuing from the right checkout must capture it, so the
    # router packages the correct source. Unset --workdir defaults to os.getcwd.
    import os
    with _FlagCtx(_QUEUE_FILE=self.path, _POWER='v7-32', _ARCHS=['v7'],
                  _TIER='PROD', _PRIORITY=0, _JOB_NAME='j-cwd', _LAUNCH=None,
                  _METROS=None, _MAX_PRICE=None, _POWER_TOL=0.5,
                  _TOPOLOGY_LOCKED=False, _WORKDIR=None):
      Q._cmd_enqueue(['queue_cli'])
    self.assertEqual(self._load()[0].workdir, os.getcwd())

  def test_enqueue_empty_workdir_opts_into_router_dir(self):
    with _FlagCtx(_QUEUE_FILE=self.path, _POWER='v7-32', _ARCHS=['v7'],
                  _TIER='PROD', _PRIORITY=0, _JOB_NAME='j-empty', _LAUNCH=None,
                  _METROS=None, _MAX_PRICE=None, _POWER_TOL=0.5,
                  _TOPOLOGY_LOCKED=False, _WORKDIR=''):
      Q._cmd_enqueue(['queue_cli'])
    self.assertEqual(self._load()[0].workdir, '')     # explicit '' preserved

  def test_enqueue_rejects_duplicate_job_id(self):
    common = dict(_QUEUE_FILE=self.path, _POWER='v7-32', _ARCHS=['v7'],
                  _TIER='PROD', _PRIORITY=0, _JOB_NAME='dup', _LAUNCH=None,
                  _METROS=None, _MAX_PRICE=None, _POWER_TOL=0.5,
                  _TOPOLOGY_LOCKED=False)
    with _FlagCtx(**common):
      self.assertEqual(Q._cmd_enqueue(['queue_cli']), 0)
      self.assertEqual(Q._cmd_enqueue(['queue_cli']), 1)   # duplicate
    self.assertEqual(len(self._load()), 1)

  def test_enqueue_requires_job_name(self):
    # ★No auto-mint: an omitted --job_name is a usage error, not a power-hash.
    # This is the inverse of the old test_enqueue_autogenerates_job_id, which
    # asserted the exact behavior we removed -- a job named after nothing.
    with _FlagCtx(_QUEUE_FILE=self.path, _POWER='v6e-32', _ARCHS=['v6e'],
                  _TIER='PROD', _PRIORITY=0, _JOB_NAME=None, _LAUNCH=None,
                  _METROS=None, _MAX_PRICE=None, _POWER_TOL=0.5,
                  _TOPOLOGY_LOCKED=False):
      self.assertEqual(Q._cmd_enqueue(['queue_cli']), 2)   # REFUSED
    self.assertEqual(self._load(), [])                     # nothing enqueued

  def test_dequeue_removes_by_id(self):
    # seed two
    for jid in ('a', 'b'):
      with _FlagCtx(_QUEUE_FILE=self.path, _POWER='v7-32', _ARCHS=['v7'],
                    _TIER='PROD', _PRIORITY=0, _JOB_NAME=jid, _LAUNCH=None,
                    _METROS=None, _MAX_PRICE=None, _POWER_TOL=0.5,
                    _TOPOLOGY_LOCKED=False, _NO_SNAPSHOT=True):
        Q._cmd_enqueue(['queue_cli'])
    with _FlagCtx(_QUEUE_FILE=self.path, _JOB_NAME=None):
      # -f: a QUEUED row is in _DEQUEUE_UNSAFE_STATES, so removal needs --force.
      rc = Q._cmd_dequeue(['queue_cli', 'a', '-f'])   # positional id
    self.assertEqual(rc, 0)
    self.assertEqual([e.job_id for e in self._load()], ['b'])

  def test_dequeue_missing_id_is_error(self):
    with _FlagCtx(_QUEUE_FILE=self.path, _JOB_NAME=None):
      rc = Q._cmd_dequeue(['queue_cli', 'nope'])
    self.assertEqual(rc, 1)

  def test_dequeue_no_id_usage_error(self):
    with _FlagCtx(_QUEUE_FILE=self.path, _JOB_NAME=None):
      rc = Q._cmd_dequeue(['queue_cli'])
    self.assertEqual(rc, 2)


class MainDispatchTest(unittest.TestCase):

  def test_unknown_subcommand(self):
    self.assertEqual(Q.main(['queue_cli', 'bogus']), 2)

  def test_no_subcommand(self):
    self.assertEqual(Q.main(['queue_cli']), 2)

class EnqueueSnapshotTest(unittest.TestCase):
  """The enqueue-time snapshot: freeze a local copy of the checkout so edits
  between enqueue and build cannot change the code a job ships."""

  def setUp(self):
    import shutil
    self.tmp = tempfile.mkdtemp()
    self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))
    # An operator-scoped queue file, like ~/.tpu_local_queue.json.
    self.path = os.path.join(self.tmp, '.tpu_local_queue.json')
    # A realistic code checkout to snapshot.
    self.code = os.path.join(self.tmp, 'checkout', 'torch_impl')
    os.makedirs(self.code)
    with open(os.path.join(self.code, 'main.py'), 'w') as f:
      f.write('print(1)\n')
    with open(os.path.join(self.code, 'BUILD'), 'w') as f:
      f.write('py_binary(name="main")\n')

  def _load(self):
    return RC.load_queue(self.path)

  def _common(self, **kw):
    base = dict(_QUEUE_FILE=self.path, _POWER='v7-32', _ARCHS=['v7'],
                _TIER='PROD', _PRIORITY=0, _LAUNCH=None, _METROS=None,
                _MAX_PRICE=None, _POWER_TOL=0.5, _TOPOLOGY_LOCKED=False,
                _NO_SNAPSHOT=False)
    base.update(kw)
    return base

  def test_snapshot_root_is_operator_scoped(self):
    # sqa's and npu's queue files must yield DISTINCT snapshot roots, or one
    # operator's snapshots land in the other's dir (same bug the queue file and
    # tmux session each shipped before scoping).
    sqa = Q._snapshot_root('/home/u/.tpu_local_queue.json')
    npu = Q._snapshot_root('/home/u/lyy-work/.npu_local_queue.json')
    self.assertNotEqual(sqa, npu)
    self.assertTrue(sqa.endswith('.tpu_enqueue_snapshots'))
    self.assertTrue(npu.endswith('.npu_enqueue_snapshots'))

  def test_enqueue_creates_snapshot_and_records_it(self):
    with _FlagCtx(**self._common(_JOB_NAME='j-snap', _WORKDIR=self.code)):
      rc = Q._cmd_enqueue(['queue_cli'])
    self.assertEqual(rc, 0)
    e = self._load()[0]
    # snapshot_dir is set, distinct from workdir, and on local disk (our tmp).
    self.assertTrue(e.snapshot_dir)
    self.assertNotEqual(e.snapshot_dir, e.workdir)
    self.assertTrue(os.path.isdir(e.snapshot_dir))
    # workdir is still recorded (provenance), and package_dir prefers the snap.
    self.assertEqual(e.workdir, self.code)
    self.assertEqual(R.package_dir(e), e.snapshot_dir)
    # The frozen copy has the real files.
    self.assertTrue(os.path.isfile(os.path.join(e.snapshot_dir, 'main.py')))
    self.assertTrue(os.path.isfile(os.path.join(e.snapshot_dir, 'BUILD')))

  def test_snapshot_is_frozen_against_later_edits(self):
    # THE WHOLE POINT: edit the source AFTER enqueue; the snapshot must not move.
    with _FlagCtx(**self._common(_JOB_NAME='j-frozen', _WORKDIR=self.code)):
      Q._cmd_enqueue(['queue_cli'])
    snap = self._load()[0].snapshot_dir
    with open(os.path.join(self.code, 'main.py'), 'w') as f:
      f.write('print("EDITED AFTER ENQUEUE")\n')
    frozen = open(os.path.join(snap, 'main.py')).read()
    self.assertEqual(frozen, 'print(1)\n')          # unchanged

  def test_snapshot_excludes_junk(self):
    # The excludes mirror the build-time stage excludes; heavy/irrelevant dirs
    # must not be copied.
    os.makedirs(os.path.join(self.code, '.git'))
    with open(os.path.join(self.code, '.git', 'HEAD'), 'w') as f:
      f.write('ref: x\n')
    with open(os.path.join(self.code, 'model.pt'), 'w') as f:
      f.write('weights')
    os.makedirs(os.path.join(self.code, 'data'))
    with open(os.path.join(self.code, 'data', 'big.rec'), 'w') as f:
      f.write('x' * 1000)
    with _FlagCtx(**self._common(_JOB_NAME='j-excl', _WORKDIR=self.code)):
      Q._cmd_enqueue(['queue_cli'])
    snap = self._load()[0].snapshot_dir
    self.assertFalse(os.path.exists(os.path.join(snap, '.git')))
    self.assertFalse(os.path.exists(os.path.join(snap, 'model.pt')))
    self.assertFalse(os.path.exists(os.path.join(snap, 'data')))
    self.assertTrue(os.path.isfile(os.path.join(snap, 'main.py')))

  def test_no_snapshot_flag_opts_out(self):
    with _FlagCtx(**self._common(_JOB_NAME='j-live', _WORKDIR=self.code,
                                 _NO_SNAPSHOT=True)):
      rc = Q._cmd_enqueue(['queue_cli'])
    self.assertEqual(rc, 0)
    e = self._load()[0]
    self.assertEqual(e.snapshot_dir, '')             # no snapshot
    self.assertEqual(R.package_dir(e), self.code)    # builds the live workdir

  def test_enqueue_refuses_workspace_root(self):
    # A checkout ROOT (has WORKSPACE) is the `tpu enqueue` from ~/work trap:
    # refuse it fail-closed rather than snapshot 400+ top-level dirs.
    root = os.path.join(self.tmp, 'ws')
    os.makedirs(root)
    open(os.path.join(root, 'WORKSPACE'), 'w').close()
    with _FlagCtx(**self._common(_JOB_NAME='j-ws', _WORKDIR=root)):
      rc = Q._cmd_enqueue(['queue_cli'])
    self.assertEqual(rc, 2)                           # REFUSED
    self.assertEqual(self._load(), [])               # nothing enqueued

  def test_refused_when_over_size_cap(self):
    # A tree bigger than the cap is refused (fail-closed), naming --no_snapshot.
    orig = Q._SNAPSHOT_MAX_BYTES
    Q._SNAPSHOT_MAX_BYTES = 10        # 10 bytes: our checkout is bigger
    self.addCleanup(lambda: setattr(Q, '_SNAPSHOT_MAX_BYTES', orig))
    with _FlagCtx(**self._common(_JOB_NAME='j-big', _WORKDIR=self.code)):
      rc = Q._cmd_enqueue(['queue_cli'])
    self.assertEqual(rc, 2)
    self.assertEqual(self._load(), [])
    # And --no_snapshot lets the same oversized dir through (opt-out honored).
    with _FlagCtx(**self._common(_JOB_NAME='j-big2', _WORKDIR=self.code,
                                 _NO_SNAPSHOT=True)):
      self.assertEqual(Q._cmd_enqueue(['queue_cli']), 0)

  def test_dequeue_removes_snapshot(self):
    with _FlagCtx(**self._common(_JOB_NAME='j-del', _WORKDIR=self.code)):
      Q._cmd_enqueue(['queue_cli'])
    snap = self._load()[0].snapshot_dir
    self.assertTrue(os.path.isdir(snap))
    with _FlagCtx(_QUEUE_FILE=self.path, _JOB_NAME=None):
      # -f: a QUEUED row needs --force to dequeue (it is in the unsafe set); the
      # snapshot cleanup runs on the actual removal, which force permits.
      Q._cmd_dequeue(['queue_cli', 'j-del', '-f'])
    self.assertFalse(os.path.exists(snap))           # reclaimed on dequeue

  def test_duplicate_enqueue_does_not_leak_or_clobber_snapshot(self):
    with _FlagCtx(**self._common(_JOB_NAME='dup', _WORKDIR=self.code)):
      self.assertEqual(Q._cmd_enqueue(['queue_cli']), 0)
      snap = self._load()[0].snapshot_dir
      self.assertEqual(Q._cmd_enqueue(['queue_cli']), 1)   # duplicate id
    # the original snapshot survives; the rejected dup left no staging leftover
    self.assertTrue(os.path.isdir(snap))
    root = Q._snapshot_root(self.path)
    leftovers = [n for n in os.listdir(root) if '.staging.' in n]
    self.assertEqual(leftovers, [])

  def test_gc_reclaims_orphan_snapshot(self):
    # A snapshot dir whose job_id is no longer in the queue, older than the
    # grace window, is reclaimed by the next enqueue's GC.
    with _FlagCtx(**self._common(_JOB_NAME='j-keep', _WORKDIR=self.code)):
      Q._cmd_enqueue(['queue_cli'])
    root = Q._snapshot_root(self.path)
    orphan = os.path.join(root, 'ghost-job')
    os.makedirs(orphan)
    old = time.time() - Q._SNAPSHOT_GC_GRACE_S - 60
    os.utime(orphan, (old, old))
    # A second enqueue triggers the opportunistic GC.
    with _FlagCtx(**self._common(_JOB_NAME='j-keep2', _WORKDIR=self.code)):
      Q._cmd_enqueue(['queue_cli'])
    self.assertFalse(os.path.exists(orphan))         # orphan reclaimed
    # the two live jobs' snapshots are kept
    live = {e.snapshot_dir for e in self._load()}
    for d in live:
      self.assertTrue(os.path.isdir(d))

  def test_gc_spares_fresh_orphan_within_grace(self):
    # A brand-new dir not yet in the queue may be a concurrent enqueue mid-copy;
    # the grace window must protect it.
    with _FlagCtx(**self._common(_JOB_NAME='j1', _WORKDIR=self.code)):
      Q._cmd_enqueue(['queue_cli'])
    root = Q._snapshot_root(self.path)
    fresh = os.path.join(root, 'inflight-job')
    os.makedirs(fresh)                                # mtime = now
    with _FlagCtx(**self._common(_JOB_NAME='j2', _WORKDIR=self.code)):
      Q._cmd_enqueue(['queue_cli'])
    self.assertTrue(os.path.isdir(fresh))            # spared (within grace)



if __name__ == '__main__':
  unittest.main()
