r"""`tpu enqueue` / `tpu queue-status` / `tpu dequeue`: the local-queue CLI.

This is the operator's side-by-side path to the SMART queue. It does NOT touch
the existing `tpu queue` (the one-shot immediate submit). Instead it appends a
DESIRED run to a durable local queue (~/.tpu_local_queue.json, unlimited -- a
queued job costs nothing because PENDING does not bill), which the router
(route_check) later drains into the XM queue when a cell can actually place it.

Subcommands:
  enqueue    add a desired run (power + allowed archs + policy + launch kwargs)
  queue-status  show the local queue and WHY each job is or is not placeable now
  dequeue    remove one or more jobs from the local queue by job_id

Everything here is thin argument-marshalling over route_lib (the schema) and
route_check (queue persistence + a dry-run planning tick for the status view).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

from absl import app
from absl import flags

from google3.experimental.users.qiaos.tpu_utils import avail_provider
from google3.experimental.users.qiaos.tpu_utils import route_check
from google3.experimental.users.qiaos.tpu_utils import route_lib


# `queue_file` and `group` are already defined by route_check (imported above);
# reuse those flag objects rather than redefining them, or absl raises
# DuplicateFlagError the moment both modules load in one binary.
_QUEUE_FILE = route_check._QUEUE_FILE  # pylint: disable=protected-access

# ★MUST be declared to absl even though _cmd_dequeue reads raw argv. absl parses
# the command line FIRST and aborts on any flag it does not know, so
# `tpu dequeue <id> --force` died with "Unknown command line flag 'force'" and
# the escape hatch documented in the refusal message could never actually be
# used. Same shape as the `--group=9` flag absl swallowed (v19 §2a): the code
# that consumes a flag and the parser that admits it are two different places,
# and reading argv directly does not exempt you from declaring it.
_FORCE = flags.DEFINE_bool(
    'force', False,
    'Dequeue even rows the safety gate refuses (live states and HELD). You '
    'take responsibility for any XID the row was tracking.')

# ★Register the dry-run SPELLINGS the code already claims to accept. Same trap
# as --force: `_cmd_dequeue` matched '--dry-run' / '--dryrun' / '-n' out of raw
# argv, but absl rejects an undeclared flag before any of that runs, so all
# three died with "Unknown command line flag" -- while --dry_run worked. The
# failure mode is the dangerous direction: someone types the spelling they
# remember, gets an error they read as a typo, and re-runs WITHOUT the flag.
# DEFINE_alias makes the parser accept them and point at the real flag.
for _alias in ('dry-run', 'dryrun', 'n'):
  if _alias not in flags.FLAGS:
    flags.DEFINE_alias(_alias, 'dry_run')
_GROUP = route_check._GROUP            # pylint: disable=protected-access

# enqueue flags
_POWER = flags.DEFINE_string(
    'power', None, 'Compute target: a shape like v7-32 / v6p-32, or a bare '
    'v5p-equivalent chip count. REQUIRED for enqueue.')
_ARCHS = flags.DEFINE_list(
    'archs', None, 'Comma-separated accelerator families this job accepts, '
    'e.g. --archs=v7,v6p. REQUIRED for enqueue.')
_TIER = flags.DEFINE_string('tier', 'PROD', 'PROD | BATCH.')
_METROS = flags.DEFINE_list(
    'metros', None, 'Optional metro allow-list, e.g. --metros=cbf,tul,lpp. '
    'Empty = any metro.')
_PRIORITY = flags.DEFINE_integer('priority', 0, 'Higher routes first.')
_POWER_TOL = flags.DEFINE_float(
    'power_tolerance', 0.5, 'Accept power within +/- half this fraction of the '
    'target (0.5 => [0.75x, 1.5x]).')
_MAX_PRICE = flags.DEFINE_float(
    'max_price', None, 'Skip cells pricier than this (credits/chip-hr).')
_TOPOLOGY_LOCKED = flags.DEFINE_bool(
    'topology_locked', False, 'Job resumes from a mesh-sharded checkpoint, so '
    'the router may only move it between SAME-geometry shapes (v6p-32<->v7-32, '
    'never v6e-32).')
_JOB_NAME = flags.DEFINE_string(
    'job_name', None,
    'The handle this run goes by in the local queue -- queue-status, the '
    'build-worker log, and dequeue all show it, and a resume or cancel refers '
    'to it. REQUIRED on enqueue: pass an informative, content-bearing name '
    'that says what the run IS. Matching exp_name is the convention '
    '(e.g. parcae-140m-torch-fix-nowtehead-dw-align); a bare power-hash names '
    'nothing and makes a queue of hundreds unreadable. On dequeue: the '
    'name(s), comma-separated, to remove.')
# Back-compat: --job_id was this flag's old name. Accept it (and the hyphen
# spellings) as aliases so existing scripts and muscle memory keep working --
# same pattern as --dry-run above.
for _jn_alias in ('job_id', 'job-id', 'job-name'):
  if _jn_alias not in flags.FLAGS:
    flags.DEFINE_alias(_jn_alias, 'job_name')
_LAUNCH = flags.DEFINE_list(
    'launch', None, 'Extra args passed VERBATIM to `tpu queue` at submit time, '
    'as k=v pairs: --launch=config=cfg.py,skip_preflight,force. A bare token '
    'becomes a bare flag (--skip_preflight).')
_WORKDIR = flags.DEFINE_string(
    'workdir', None, 'Directory the router runs `tpu queue` FROM, i.e. the '
    'checkout it packages into the stagedir. DEFAULTS to the current directory '
    'at enqueue time -- so enqueue from the checkout whose source/config this '
    "run needs. Only override if you know the source lives elsewhere. Pass ''"
    'to force the router process dir (safe ONLY if every diff is a --flag).')
_NO_SNAPSHOT = flags.DEFINE_bool(
    'no_snapshot', False,
    'Skip the enqueue-time snapshot and package the LIVE --workdir at build '
    'time (the OLD behavior). Use only when the checkout will not change before '
    'the build, or the tree is too large to copy. Default False: enqueue freezes '
    'a local copy immediately so edits made between enqueue and build cannot '
    'change what the job ships.')


def _parse_launch_kwargs(items: list[str] | None) -> dict:
  """['config=cfg.py', 'force'] -> {'config': 'cfg.py', 'force': True}."""
  out: dict = {}
  for item in items or []:
    if '=' in item:
      k, v = item.split('=', 1)
      out[k.strip()] = v
    else:
      out[item.strip()] = True
  return out


# Metros where the GROUP has a CNS storage registration, and the two where it
# does NOT. Mirrors xm_launcher.py:_METRO_STORAGE_CELL / _PERSONAL_ONLY_METROS.
# Literal copies ON PURPOSE: importing the launcher would drag xmanager into
# every enqueue. Safe because this is a REFUSAL list -- gaining an entry only
# refuses more, and losing one is still caught by the launcher's own gate.
# Verify: grep -A14 '_METRO_STORAGE_CELL = ' ~/work/tpu_cmd/xm_launcher.py
GROUP_STORAGE_METROS = frozenset({
    'cbf', 'ckv', 'cmh', 'dfw', 'grq', 'las', 'lpp', 'mrn', 'sin', 'tul',
})
PERSONAL_ONLY_METROS = frozenset({'phx', 'ske'})


# ---- enqueue-time snapshot -------------------------------------------------
# THE RACE THIS CLOSES: `tpu enqueue` records a workdir and returns instantly,
# but the build that rsyncs that directory into a CitC stagedir runs minutes to
# hours later, when a cell frees up. Anything edited in the source tree in that
# window silently changes the code a queued job ships -- a parked row "fires
# against a moved-on checkout". So enqueue now FREEZES a local copy of the
# checkout the instant it runs, and the builder packages THAT (route_lib
# .package_dir picks snapshot_dir over workdir). The frozen copy lives on local
# ext4: fast, and -- unlike the build-time stage-write -- it does NOT draw on
# the CitC CreateSnapshot token bucket, so a batch enqueue cannot storm it.

# Mirrors the build-time staging excludes in tpu_wrapper.sh EXACTLY, so the
# frozen copy is precisely the set of files the build would have packaged. `-aL`
# (below) dereferences symlinks into real files -- essential, because code dirs
# use relative symlinks to a shared source that would otherwise dangle inside
# the snapshot, or worse point back at the mutable original.
_STAGE_EXCLUDES = (
    'bazel-*', '.citc', '.git', '.jj', '.venv', '__pycache__',
    '*.npy', '*.npz', '*.ckpt', '*.pth', '*.pt', '*.safetensors',
    'data', 'logs', 'wandb',
)

# States in which a job STILL NEEDS its enqueue snapshot: everything before the
# build has produced its own durable CitC stagedir. Once SUBMITTED (build ran,
# XID exists, stagedir recorded in ~/.tpu_jobs.json) or terminal, the enqueue
# snapshot is redundant and the GC may reclaim it. BUILDING is KEPT -- a worker
# is rsyncing from it right now.
_SNAPSHOT_NEEDING_STATES = frozenset({
    route_lib.JobState.QUEUED,
    route_lib.JobState.BUILD_REQUESTED,
    route_lib.JobState.BUILDING,
    route_lib.JobState.HELD,
    route_lib.JobState.BUDGET_DEFERRED,
})

# A snapshot bigger than this is refused (override with --no_snapshot). Measured:
# real code checkouts are sub-GB after the excludes above (trm 0.8M, parcae
# 0.9G, elt_dit 5.5M); ~/work with all sibling checkouts is 3.4G and a google3
# root is enormous. 2 GiB cleanly separates a code subdir from the two
# directories people paste by mistake.
_SNAPSHOT_MAX_BYTES = 2 * 1024 * 1024 * 1024
_SNAPSHOT_DU_TIMEOUT_S = 30
_SNAPSHOT_RSYNC_TIMEOUT_S = 300
_SNAPSHOT_GC_GRACE_S = 900   # never reap an orphan dir younger than this: a
                             # concurrent enqueue may be mid-copy into it.


def _snapshot_root(queue_file: str) -> str:
  """Local-disk root for enqueue snapshots, operator-scoped BESIDE the queue
  file (so npu's snapshots never mingle with sqa's, exactly as the queue file
  itself is scoped). Overridable with $TPU_ENQUEUE_SNAPSHOT_ROOT."""
  env = os.environ.get('TPU_ENQUEUE_SNAPSHOT_ROOT', '').strip()
  if env:
    return env
  d = os.path.dirname(os.path.abspath(queue_file)) or '.'
  base = os.path.basename(queue_file)
  if base.endswith('_local_queue.json'):
    stem = base[:-len('_local_queue.json')]        # '.tpu' / '.npu'
    name = f'{stem}_enqueue_snapshots'
  else:
    name = base + '.enqueue_snapshots'
  return os.path.join(d, name)


def _dangerous_workdir(workdir: str) -> str | None:
  """A reason string if `workdir` must NOT be snapshotted wholesale, else None.

  Catches the two directories people paste by mistake: a home dir / filesystem
  root, and a CitC/bazel workspace root (417 top-level dirs). Packaging either
  is the documented `tpu enqueue` from ~/work trap; refusing at enqueue is the
  cheapest place to catch it. A code subdir inside a workspace trips none of
  these (the markers are only at the workspace ROOT).
  """
  wd = os.path.abspath(workdir)
  if wd == os.path.abspath(os.path.expanduser('~')):
    return 'it is your HOME directory'
  if os.path.dirname(wd) == wd:
    return 'it is a filesystem root'
  for marker in ('WORKSPACE', 'WORKSPACE.bazel', 'MODULE.bazel', '.citc'):
    if os.path.exists(os.path.join(wd, marker)):
      return (f'it is a workspace root (has {marker}) -- point --workdir at the '
              f'code subdirectory, not the checkout root')
  return None


def _dir_size_bytes(workdir: str, timeout_s: int) -> int | None:
  """Apparent size of `workdir` after the stage excludes, or None if it could
  not be measured in time. Bounded: a du that cannot finish fast means a
  pathological tree, which the caller treats as a refusal (fail closed)."""
  cmd = ['du', '-sb']
  for pat in _STAGE_EXCLUDES:
    cmd.append(f'--exclude={pat}')
  cmd.append(workdir)
  try:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
  except (subprocess.TimeoutExpired, OSError):
    return None
  if p.returncode != 0:
    return None
  try:
    return int(p.stdout.split(None, 1)[0])
  except (ValueError, IndexError):
    return None


def _safe_rmtree(root: str, name: str) -> bool:
  """rm -rf <root>/<name>, but ONLY if it is genuinely a direct child of root.
  Mirrors the wrapper's rm-rf assertion: this deletes, so it must never act on a
  path that lost its trailing component."""
  if not name or name in ('.', '..') or '/' in name or '\\' in name:
    return False
  root_abs = os.path.abspath(root)
  target = os.path.abspath(os.path.join(root, name))
  if os.path.dirname(target) != root_abs:
    return False
  if os.path.islink(target):
    try:
      os.remove(target)
      return True
    except OSError:
      return False
  if os.path.isdir(target):
    shutil.rmtree(target, ignore_errors=True)
    return True
  return False


def _make_enqueue_snapshot(queue_file: str, job_id: str,
                           workdir: str) -> tuple[str, str | None]:
  """Freeze a local copy of `workdir` for `job_id`. Returns (snapshot_dir, err);
  a non-None err means NO snapshot was created and the caller must decide.

  Fail-closed on every uncertainty: a partial or oversized copy is worse than no
  feature, because it looks faithful and ships wrong code -- the exact class of
  bug this closes. The caller turns any err into a REFUSED enqueue (with the
  --no_snapshot escape) rather than silently packaging the live checkout.

  Copies into a `<job_id>.staging.<pid>` temp dir and returns THAT; the caller
  renames it to the final `<job_id>` under the queue lock, after the duplicate
  check, so a rejected/duplicate enqueue never clobbers an existing job's
  snapshot.
  """
  bad = _dangerous_workdir(workdir)
  if bad:
    return '', f'refusing to snapshot {workdir}: {bad}'
  size = _dir_size_bytes(workdir, _SNAPSHOT_DU_TIMEOUT_S)
  if size is None:
    return '', (f'could not measure the size of {workdir} within '
                f'{_SNAPSHOT_DU_TIMEOUT_S}s (a very large or slow tree?)')
  if size > _SNAPSHOT_MAX_BYTES:
    return '', (f'{workdir} is {size / 1e9:.1f} GB after excludes, over the '
                f'{_SNAPSHOT_MAX_BYTES / 1e9:.0f} GB snapshot cap -- point '
                f'--workdir at a code subdir, or pass --no_snapshot')
  root = _snapshot_root(queue_file)
  try:
    os.makedirs(root, exist_ok=True)
  except OSError as e:
    return '', f'could not create snapshot root {root}: {e}'
  tmp_name = f'{job_id}.staging.{os.getpid()}'
  dest = os.path.join(root, tmp_name)
  _safe_rmtree(root, tmp_name)   # clear a leftover from a killed prior attempt
  cmd = ['rsync', '-aL']
  for pat in _STAGE_EXCLUDES:
    cmd.append(f'--exclude={pat}')
  cmd.append(workdir.rstrip('/') + '/')
  cmd.append(dest + '/')
  try:
    p = subprocess.run(cmd, capture_output=True, text=True,
                       timeout=_SNAPSHOT_RSYNC_TIMEOUT_S)
  except (subprocess.TimeoutExpired, OSError) as e:
    _safe_rmtree(root, tmp_name)
    return '', (f'snapshot rsync failed to start or hung past '
                f'{_SNAPSHOT_RSYNC_TIMEOUT_S}s: {e}')
  if p.returncode != 0:
    _safe_rmtree(root, tmp_name)   # never leave a partial snapshot behind
    tail = (p.stderr or p.stdout or '').strip()[-400:]
    return '', f'snapshot rsync exited {p.returncode}: {tail}'
  return dest, None


def _finalize_snapshot(queue_file: str, job_id: str, tmp_dir: str) -> str:
  """Rename the temp snapshot to its final `<root>/<job_id>` and return that.
  Called under the queue lock, after the duplicate check has passed."""
  root = _snapshot_root(queue_file)
  final = os.path.join(root, job_id)
  _safe_rmtree(root, job_id)      # a re-used --job_id: clean slate
  try:
    os.rename(tmp_dir, final)
    return final
  except OSError:
    # Rename failed (e.g. cross-dir); keep the temp dir as the snapshot rather
    # than lose the copy. package_dir reads whatever we store here.
    return tmp_dir


def _gc_orphan_snapshots(queue_file: str, entries, now: float) -> list[str]:
  """Reclaim snapshot dirs no live row still needs. Opportunistic, called from
  enqueue under the queue lock. Removes a dir iff its job_id is (a) present in
  the queue but PAST the snapshot-needing states (the build's own durable
  stagedir now exists), or (b) absent from the queue AND older than the grace
  window (so a snapshot a concurrent enqueue is still writing is never reaped).
  A BUILDING row's dir is always kept."""
  root = _snapshot_root(queue_file)
  if not os.path.isdir(root):
    return []
  keep = {e.job_id for e in entries if e.state in _SNAPSHOT_NEEDING_STATES}
  in_queue = {e.job_id for e in entries}
  removed: list[str] = []
  try:
    names = os.listdir(root)
  except OSError:
    return []
  for name in names:
    d = os.path.join(root, name)
    if not os.path.isdir(d) or name in keep:
      continue
    if name not in in_queue:
      # orphan OR mid-creation (incl. a `<id>.staging.<pid>` temp): the grace
      # window protects a fresh dir a concurrent enqueue is still filling.
      try:
        if now - os.path.getmtime(d) < _SNAPSHOT_GC_GRACE_S:
          continue
      except OSError:
        continue
    if _safe_rmtree(root, name):
      removed.append(name)
  return removed


def _remove_snapshot_for(queue_file: str, job_id: str) -> None:
  """Drop one job's snapshot (called on dequeue)."""
  _safe_rmtree(_snapshot_root(queue_file), job_id)


def _cmd_enqueue(argv: list[str]) -> int:
  if not _POWER.value or not _ARCHS.value:
    print('enqueue: --power and --archs are REQUIRED.\n'
          '  e.g. tpu enqueue --power=v7-32 --archs=v7,v6p '
          '--launch=config=configs/eqr.py', file=sys.stderr)
    return 2
  # ★REFUSE personal-only metros here, at the moment the human typed them.
  # phx / ske are metros the GROUP has no storage registration in, and they
  # fail differently from every other bad metro: an unknown metro makes
  # xm_launcher SystemExit into an inert zero-work-unit shell (visibly
  # broken), whereas phx/ske RESOLVE -- the launch proceeds, bills, and writes
  # to the personal 500 GiB quota (~468G used, handle poisoned) where the
  # write fails with resource_exhausted AND STILL LEAVES A 0-BYTE FILE. The
  # loss looks like a file that exists.
  # This is the earliest of three gates (here, jobchain.validate_enqueue for
  # the v2 store, and xm_launcher._local_bucket which nothing can bypass).
  # Earliest matters: refusing at enqueue costs zero credits and zero XIDs.
  personal = sorted({m.strip().lower() for m in (_METROS.value or [])}
                    & PERSONAL_ONLY_METROS)
  if personal:
    print(f'enqueue: REFUSED -- metro(s) {personal} have NO group storage '
          f'registration. Every write would land on the personal 500 GiB '
          f'per-cell quota (~468G used, poisoned): it fails with '
          f'resource_exhausted and still leaves a 0-byte file, so the job '
          f'looks like it produced output.\n'
          f'  Use a metro with group storage: '
          f'{", ".join(sorted(GROUP_STORAGE_METROS))}\n'
          f'  or pass an explicit group-billed bucket via '
          f'--launch=bucket=/cns/<cell>/... if you chose this on purpose.',
          file=sys.stderr)
    return 2
  # ★job_name is REQUIRED -- no auto-mint. An auto `<power>-<hash>` names
  # nothing, so a queue of hundreds is unreadable and it is easy to dequeue or
  # cancel the wrong arm. Force the human to say what the run IS, at the one
  # moment they know. (The local var stays `job_id`: it feeds the serialized
  # QueueEntry.job_id field, which is unchanged.)
  job_id = _JOB_NAME.value
  if not job_id:
    print('enqueue: --job_name is REQUIRED -- give this run an informative '
          'name.\n'
          '  It is the ONLY handle the run carries in queue-status, the '
          'build-worker log, and dequeue; a bare power-hash names nothing, so a '
          'queue of hundreds is unreadable and the wrong arm is easily '
          'dequeued.\n'
          '  Name what the run IS -- the convention is to match exp_name:\n'
          '    tpu enqueue ... '
          '--job_name=parcae-140m-torch-fix-nowtehead-dw-align\n'
          '  (--job_id is still accepted as the old name for this flag.)',
          file=sys.stderr)
    return 2
  # workdir default = the CWD at enqueue time, so enqueuing from the right
  # checkout just works. A flag value of '' explicitly opts into the router's
  # own dir (only safe when every difference is an explicit --flag).
  workdir = _WORKDIR.value if _WORKDIR.value is not None else os.getcwd()
  # ★ENQUEUE-TIME SNAPSHOT. Freeze a local copy of the checkout NOW, so edits
  # made between this enqueue and the (minutes-to-hours-later) build cannot
  # change the code this job ships. Done BEFORE the queue lock -- rsync is slow
  # and must not block other enqueues -- into a pid-scoped temp dir that is
  # renamed to its final name under the lock, once the duplicate check passes.
  snapshot_tmp = ''
  if _NO_SNAPSHOT.value:
    pass                       # opted out: the builder packages the live workdir
  elif not workdir:
    pass                       # flag-only run (workdir=''): nothing to snapshot
  elif not os.path.isdir(workdir):
    # A non-existent workdir is NOT a snapshot failure -- it is the existing
    # bad-workdir case the build-time guard already parks in HELD. Leave the
    # snapshot empty and let that path handle it unchanged, rather than refusing
    # here and changing behavior for a case that already fails loudly later.
    pass
  else:
    snapshot_tmp, snap_err = _make_enqueue_snapshot(
        _QUEUE_FILE.value, job_id, workdir)
    if snap_err:
      # FAIL CLOSED. Silently packaging the live checkout is the exact bug this
      # feature closes, so a snapshot we could not take REFUSES the enqueue and
      # names the one-flag escape rather than quietly reverting to old behavior.
      print(f'enqueue: REFUSED -- {snap_err}.\n'
            f'  The enqueue-time snapshot exists so edits between now and the '
            f'build cannot change the code this job ships. To package the LIVE '
            f'checkout at build time instead (the old behavior), re-run with '
            f'--no_snapshot.', file=sys.stderr)
      return 2
  # ★SAY WHICH POOL THIS WILL LAND IN WHEN NOBODY ASKED. `--group` defaults to
  # '9', but `.present` is what decides (see pin_group below), so NOT typing it
  # is not the same as typing 9: the row goes out unpinned and the router
  # applies its own g5/g3-first preference. That is the intended design -- the
  # preference exists so it can be used -- but it is invisible at the call site,
  # and g5/g3 have far less bidding power than g9 (measured 351/166 vs 22192),
  # so an unpinned PROD training job is preempted much sooner while every line
  # of output looks normal. A warning, NOT a refusal: refusing would delete the
  # operator's own preference, which is a bigger bug than the surprise.
  if not _GROUP.present:
    print('[enqueue] no --group given: this row stays UNPINNED and the router '
          'applies its g5/g3-first preference. g5/g3 have much lower bidding '
          'power than g9, so a PROD training job there is preempted sooner. '
          'Pass --group=9 to pin it.')
  entry = route_lib.QueueEntry(
      job_id=job_id,
      power=_POWER.value,
      allowed_archs=[a.strip() for a in _ARCHS.value],
      tier=_TIER.value,
      allowed_metros=[m.strip() for m in _METROS.value] if _METROS.value else None,
      priority=_PRIORITY.value,
      power_tolerance=_POWER_TOL.value,
      max_price=_MAX_PRICE.value,
      topology_locked=_TOPOLOGY_LOCKED.value,
      launch_kwargs=_parse_launch_kwargs(_LAUNCH.value),
      workdir=workdir,
      # ★READ `.present`, NOT `.value`. `--group` is route_check's flag (shared
      # via the import above) and it DEFAULTS TO '9', so `.value` is '9' for
      # every enqueue that never typed it -- storing that would pin the entire
      # fleet to the one pool with the hard income/10 bar and silently retire
      # the operator's g5/g3 preference. `.present` is the only thing that
      # distinguishes "the user asked for g9" from "nobody said anything".
      # Same trap as `--dry_run`, which absl also consumes before the
      # subcommand ever sees argv.
      pin_group=(_GROUP.value if _GROUP.present else None),
  )
  # The whole read-modify-write runs under the queue lock: load the LIVE queue,
  # check for a duplicate id, append, save -- atomically. Loading outside the
  # lock (the old bug) let a concurrent route tick's whole-queue overwrite land
  # between our load and save and silently drop this freshly enqueued row.
  gc_removed: list[str] = []
  with route_check.with_queue_lock(_QUEUE_FILE.value):
    entries = route_check.load_queue(_QUEUE_FILE.value)
    if any(e.job_id == job_id for e in entries):
      # This enqueue is rejected, so drop the temp snapshot we made for it
      # outside the lock -- otherwise a rejected enqueue leaks a copy.
      if snapshot_tmp:
        _safe_rmtree(_snapshot_root(_QUEUE_FILE.value),
                     os.path.basename(snapshot_tmp))
      print(f'enqueue: job_name {job_id!r} already in the queue; pass a '
            'different --job_name.', file=sys.stderr)
      return 1
    # Finalize the snapshot now the id is known-unique: rename the pid-scoped
    # temp dir to <root>/<job_id> and record it on the row. The builder reads it
    # via route_lib.package_dir (snapshot_dir wins over workdir).
    if snapshot_tmp:
      entry.snapshot_dir = _finalize_snapshot(_QUEUE_FILE.value, job_id,
                                              snapshot_tmp)
    entries.append(entry)
    route_check.save_queue(_QUEUE_FILE.value, entries)
    # Opportunistic cleanup while we hold the lock: reclaim snapshots no live
    # row still needs (a build has since produced its own durable stagedir, or
    # the row is gone). Bounded and best-effort; never touches a needed dir.
    gc_removed = _gc_orphan_snapshots(_QUEUE_FILE.value, entries, time.time())
  print(f'enqueued {job_id}: power={entry.power} archs={entry.allowed_archs} '
        f'tier={entry.tier} priority={entry.priority}'
        + (f' metros={entry.allowed_metros}' if entry.allowed_metros else '')
        + (' [topology-locked]' if entry.topology_locked else ''))
  # Echo the pin: a caller who typed --group must SEE that it took effect, or
  # the next silent no-op looks identical to the one this fixed.
  _pin = route_lib.pinned_group(entry)
  if _pin:
    print(f'  alloc group PINNED to g{_pin}: the router will submit under it '
          f'and skip its g5/g3-first preference for this job.')
  if entry.snapshot_dir:
    print(f'  packaged from: {entry.snapshot_dir}')
    print(f'    (FROZEN snapshot of {entry.workdir} taken now -- edits after '
          f'this will NOT change what this job builds)')
  elif _NO_SNAPSHOT.value:
    print(f'  packaged from: {entry.workdir or "(router process dir)"}  '
          f'[--no_snapshot: LIVE checkout rsynced at BUILD time]')
  else:
    print(f'  packaged from: {entry.workdir or "(router process dir)"}')
  if gc_removed:
    print(f'  (reclaimed {len(gc_removed)} stale enqueue snapshot(s))')
  print(f'  queue now holds {len(entries)} job(s). See: tpu queue-status')
  return 0


# States a dequeue refuses without --force: the job may still be live on the
# cluster, so dropping the row would orphan something nobody is watching.
# ★Deliberately NOT route_lib.TERMINAL_STATES, which includes RUNNING: that set
# means "the router stops tracking", the opposite of what is safe to delete.
# Everything absent here (DONE / FAILED, and any future terminal state) is a
# historical record and is safe to drop -- the cluster no longer has the job.
_DEQUEUE_UNSAFE_STATES = frozenset({
    'QUEUED',            # the router will place it
    'BUILD_REQUESTED',   # the serial builder will claim it
    'BUILDING',          # a worker is running its build right now
    'BUDGET_DEFERRED',   # auto-recovers to QUEUED when headroom opens
    'SUBMITTED',         # handed to XM, may already be starting
    'RUNNING',           # confirmed running on the cluster
    'HELD',              # parked by a human; the note is the warning itself
})


def _cmd_dequeue(argv: list[str]) -> int:
  """Remove queue entries — refusing the two states where removal does not stop
  the work, because the queue row is not the job.

  ★Dequeuing is bookkeeping, not cancellation. A BUILDING row has a worker
  running `tpu queue` for it RIGHT NOW; deleting the row does not signal that
  process, so the build finishes and submits an XID that no longer has any queue
  entry pointing at it — an orphan nobody is watching. Measured 2026-08-30:
  XID 284831213 ran 8xH100 for ~4 hours after its row was dequeued.
  A row that already HAS an xid is the same hazard after the fact: the job is on
  Borg, and removing the row only removes the evidence.

  ★And the natural check for "did my dequeue work?" cannot see this. `tpu
  queue-status | grep <id>` returning nothing is EXACTLY what a successful
  dequeue and a dequeue-that-submitted-anyway both look like, because a running
  job is not in the local queue either. So this command now prints the XID it
  knows about and what to run to actually stop it — the check has to be against
  XManager, never against the queue.
  """
  ids = set()
  if _JOB_NAME.value:
    ids |= {x.strip() for x in _JOB_NAME.value.split(',')}
  ids |= {a for a in argv[1:] if not a.startswith('-')}
  # ★Read `.present`, not `.value`: `.value` is False by default, which is the
  # right answer here, but the argv scan stays as the second route so `-f` and
  # a caller that never went through absl both keep working.
  force = (_FORCE.present and _FORCE.value) or '-f' in argv[1:]
  # ★--dry_run must PREVIEW, never delete (infra-v17, reported by elt-v5 which
  # ran it "to be safe" and watched the queue go 215 -> 214). The flag was not
  # parsed here at all: unknown flags are filtered out by the `not
  # a.startswith('-')` above, so it was silently swallowed and the delete ran
  # anyway -- printing "dequeued <id>" with rc=0. A preview that deletes is the
  # worst failure direction there is: the caller chose it BECAUSE they were
  # unsure, and every signal they get back says it worked.
  # ★Read the FLAG, not argv: route_check defines --dry_run, so absl consumes it
  # during parsing and it never reaches argv here. Scanning argv (the obvious
  # implementation) therefore never fires -- which is exactly how the flag came
  # to be silently ignored while the delete ran and printed rc=0.
  # `present` distinguishes "user typed it" from route_check's default of True:
  # keying off the value alone would turn EVERY dequeue into a no-op preview.
  dry_run = (route_check._DRY_RUN.present and route_check._DRY_RUN.value) or any(
      a in ('--dry_run', '--dry-run', '--dryrun', '-n') for a in argv[1:])
  if not ids:
    print('dequeue: pass job_id(s): tpu dequeue <job_id> [<job_id> ...]',
          file=sys.stderr)
    return 2
  refused: list[tuple[str, str, str]] = []   # (job_id, state, xid)
  with route_check.with_queue_lock(_QUEUE_FILE.value):
    entries = route_check.load_queue(_QUEUE_FILE.value)
    targets = [e for e in entries if e.job_id in ids]
    if not targets:
      print(f'dequeue: none of {sorted(ids)} found in the queue.', file=sys.stderr)
      return 1
    unsafe = set()
    if not force:
      for e in targets:
        # ★Compare the enum itself, never str(): JobState subclasses str, so
        # `e.state == 'BUILDING'` is True, but `str(e.state)` renders
        # 'JobState.BUILDING' and silently matches nothing. A guard written that
        # way passes review, reads correctly, and refuses nothing.
        raw = getattr(e, 'state', '')
        state = getattr(raw, 'value', raw) or ''
        xid = str(getattr(e, 'xid', '') or '')
        # ★REFUSE ON LIVE STATE, NOT ON "HAS AN XID". The guard used to read
        # `state == 'BUILDING' or xid`, which conflates "is on the cluster now"
        # with "was on the cluster once": FAILED / DONE / CANCELLED rows keep
        # their XID forever, so every row that ever launched became permanently
        # undeletable. Measured 2026-09-01: 268 queue rows, of which 190 FAILED
        # (58 older than 7 days) -- the board they render is mostly archaeology,
        # and a real alert has to be spotted among them.
        # The thing worth refusing is a job that is still live on the cluster,
        # which is exactly what the state field says. A terminal row's XID is a
        # historical fact, not a running process.
        if state in _DEQUEUE_UNSAFE_STATES:
          unsafe.add(e.job_id)
          refused.append((e.job_id, state, xid))
    keep = [e for e in entries if e.job_id not in (ids - unsafe)]
    removed = [e.job_id for e in targets if e.job_id not in unsafe]
    if removed and not dry_run:
      route_check.save_queue(_QUEUE_FILE.value, keep)
      # Drop each removed row's enqueue snapshot -- the row is gone, so nothing
      # will ever build from it. (A --force drop of a BUILDING row is the one
      # case a worker might still be rsyncing from it; harmless, the worker has
      # the source open and finishes, and the next enqueue's GC would reap it
      # anyway.)
      for r in removed:
        _remove_snapshot_for(_QUEUE_FILE.value, r)
  for r in removed:
    print(f'[DRY] would dequeue {r}' if dry_run else f'dequeued {r}')
  for job_id, state, xid in refused:
    print(f'dequeue: REFUSED {job_id} (state={state or "?"}'
          f'{", xid=" + xid if xid else ""}).', file=sys.stderr)
    if state == 'BUILDING' and not xid:
      print('  A worker is running its build NOW. Removing the row does not stop '
            'it: the build will finish and submit an XID with no queue entry '
            'behind it -- an orphan nobody is watching.', file=sys.stderr)
      print('  Wait for it to reach SUBMITTED (tpu queue-status), then cancel by '
            'XID; or --force to drop the row anyway and take responsibility for '
            'the XID it produces.', file=sys.stderr)
    elif state == 'HELD':
      print('  HELD is a human parking a job with a reason attached; the note IS '
            'the warning. Read it before dropping the row -- some record a trap '
            '(a workdir that rsyncs a whole depot) that will be re-hit by '
            'whoever re-enqueues. --force to drop it anyway.', file=sys.stderr)
    else:
      print(f'  This job is already on the cluster. Dequeuing removes the record, '
            f'not the job. Stop it with:  tpu cancel {xid}', file=sys.stderr)
  missing = ids - {e.job_id for e in targets}
  if missing:
    print(f'  (not found: {sorted(missing)})')
  if dry_run:
    print(f'  [DRY RUN] queue NOT modified; it still holds {len(entries)} job(s). '
          f'Re-run without --dry_run to act.')
  else:
    print(f'  queue now holds {len(keep)} job(s).')
  if refused:
    print('  ★Verify a cancellation against XManager, never against '
          '`tpu queue-status`: a job that left the queue and a job that was '
          'never stopped both show zero rows there.', file=sys.stderr)
    return 1
  return 0


def _cmd_requeue(argv: list[str]) -> int:
  """Return HELD job(s) to QUEUED after the operator has fixed the cause. With
  no id, requeues ALL held jobs."""
  ids = set()
  if _JOB_NAME.value:
    ids |= {x.strip() for x in _JOB_NAME.value.split(',')}
  ids |= {a for a in argv[1:] if not a.startswith('-')}
  with route_check.with_queue_lock(_QUEUE_FILE.value):
    entries = route_check.load_queue(_QUEUE_FILE.value)
    held = [e for e in entries if e.state == route_lib.JobState.HELD]
    if not held:
      print('requeue: no HELD jobs.', file=sys.stderr)
      return 1
    targets = [e for e in held if (not ids or e.job_id in ids)]
    if not targets:
      print(f'requeue: none of {sorted(ids)} are HELD. Held: '
            f'{[e.job_id for e in held]}', file=sys.stderr)
      return 1
    for e in targets:
      route_lib.requeue_held(e)
      print(f'requeued {e.job_id} (HELD -> QUEUED)')
    route_check.save_queue(_QUEUE_FILE.value, entries)
  print(f'  {len(targets)} job(s) back in the queue.')
  return 0


# ANSI helpers for the status board.
def _c(code: str, s: str) -> str:
  return f'\x1b[{code}m{s}\x1b[0m'


def _cmd_status(argv: list[str]) -> int:
  entries = route_check.load_queue(_QUEUE_FILE.value)
  if not entries:
    print('local queue is empty. Add one: tpu enqueue --power=v7-32 '
          '--archs=v7,v6p --launch=config=...')
    return 0

  # Run a dry-run planning tick to compute a live placement reason per job.
  reason_by_id: dict[str, str] = {}
  placed_ids: set[str] = set()
  try:
    provider = avail_provider.AvailabilityProvider(group=_GROUP.value)
    avail_by_cell, arch_price, arch_pool = provider.fetch()
    placements = route_lib.select_and_plan(
        entries, avail_by_cell, time.time(),
        arch_price=arch_price, arch_pool=arch_pool)
    for p in placements:
      reason_by_id[p.job_id] = p.reason
      placed_ids.add(p.job_id)
    live = True
  except Exception as e:  # pylint: disable=broad-except
    print(_c('33', f'[queue-status] live availability unavailable ({e}); '
             'showing stored state only.'))
    live = False

  print(_c('1;36', f'━━ Local Queue ({len(entries)} job(s)) ━━'))
  hdr = f'{"JOB_ID":22s} {"STATE":9s} {"POWER":9s} {"ARCHS":14s} {"PRIO":>4s}  WHY / PLACEMENT'
  print(_c('1;35', hdr))
  # QUEUED first (by priority desc), then the rest.
  def sort_key(e):
    return (0 if e.state == route_lib.JobState.QUEUED else 1, -e.priority,
            e.job_id)
  for e in sorted(entries, key=sort_key):
    archs = ','.join(e.allowed_archs)
    if e.state == route_lib.JobState.QUEUED:
      if e.job_id in placed_ids:
        why = _c('32', 'PLACEABLE now: ' + reason_by_id[e.job_id])
      elif live:
        why = _c('33', 'waiting: no placeable cell (oversold/full/cooled-down)')
      else:
        why = e.last_reason or '(availability unknown)'
      state_disp = _c('36', e.state.value)
    elif e.state == route_lib.JobState.BUILDING:
      why = _c('35', f'building now (worker {e.worker_id or "?"}): {e.last_reason}')
      state_disp = _c('1;35', e.state.value)
    elif e.state == route_lib.JobState.HELD:
      why = _c('31', f'{e.last_reason}  -> fix + `tpu enqueue` again, or `tpu queue-status` after re-enqueue')
      state_disp = _c('1;31', e.state.value)
    elif e.state == route_lib.JobState.SUBMITTED:
      why = f'xid={e.xid} cell={e.cell} {e.arch}-{e.chips}; {e.last_reason}'
      state_disp = _c('34', e.state.value)
    else:
      why = e.last_reason or ''
      state_disp = e.state.value
    lock = _c('35', ' [lock]') if e.topology_locked else ''
    snap = _c('32', ' [snap]') if getattr(e, 'snapshot_dir', '') else (
        _c('33', ' [live]') if not getattr(e, 'xid', '') else '')
    print(f'{e.job_id:22s} {state_disp:18s} {e.power:9s} {archs:14s} '
          f'{e.priority:>4d}  {why}{lock}{snap}')
    # ★Show the packaged directory so the frozen snapshot is never lost from the
    # record (operator: "job status 里面需要记录每个 job 的 stagedir 别丢了").
    # snapshot_dir is the enqueue-time frozen copy; once the build has run its
    # own durable CitC stagedir lives in ~/.tpu_jobs.json (shown by `tpu check`).
    pkg = route_lib.package_dir(e)
    if getattr(e, 'snapshot_dir', ''):
      print(_c('2', f'{"":22s} packaged from (frozen): {pkg}'))
    elif pkg:
      print(_c('2', f'{"":22s} packaged from (live at build): {pkg}'))
  n_q = sum(1 for e in entries if e.state == route_lib.JobState.QUEUED)
  n_placeable = len(placed_ids)
  print()
  print(_c('2', f'{n_q} queued, {n_placeable} placeable this tick. The router '
           'submits placeable jobs on its next pass; a queued job costs nothing.'))
  return 0


def main(argv):
  # argv[0] is the program; argv[1] is the subcommand the wrapper forwards.
  if len(argv) < 2:
    print('usage: queue_cli {enqueue|queue-status|dequeue} [flags]',
          file=sys.stderr)
    return 2
  cmd = argv[1]
  rest = [argv[0]] + argv[2:]
  if cmd in ('enqueue', 'add'):
    return _cmd_enqueue(rest)
  if cmd in ('queue-status', 'status', 'qs'):
    return _cmd_status(rest)
  if cmd in ('dequeue', 'remove', 'rm'):
    return _cmd_dequeue(rest)
  if cmd in ('requeue', 'unhold'):
    return _cmd_requeue(rest)
  print(f'unknown subcommand {cmd!r}; expected '
        'enqueue|queue-status|dequeue|requeue', file=sys.stderr)
  return 2


if __name__ == '__main__':
  sys.exit(app.run(main))
