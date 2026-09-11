r"""Local-queue router tick: drain the local queue into the XM queue.

WHAT IT DOES, once per invocation (the daemon calls it on a loop):

  1. load the durable local queue (~/.tpu_local_queue.json),
  2. fetch live availability (avail_provider: free chips + price + pool),
  3. plan placements (route_lib.select_and_plan: priority + fairness, cheapest
     effective-price type, most-placeable cell, never into an oversold/full one),
  4. for each planned placement, submit via `tpu queue` (pinned --cell) and
     record the XID -- OR, in --dry_run (the DEFAULT), just print the plan.

WHY IT SHELLS OUT TO `tpu queue` rather than launching itself: `tpu queue`
already owns stagedir snapshotting, preflight, limit-order caps, the locality
guard, the ~/.tpu_jobs.json registration, and the ANSI-proof XID capture. The
router is the SCHEDULER on top; it must not re-implement the launcher.

Reads/writes only ~/.tpu_local_queue.json. Submuts are side effects behind a
`Submitter` seam so the whole tick is unit-testable with a fake submitter, and
default is dry-run so a first live run shows the plan before touching XM.
"""

from __future__ import annotations

import contextlib
import dataclasses
import fcntl
import getpass
import json
import os
import re
import shlex
import subprocess
import tempfile
import time
import uuid
from typing import Callable, Optional, Protocol

from absl import app
from absl import flags

from google3.experimental.users.qiaos.tpu_utils import avail_provider
from google3.experimental.users.qiaos.tpu_utils import cell_locality
from google3.experimental.users.qiaos.tpu_utils import route_lib


class _Provider(Protocol):
  """What run_tick needs from an availability source (AvailabilityProvider, or
  a fake in tests): one method returning (avail_by_cell, arch_price, arch_pool)."""

  def fetch(self) -> tuple[dict[str, route_lib.CellAvail], dict[str, float],
                           dict[str, float]]:
    ...


class _Submitter(Protocol):
  """What run_tick needs to submit: shells out to `tpu queue` in production, a
  recorder in tests. `cwd` is the checkout `tpu queue` packages from."""

  def submit(self, argv: list[str], cwd: str = '') -> tuple[Optional[str], str]:
    ...

  def cancel(self, xid: str) -> tuple[bool, str]:
    ...

  def find_xid_by_name(self, exp_name: str,
                       timeout_s: float = 120.0) -> tuple[Optional[str], str]:
    """Newest XID whose experiment name matches EXACTLY, or (None, why).

    Part of the protocol because adopt_escaped_builds depends on it, and on the
    WORDING of its second element: a caller must be able to tell 'the lookup ran
    and saw nothing' (safe to rebuild) from 'the lookup could not run' (must not
    rebuild -- the experiment may be there and unseen).
    """
    ...


# Live scheduling states for a submitted XID, collapsed to what re-route needs.
STATUS_PENDING = 'PENDING'     # still in the auction -- the re-route trigger
STATUS_RUNNING = 'RUNNING'     # scheduled/coming up/running -- leave it alone
STATUS_TERMINAL = 'TERMINAL'   # ended BADLY (failed/stopped/cancelled) -- zombie
STATUS_COMPLETED = 'COMPLETED'  # ended WELL (ran to completion) -- NOT a failure
STATUS_UNKNOWN = 'UNKNOWN'     # probe failed -- do NOT act (never cancel blind)
STATUS_GONE = 'GONE'           # experiment resolves but has ZERO work units

# NOTE: TERMINAL and COMPLETED were ONE constant until it was found that
# reconcile wrote FAILED for every job that merely finished: 105 of 227 queue
# rows carried 'zombie cleaned up' and 100% of them read FAILED, including runs
# with confirmed results. The distinction is made HERE, at the probe, because
# once is_failed/is_completed are OR-ed together the information is gone and no
# downstream decision can recover it.


class _StatusProbe(Protocol):
  """Returns the collapsed live state of one XID (one of STATUS_*). Backed by
  XManager in production, scripted in tests."""

  def status(self, xid: str) -> str:
    ...


class _OutputProbe(Protocol):
  """Returns the newest mtime (epoch seconds) under a job's output dir, or None
  if the dir is missing / the lookup failed. Backed by a CNS listing in
  production, scripted in tests. A None NEVER means 'alive' -- it means 'no disk
  evidence', so the caller falls back to the two-sample probe."""

  def latest_mtime(self, entry: 'route_lib.QueueEntry') -> Optional[float]:
    ...


# ★Re-route timestamps for the global brake, PERSISTED. An in-memory list would
# be correct for one --reroute_loop process, but that process died 4 times in a
# 90-second window earlier today; a brake whose history resets on restart is
# exactly useless in the situation that restarts it. Small append-only JSON,
# pruned to the window on every read.
REROUTE_HISTORY_FILE = os.path.expanduser('~/.tpu_reroute_history.json')
_RECENT_REROUTES: list[float] = []


def _load_reroute_history(path: str = REROUTE_HISTORY_FILE,
                          window_s: float = 3600.0) -> list[float]:
  """Timestamps inside the window. A missing/corrupt file reads as EMPTY, which
  fails OPEN (brake disengaged): losing the history must not silently suspend
  re-routing fleet-wide."""
  try:
    with open(path) as f:
      raw = json.load(f)
    cutoff = time.time() - window_s
    return [float(t) for t in raw if float(t) >= cutoff]
  except (OSError, ValueError, TypeError):
    return []


def _save_reroute_history(times: list[float],
                          path: str = REROUTE_HISTORY_FILE) -> None:
  """Atomic write (tmp+rename): a torn read must never look like 'no churn'."""
  try:
    d = os.path.dirname(path) or '.'
    fd, tmp = tempfile.mkstemp(dir=d, prefix='.reroute_hist.')
    with os.fdopen(fd, 'w') as f:
      json.dump(times[-500:], f)
    os.replace(tmp, path)
  except OSError:
    pass  # history is best-effort; never break re-routing over it


DEFAULT_QUEUE_FILE = os.path.expanduser('~/.tpu_local_queue.json')
# The wrapper defining the `tpu` shell function; we source it, then call `tpu`.
TPU_WRAPPER = os.path.expanduser('~/work/tpu_cmd/tpu_wrapper.sh')
# The XManager CLI, by ABSOLUTE path: `xmanager` is a shell function and
# `xmanager.par` is not on PATH, so a bare name exits 127 -- which, behind a
# pipe, is indistinguishable from "the experiment does not exist".
XMANAGER_PAR = '/google/bin/releases/xmanager/cli/xmanager.par'
DEFAULT_GROUP = '9'
# ★Preference order for placement (operator standing order 2026-08-31): the
# operator's OWN pools g5/g3 first -- they are exempt from the G9 income/10 cap
# and self-limit at 100% of their own income (an overspend there parks itself,
# nobody is on the hook) -- and only then the g9 floor, whose 1/10 ceiling is
# the one a human answers for. "优先" means TRY IN THIS ORDER, per job.
DEFAULT_GROUP_ORDER = ['5', '3', '9']

# Same acceptance the wrapper uses (ANSI-stripped): XManager prints "Launched
# experiment <id>" on create and "Added N work unit(s) to experiment <id>" on
# resume. Strip color first -- xmanager sometimes colorizes the id.
_XID_RE = re.compile(r'(?:Launched experiment|work unit\(s\) to experiment)\s+(\d+)')
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')


_QUEUE_FILE = flags.DEFINE_string(
    'queue_file', DEFAULT_QUEUE_FILE, 'Local queue JSON file.')
_DRY_RUN = flags.DEFINE_bool(
    'dry_run', True, 'Plan only; do NOT submit. Default True -- pass '
    '--nodry_run to actually submit.')
_GROUP = flags.DEFINE_string('group', DEFAULT_GROUP, 'Alloc group for submits.')
_GROUP_ORDER = flags.DEFINE_string(
    'group_order', None,
    'Comma-separated alloc groups to TRY IN ORDER for placement, e.g. "5,9": '
    'prefer the cheaper/free vqfree pool (g5) and only fall back to the g9 floor '
    'for jobs g5 cannot place this tick. Each group is tried with its own live '
    'availability; a job placed under an earlier group is no longer QUEUED so '
    'later groups only see the remainder. None = single-group behaviour (--group).')
_MAX_PLACEMENTS = flags.DEFINE_integer(
    'max_placements', None, 'Cap placements this tick (None = no cap).')
_VERBOSE = flags.DEFINE_bool('verbose', True, 'Print per-job planning detail.')
_REROUTE = flags.DEFINE_bool(
    'reroute', False, 'Instead of placing QUEUED jobs, sweep SUBMITTED jobs '
    'still PENDING past --reroute_after_s: cancel and return them to QUEUED so '
    'the next tick re-places them on a cell that can actually schedule.')
_REROUTE_AFTER_S = flags.DEFINE_float(
    'reroute_after_s', 600.0, 'A SUBMITTED job still PENDING this many seconds '
    'after submit is cancelled and re-routed (operator default: 10 min).')
_COOLDOWN_S = flags.DEFINE_float(
    'cooldown_s', 7200.0, 'After a re-route, cool the stuck cell AND the arch '
    'for this long (operator 2026-09-10: 30min -> 2h). The old 30min window '
    'expired before the backlogged serial build-worker re-dispatched the job, '
    'so the penalty read as gone at the moment it should have applied. Feeds '
    'both cooldown_cells (decaying) and cooldown_archs (flat, stacking).')
_CONFIRM_GAP_S = flags.DEFINE_float(
    'reroute_confirm_gap_s', 15.0, 'Before cancelling a PENDING job, wait this '
    'long and re-probe; only cancel if STILL pending. Clears BATCH shadow-WU '
    'gaps that momentarily read all-pending.')
_FRESH_OUTPUT_S = flags.DEFINE_float(
    'reroute_fresh_output_s', 1200.0, 'A job whose output dir was written within '
    'this many seconds is judged ALIVE and never re-routed, whatever XManager '
    'says (disk evidence beats a PENDING snapshot). Default 20 min > a BATCH '
    'work-unit segment.')
_NOMINAL_RUNNING_GRACE_S = flags.DEFINE_float(
    'reroute_nominal_running_grace_s', 3600.0, 'How long a row may sit RUNNING '
    'with NO Borg VM group in RUN and nothing ever written before it is treated '
    'as stuck and re-routed. XManager reports RUNNING for a job whose VM groups '
    'never left PENDING; that shape burned 12 h on one XID with zero output. '
    'Only the full conjunction acts (Borg answered + no group RUN + nothing '
    'written + past this grace); any doubt promotes as before.')
# --- Step2: XM-truth reconcile + the standalone reroute-loop process --------
_RECONCILE = flags.DEFINE_bool(
    'reconcile', False, 'Instead of placing QUEUED jobs, run ONE XM-truth '
    'reconcile pass: re-verify every RUNNING/SUBMITTED/BUILDING entry against '
    'XManager and rewrite zombies (XM terminal) to FAILED, promote XM-running '
    'SUBMITTED to RUNNING. Fixes R3 (stale local state poisoning the route path). '
    'Read-only against XM; only local queue rows change. UNKNOWN never acts.')
_AUTO_RESUME_PRUNED = flags.DEFINE_bool(
    'auto_resume_pruned', False, 'In the reconcile pass, when a row is cleaned '
    'up to FAILED, decide whether it was PRUNED/PREEMPTED (a healthy run killed '
    'from outside: XM terminal + a surviving checkpoint + no code-bug signature '
    'in the log tail + no live same-config sibling) and, if so, auto-append a '
    'fresh QUEUED row that resumes it WARM from the last checkpoint. OFF by '
    'default: turning it on makes reconcile SUBMIT work, so it must be an '
    'explicit choice, and it honours --dry_run (logs the plan, appends nothing). '
    'Every guard defaults to leaving the dead row alone; see '
    'route_lib.plan_pruned_restart.')
_AUTO_RESUME_MAX = flags.DEFINE_integer(
    'auto_resume_max', 3, 'Max automatic warm-restarts of one run before the '
    'reconcile pass stops and leaves it for a human (carried on the entry as '
    'auto_resumes). Guards against a run that dies for a non-pruner reason the '
    'code-bug scan missed looping on PROD budget.')
_REROUTE_LOOP = flags.DEFINE_bool(
    'reroute_loop', False, 'Run as the STANDALONE tpu-reroute PROCESS: an own '
    'loop that each round does (A) an XM-truth reconcile pass then (B) a reroute '
    'pass, polling every --reroute_loop_poll_s. This is the design\'s separate '
    'reroute process -- never blocked by the builder, so B\'s pending>deadline '
    'safety net always runs. Combine with the daemon set to skip its in-lane '
    'reroute (TPU_ROUTE_INLANE_REROUTE=0) so reroute does not double-run.')
_REROUTE_LOOP_POLL_S = flags.DEFINE_float(
    'reroute_loop_poll_s', 120.0, 'Poll interval for the standalone --reroute_loop '
    'process (design default ~120s). Reconcile+reroute each round.')
_WORKER = flags.DEFINE_bool(
    'worker', False, 'Run as the SERIAL build-worker loop: claim one QUEUED job '
    'at a time as BUILDING, run `tpu queue` for it, record the result, repeat. '
    'Only ever one build in flight -- the cure for concurrent-build failures.')
_DISPATCH_WORKER = flags.DEFINE_bool(
    'dispatch_worker', False, 'Step3: run the REWRITTEN worker = router-dispatch '
    '+ serial builder in ONE loop. Each round: promote budget-deferred, '
    'backpressure-gate, greedy plan_dispatch under XM-truth headroom (mark '
    'BUILD_REQUESTED/BUDGET_DEFERRED), then serially build one BUILD_REQUESTED. '
    'Replaces --worker at go-live; the daemon in-lane place pass is gated off so '
    'this is the ONLY drainer (kills R1). Budget refusal -> BUDGET_DEFERRED, not '
    'a build failure (kills R2).')
_WORKER_POLL_S = flags.DEFINE_float(
    'worker_poll_s', 15.0, 'Worker idle poll interval when the queue is empty.')
_BUILD_STALE_S = flags.DEFINE_float(
    'build_stale_s', 1800.0, 'A BUILDING claim older than this is treated as a '
    'crashed worker and reclaimed to QUEUED (a real build is minutes).')
_SRCFS_FAIL_BRAKE = flags.DEFINE_integer(
    'srcfs_fail_brake', 20, 'If this many NEW srcfs/CreateSnapshot failures '
    'appear between two polls, skip claiming a build this round (the CitC token '
    'bucket is draining). 0 disables the brake.')
_MAX_BUILD_ATTEMPTS = flags.DEFINE_integer(
    'max_build_attempts', 3, 'After this many failed build attempts a job is '
    'moved to HELD instead of requeued forever, so one bad job cannot churn the '
    'worker and starve the rest. A human re-enqueues it once fixed.')


# --- queue persistence (atomic, flock'd, like ~/.tpu_jobs.json) -----------
def load_queue(path: str) -> list[route_lib.QueueEntry]:
  """Read the local queue. Missing/empty file = empty queue."""
  if not os.path.exists(path):
    return []
  try:
    with open(path) as f:
      fcntl.flock(f, fcntl.LOCK_SH)
      try:
        raw = json.load(f)
      finally:
        fcntl.flock(f, fcntl.LOCK_UN)
  except (OSError, ValueError):
    return []
  entries = raw.get('entries', raw) if isinstance(raw, dict) else raw
  out = []
  for d in entries:
    try:
      out.append(route_lib.QueueEntry.from_dict(d))
    except (TypeError, ValueError):
      continue
  return out


def save_queue(path: str, entries: list[route_lib.QueueEntry]) -> None:
  """Write the queue atomically: temp file + flock + rename."""
  tmp = f'{path}.tmp.{os.getpid()}'
  payload = {'entries': [e.to_dict() for e in entries],
             'updated': time.time()}
  with open(tmp, 'w') as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    try:
      json.dump(payload, f, indent=2)
      f.flush()
      os.fsync(f.fileno())
    finally:
      fcntl.flock(f, fcntl.LOCK_UN)
  os.replace(tmp, path)


# --- atomic claim / update (the cross-process serial lock) -----------------
# The single-build invariant must hold across SEPARATE worker processes, not
# just within one. load_queue+save_queue each take the lock briefly, so a
# read-modify-write done as two calls has a window where two workers both see
# 'no build in flight' and both claim. So the CLAIM is one read-modify-write
# under ONE held exclusive lock on a sidecar lockfile.
def _lockfile(path: str) -> str:
  return f'{path}.lock'


@contextlib.contextmanager
def with_queue_lock(path: str):
  """Hold the cross-process exclusive lock on the queue's sidecar lockfile for
  the whole `with` block.

  Every read-modify-write of the queue MUST happen inside this block so that a
  load...mutate...save is atomic against other processes. The old pattern --
  load_queue() then (much later) save_queue() as two separate calls -- each took
  the lock only briefly, leaving a wide window in which another writer's save
  clobbered entries added in between. That window is exactly how a `tpu enqueue`
  landing during a slow route tick vanished: the tick wrote back its stale
  in-memory snapshot over the freshly enqueued row.

  The lock is a SEPARATE sidecar file ('{path}.lock'), never the queue file
  itself, because save_queue swaps the queue in via os.replace -- a lock taken
  on the queue inode would be lost at the rename. load_queue/save_queue take no
  lock of their own; callers serialize through this manager (or the higher-level
  helpers below that wrap it).
  """
  lock_path = _lockfile(path)
  with open(lock_path, 'w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
      yield
    finally:
      fcntl.flock(lock, fcntl.LOCK_UN)


def merge_and_save_touched(
    path: str,
    touched: list[route_lib.QueueEntry],
    dropped_job_ids: Optional[set] = None,
) -> None:
  """Re-load the queue under the lock and write back only the entries this pass
  actually changed, keyed by job_id, then persist atomically.

  This is the write half of the route pass fix. The slow part of a tick (the
  RPCs in run_tick/run_reroute) runs on an in-memory snapshot WITHOUT the lock,
  so enqueue is never blocked for minutes. Only the fast merge-write is locked:
  we re-read the live queue (which may now contain rows enqueued during the
  RPCs), overwrite just the job_ids we touched with our updated copies, drop any
  the pass removed, and leave every other row -- including newly enqueued ones
  -- exactly as found. So a concurrent enqueue can no longer be clobbered by a
  tick writing back a stale whole-queue snapshot.

  Ordering: preserve the live queue's order for rows that already existed;
  append touched rows that are new (shouldn't happen for route, but harmless).
  """
  touched_by_id = {e.job_id: e for e in touched}
  dropped = dropped_job_ids or set()
  with with_queue_lock(path):
    live = load_queue(path)
    merged: list[route_lib.QueueEntry] = []
    seen = set()
    for e in live:
      if e.job_id in dropped:
        seen.add(e.job_id)
        continue
      if e.job_id in touched_by_id:
        t = touched_by_id[e.job_id]
        # ★Do NOT let a stale snapshot erase the ROUTER's group choice.
        # This pass read its snapshot before its (slow) XM RPCs; meanwhile the
        # dispatch worker may have admitted the row under g5/g3 and written
        # `group` onto it. Our copy still has the pre-RPC value (None), so a
        # blind overwrite silently reverts the placement preference -- measured
        # 2026-08-31 02:1xZ: dispatch logged "-> group g5" for three cars and
        # the reconcile pass wrote all three back as group=None.
        # A field the pass does not own must be carried over from the LIVE row.
        if getattr(t, 'group', None) is None and getattr(e, 'group', None):
          t.group = e.group
        # Same reasoning for the caller's PIN, which no pass owns either: a
        # reconcile snapshot taken before `tpu enqueue --group=` landed would
        # otherwise write the row back unpinned, and the pin would evaporate
        # between the moment the caller set it and the first dispatch round.
        if (getattr(t, 'pin_group', None) is None
            and getattr(e, 'pin_group', None)):
          t.pin_group = e.pin_group
        merged.append(t)
      else:
        merged.append(e)
      seen.add(e.job_id)
    # Touched rows that were not in the live queue (newly created by the pass).
    for e in touched:
      if e.job_id not in seen:
        merged.append(e)
    save_queue(path, merged)


def claim_next_build(path: str, now: float, worker_id: str,
                     stale_after_s: float,
                     pick: 'Optional[Callable[[list[route_lib.QueueEntry]], Optional[route_lib.QueueEntry]]]' = None
                     ) -> Optional[route_lib.QueueEntry]:
  """Atomically: reclaim stale BUILDING, then IF no live build is in flight,
  mark the next claimable entry BUILDING and persist. Returns the claimed entry
  (a copy reflecting the persisted state) or None if nothing was claimed
  (queue empty, or a build already in flight). Serialized by an exclusive
  flock held across the whole read-modify-write.

  `pick` selects which entry to claim from the loaded list; default
  route_lib.next_queued (the old single-drainer behavior: claim from QUEUED).
  The Step3 dispatch worker passes route_lib.next_build_requested so the builder
  claims only what the router already dispatched this round (BUILD_REQUESTED),
  never re-picking a raw QUEUED job the router has not yet budget-admitted."""
  if pick is None:
    pick = route_lib.next_queued
  with with_queue_lock(path):
    entries = load_queue(path)
    route_lib.reclaim_stale_building(entries, now, stale_after_s)
    if not route_lib.can_claim_build(entries, now, stale_after_s):
      save_queue(path, entries)   # persist any reclaim even if we don't claim
      return None
    nxt = pick(entries)
    if nxt is None:
      save_queue(path, entries)
      return None
    route_lib.claim_for_build(nxt, now, worker_id)
    save_queue(path, entries)
    return nxt


def update_entry(path: str, job_id: str,
                 mutate: 'Callable[[route_lib.QueueEntry], None]') -> bool:
  """Atomically apply `mutate` to the entry with `job_id` and persist. Returns
  True if the entry was found. Used to write the post-build result (SUBMITTED,
  or back to QUEUED) without clobbering concurrent edits to other entries."""
  with with_queue_lock(path):
    entries = load_queue(path)
    found = False
    for e in entries:
      if e.job_id == job_id:
        mutate(e)
        found = True
        break
    if found:
      save_queue(path, entries)
    return found


# --- pure helpers ---------------------------------------------------------
def _decode_stream(s) -> str:
  """`TimeoutExpired.stdout` is bytes even when the run asked for text."""
  if s is None:
    return ''
  if isinstance(s, bytes):
    return s.decode('utf-8', 'replace')
  return str(s)


def _exp_name_of(argv: list[str]) -> Optional[str]:
  """The `--exp_name=` value in a `tpu queue` argv, or None."""
  for a in argv or []:
    if a.startswith('--exp_name='):
      return a.split('=', 1)[1]
  return None


def build_tpu_queue_cmd(placement: route_lib.Placement,
                        entry: route_lib.QueueEntry,
                        group: str = DEFAULT_GROUP) -> list[str]:
  """The `tpu queue ...` argv for one placement. Pure and inspectable.

  Pins the router's chosen cell and shape; passes tier and any launch_kwargs
  (config, exp_name, ...) through verbatim. launch_kwargs values of None/True
  become bare flags (`--flag`); everything else becomes `--k=v`.
  """
  argv = ['tpu', 'queue',
          f'--tpu_type={placement.arch}-{placement.chips}',
          f'--group={group}',
          f'--cell={placement.cell}']
  if entry.tier:
    argv.append(f'--tier={entry.tier}')
  for k, v in (entry.launch_kwargs or {}).items():
    # ★`group` is the ROUTER's decision, not a passthrough. Many rows carry a
    # stale launch_kwargs['group']='9' from whoever enqueued them (88 of 225
    # rows, 2026-08-31), and emitting it here appends a SECOND --group= after
    # the router's -- so the caller's g9 silently wins and the operator's
    # "prefer g5/g3" preference is a no-op for exactly those jobs. The router
    # already resolved the group (g5 -> g3 -> g9) against each pool's live
    # budget gate; drop the passthrough copy rather than emit a duplicate flag.
    if k.lstrip('-') == 'group':
      continue
    flag = k if k.startswith('--') else f'--{k}'
    if v is None or v is True:
      argv.append(flag)
    elif v is False:
      continue
    else:
      argv.append(f'{flag}={v}')
  return argv


_STDERR_MARK = '\n\x00--stderr--\n'
"""Separates the stdout and stderr halves inside a submit's combined output.
NUL cannot occur in either stream's text, so the split is unambiguous."""


def extract_xid(output: str) -> Optional[str]:
  """The XID from `tpu queue` output, ANSI-stripped, or None. Same rule as the
  wrapper: accept both the create line and the resume 'work unit(s)' line."""
  m = _XID_RE.search(_ANSI_RE.sub('', output or ''))
  return m.group(1) if m else None


def is_budget_deferral(output: str) -> bool:
  """True iff `tpu queue` output carries the budget-check marker `[[BUDGET_DEFERRED]]`
  on its own line (post-ANSI-strip). budget_check.py (wiki_agent-owned) prints
  this stable, ANSI-free line when a submit is over the G9 bar. The worker greps
  it to tell 'no XID because budget refused' (a transient FLEET state -> park
  BUDGET_DEFERRED, NO attempt++) apart from 'no XID because the build crashed'
  (a real per-job defect -> attempt++, eventually HELD). This is the R2 fix: a
  budget refusal must never be punished as a build failure."""
  clean = _ANSI_RE.sub('', output or '')
  return any(line.strip() == '[[BUDGET_DEFERRED]]' for line in clean.splitlines())


# --- the submit seam ------------------------------------------------------
class Submitter:
  """Runs `tpu queue` by sourcing the wrapper first (the `tpu` function is
  not on PATH). Injectable so the tick is testable without launching anything.
  Returns (xid_or_None, combined_output)."""

  def __init__(self, wrapper_path: str = TPU_WRAPPER, timeout_s: float = 1800.0):
    self.wrapper_path = wrapper_path
    self.timeout_s = timeout_s

  def submit(self, argv: list[str], cwd: str = '') -> tuple[Optional[str], str]:
    # argv[0] is 'tpu' (a shell function); build a sourced-shell command.
    # `cwd` is where `tpu queue` runs, hence what its rsync packages -- it MUST
    # be the job's own checkout or the wrong source is shipped. Empty = inherit
    # the router's CWD (only safe when every difference rides on an explicit
    # flag). A non-existent cwd is refused up front rather than silently
    # packaging whatever the fallback directory happens to be.
    inner = ' '.join(_shquote(a) for a in argv)
    script = f'source {_shquote(self.wrapper_path)} >/dev/null 2>&1; {inner}'
    run_cwd = cwd or None
    if run_cwd is not None and not os.path.isdir(run_cwd):
      return None, f'[route_check] refusing to submit: workdir does not exist: {run_cwd}'
    try:
      proc = subprocess.run(['bash', '-c', script], capture_output=True,
                            text=True, timeout=self.timeout_s, cwd=run_cwd)
    except subprocess.TimeoutExpired as e:
      # ★A TIMEOUT IS NOT EVIDENCE THAT NOTHING WAS SUBMITTED. `tpu queue`
      # creates the experiment early and then blocks for minutes on the build,
      # so a timeout most often means "submitted, then we stopped watching".
      # Returning None here made the worker count a failed attempt and RESUBMIT:
      # the first XID then ran with no local row (invisible to every self-check
      # that walks the queue) while a second copy burned the same quota twice --
      # and the wasted spend pushed OTHER lines' jobs over the budget bar.
      # Same trap as cancellation: LOCAL FAILURE IS NOT REMOTE ABSENCE.
      return self._recover_timed_out_xid(e, argv)
    # ★Mark where stdout ends. `tpu queue` prints a ~400-char deprecation banner
    # to STDERR on every invocation, so a plain concatenation puts a fixed banner
    # AFTER the real error and any tail-excerpt returns only the banner. _tail()
    # splits on this marker and prefers stdout. Keep the marker in the string
    # (not a separate field) so the Submitter protocol and its fakes are unchanged.
    out = (proc.stdout or '') + _STDERR_MARK + (proc.stderr or '')
    return extract_xid(out), out

  def _recover_timed_out_xid(
      self, exc: subprocess.TimeoutExpired,
      argv: list[str]) -> tuple[Optional[str], str]:
    """After a submit timeout, find out whether the experiment EXISTS anyway.

    Two probes, cheapest first:
      1. the partial output captured before the timeout -- `tpu queue` prints
         `Experiment id: N` long before it returns, so this usually settles it
         at zero cost (subprocess.TimeoutExpired carries .stdout/.stderr);
      2. an XM lookup by `--experiment_name`, for the case where the timeout
         landed before the id was flushed.
    Only when BOTH come back empty do we report 'no XID' -- and then in words
    that do not claim the submit failed.
    """
    partial = _decode_stream(exc.stdout) + _STDERR_MARK + _decode_stream(exc.stderr)
    note = f'[route_check] tpu queue TIMED OUT after {self.timeout_s}s'

    xid = extract_xid(partial)
    if xid:
      return xid, (f'{note}, but the experiment WAS created: xid={xid} '
                   f'recovered from output captured before the timeout. '
                   f'Adopting it instead of resubmitting.\n{partial}')

    exp_name = _exp_name_of(argv)
    if exp_name:
      found, how = self.find_xid_by_name(exp_name)
      if found:
        return found, (f'{note}, but XManager HAS an experiment named '
                       f'{exp_name}: xid={found} ({how}). Adopting it instead '
                       f'of resubmitting.\n{partial}')
      note += f'; no XManager experiment named {exp_name} ({how})'
    else:
      note += '; no --exp_name to check XManager with, so remote state is UNKNOWN'

    return None, f'{note}. Treating as not-submitted.\n{partial}'

  def find_xid_by_name(self, exp_name: str,
                       timeout_s: float = 120.0) -> tuple[Optional[str], str]:
    """Newest XID whose experiment name matches EXACTLY, or (None, why).

    `--experiment_name` matches a SUBSTRING, so the exact-match filter below is
    load-bearing: `foo_v3` must not adopt `foo_v30`.
    """
    try:
      p = subprocess.run(
          [XMANAGER_PAR, 'list', f'--experiment_name={exp_name}',
           '--archived=no', '--columns=ID,Name,CreateTime'],
          capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
      return None, 'XM lookup itself timed out; remote state UNKNOWN'
    except OSError as e:
      return None, f'XM lookup could not run ({e}); remote state UNKNOWN'
    rows = []
    for ln in (p.stdout or '').splitlines():
      f = ln.split()
      if len(f) >= 2 and f[0].isdigit() and f[1] == exp_name:
        rows.append(f[0])
    if not rows:
      return None, 'XM lookup ran and found no exact-name match'
    return max(rows, key=int), f'XM lookup matched {len(rows)} experiment(s)'

  def cancel(self, xid: str) -> tuple[bool, str]:
    script = (f'source {_shquote(self.wrapper_path)} >/dev/null 2>&1; '
              f'tpu cancel {_shquote(xid)}')
    try:
      proc = subprocess.run(['bash', '-c', script], capture_output=True,
                            text=True, timeout=300.0)
    except subprocess.TimeoutExpired as e:
      return False, f'[route_check] tpu cancel TIMED OUT: {e}'
    return proc.returncode == 0, (proc.stdout or '') + (proc.stderr or '')


def _shquote(s: str) -> str:
  return "'" + str(s).replace("'", "'\\''") + "'"


# --- the status-probe seam (for re-route) ---------------------------------
def classify_wu_states(is_pending: bool, is_running: bool, is_terminal: bool,
                       is_completed: bool = False) -> str:
  """Collapse an XManager work unit's booleans into one STATUS_*. Pure.

  A job scheduled but not yet training (PREPARING/STARTING) counts as RUNNING
  here: it has left the auction, so re-routing it would throw away a placement
  that is about to succeed. Only PENDING -- still bidding -- is the trigger.

  `is_completed` splits the old single terminal bucket: a work unit that RAN TO
  COMPLETION is STATUS_COMPLETED (a success -- reconcile must write DONE), while
  failed/stopped/cancelled stays STATUS_TERMINAL (a real zombie -> FAILED).
  Completion is checked FIRST because XManager can report a finished work unit
  with both is_completed and is_stopped set; ending well outranks ending.
  """
  if is_completed:
    return STATUS_COMPLETED
  if is_terminal:
    return STATUS_TERMINAL
  if is_running:
    return STATUS_RUNNING
  if is_pending:
    return STATUS_PENDING
  # Not pending, not running, not terminal: it has left the auction and is
  # coming up (preparing/starting) -- treat as RUNNING, do not re-route.
  return STATUS_RUNNING


class XManagerStatusProbe:
  """Live status of an XID via XManager, collapsed to STATUS_*. A probe failure
  returns STATUS_UNKNOWN so the caller never cancels on missing data."""

  def __init__(self):
    self._client = None

  def _get_client(self):
    if self._client is None:
      from google3.learning.deepmind.xmanager2.client import xmanager_api
      self._client = xmanager_api.XManagerApi()
    return self._client

  def status(self, xid: str) -> str:
    try:
      client = self._get_client()
      experiment = client.get_experiment(int(xid))
      wus = list(experiment.get_work_units(
          populate_detailed_executable_status=True))
    except Exception:  # pylint: disable=broad-except
      return STATUS_UNKNOWN
    if not wus:
      # ★GONE, not UNKNOWN. `get_experiment` SUCCEEDED -- the id resolves -- and
      # the experiment reports no work units at all. A live job always has at
      # least one WU, so this is a definite verdict about the world, not a
      # failure to read it. Returning UNKNOWN here is what deadlocked 13 rows
      # for 5-8 days (measured 2026-08-31): reconcile's "never act on UNKNOWN"
      # guard left them SUBMITTED forever, while `tpu dequeue` refuses any row
      # that still carries an xid -- so nothing could ever clean them up.
      # The distinction that makes this safe is the try/except above: a probe
      # that cannot reach XM still returns UNKNOWN and still acts on nothing.
      return STATUS_GONE
    # A job is "still pending" only if EVERY work unit is pending; if any WU is
    # running/coming up, the placement took.
    states = []
    for wu in wus:
      # is_completed is kept SEPARATE from the failure booleans. OR-ing it in
      # here is what made every finished job read as a zombie: by the time the
      # decision layer saw 'TERMINAL' the success/failure bit no longer existed.
      is_completed = bool(getattr(wu, 'is_completed', False))
      is_terminal = bool(getattr(wu, 'is_failed', False)
                         or getattr(wu, 'is_stopped', False))
      states.append(classify_wu_states(
          bool(getattr(wu, 'is_pending', False)),
          bool(getattr(wu, 'is_running', False)),
          is_terminal,
          is_completed))
    if all(s == STATUS_PENDING for s in states):
      return STATUS_PENDING
    if any(s == STATUS_RUNNING for s in states):
      return STATUS_RUNNING
    # Order matters below: a job is only COMPLETED if EVERY work unit completed.
    # A mixed ending (some completed, some failed) is a FAILURE, not a success --
    # so TERMINAL is tested as 'any', matching the pre-existing conservative
    # bias that an ambiguous ending is never silently called a success.
    if all(s == STATUS_COMPLETED for s in states):
      return STATUS_COMPLETED
    if all(s in (STATUS_TERMINAL, STATUS_COMPLETED) for s in states):
      return STATUS_TERMINAL
    return STATUS_RUNNING


def _parse_fileutil_mtime(fields: list[str]) -> Optional[float]:
  """Epoch seconds from a `fileutil ls -l` row's date+time, or None.

  A row is like `-rw-rw---- 1 user group 15909 2026/08/24 01:49:01 <path>`.
  User/group spacing varies, so we scan for the `YYYY/MM/DD` token and take the
  `HH:MM:SS` right after it. Local time (fileutil prints local), so mktime.
  """
  for i, tok in enumerate(fields):
    if len(tok) == 10 and tok[4] == '/' and tok[7] == '/' and i + 1 < len(fields):
      stamp = f'{tok} {fields[i + 1]}'
      try:
        return time.mktime(time.strptime(stamp, '%Y/%m/%d %H:%M:%S'))
      except ValueError:
        return None
  return None


class _BorgVmProbe(Protocol):
  """Answers "is one of this job's Borg VM groups in RUN?".

  Tri-state ON PURPOSE. True: a group is RUN. False: Borg answered and NONE of
  the groups it printed is RUN. None: could not tell -- no cell pinned, the RPC
  failed, or the output was empty. Only False is a verdict about the world;
  None must never be actioned (AGENTS.md: an absence is the weakest reading).
  """

  def has_running_vmgroup(self,
                          entry: 'route_lib.QueueEntry') -> Optional[bool]:
    ...


class BorgVmProbe:
  """`borg findjobs` on the job's own cell, read at the VM-GROUP level.

  ★THE JOB STATE IS NOT THE VM-GROUP STATE, IN BOTH DIRECTIONS. jobs.md
  §`state: RUN` Is Not Evidence covers one of them: a Borg job reads RUN for
  hours with every group in ASSIGN/PENDING. This class exists for the other:
  XManager reads RUNNING for a job whose groups never left PENDING. Same fix
  either way -- look at the groups.

  An EMPTY result is None, never False. `findjobs` returns empty just as
  readily for a wrong cell or a malformed --name_re as for a job that truly
  does not exist, and an empty answer from a misaimed query is the classic
  false negative. False is returned only when Borg printed at least one
  VM-group state for this job and none of them was RUN.
  """

  _RUN_STATES = frozenset({'VMGROUP_STATE_RUN'})

  def __init__(self, timeout_s: float = 45.0):
    self._timeout_s = timeout_s

  def has_running_vmgroup(self,
                          entry: 'route_lib.QueueEntry') -> Optional[bool]:
    xid = getattr(entry, 'xid', None)
    cell = (getattr(entry, 'cell', None) or '').strip()
    if not xid or not cell:
      return None
    try:
      user = getpass.getuser()
    except Exception:  # pylint: disable=broad-except
      user = os.environ.get('USER', '')
    if not user:
      return None
    try:
      out = subprocess.run(
          ['borg', f'--borg={cell}', 'findjobs',
           f'--name_re={user}_group_{xid}\\..*'],
          capture_output=True, text=True, timeout=self._timeout_s)
    except (subprocess.TimeoutExpired, OSError):
      return None
    if out.returncode != 0:
      return None
    states = re.findall(r'VMGROUP_STATE_[A-Z_]+', out.stdout or '')
    if not states:
      return None  # nothing readable -- not evidence of absence
    return any(s in self._RUN_STATES for s in states)


# The launcher's own bucket rule, replayed (xm_launcher.py `_local_bucket` and
# `_BUCKET_SUFFIX`): an explicit --bucket wins, otherwise the bucket is resolved
# from the LANDING CELL through the measured locality table.
#
# ★A probe that reads only `launch_kwargs['bucket']` is blind to exactly the
# jobs the guides tell you to launch. Several metros and NO --bucket is the
# prescribed shape (jobs.md §Give The Router More Than One Way To Say Yes),
# because a hardcoded bucket under a multi-metro list writes cross-metro and
# gets the job pruned mid-run. Every such job returned None here, so
# `output_is_fresh(None, ...)` was False and the disk guard degraded to "no
# evidence" for all of them -- silently, and in the safe direction, which is why
# it survived: a guard that can only turn re-routing OFF looks harmless when it
# stops working. It is not harmless; it is the guard that keeps a live job from
# being cancelled.
_BUCKET_SUFFIX = 'home/qiaos/eqr_data'


def _bucket_for_entry(entry: 'route_lib.QueueEntry') -> Optional[str]:
  """Where this job WRITES, or None when that cannot be PROVEN.

  Never guesses a default prefix. An unknown cell returns None ("no disk
  evidence") rather than a plausible path, for the same reason the launcher
  refuses to launch on one: the wrong path is a perfectly valid path.
  """
  explicit = (getattr(entry, 'launch_kwargs', None) or {}).get('bucket')
  if explicit:
    return str(explicit)
  cell = (getattr(entry, 'cell', None) or '').strip()
  if not cell:
    return None
  try:
    return cell_locality.bucket_for(cell, _BUCKET_SUFFIX, allow_personal=False)
  except Exception:  # pylint: disable=broad-except
    # UnknownCellError, a metro with no group storage, or a locality table that
    # cannot answer. All mean the same thing here: we cannot prove where this
    # job writes, so we have no disk evidence about it.
    return None


class CnsOutputProbe:
  """Newest mtime of ANY file under a job's XID-prefixed output dir on CNS.

  A job that is placed and running writes rank/task logs (rank_0_attempt3.log,
  stdout, ...) LONG before its first final metric. So 'alive' must mean 'any
  output file written recently', NOT just the final arc_metrics -- a young eval
  arm that has placed, restarted an attempt, and is streaming rank logs but has
  not emitted metric #1 yet is very much alive (xid 282682357, 2026-08-24: rank
  logs writing every few seconds, zero final metrics, borg RUNNING).

  XID-prefixed dirs live at `<bucket>/logs/<project>/xid_<xid>_<ts>_<name>`, so
  we glob `<bucket>/logs/*/xid_<xid>_*` and take the newest mtime across the
  whole tree (`ls -l -R`). fileutil has no `--format`, so we parse its default
  `-l` output: columns `... <date> <time> <path>` (date/time are fields 6/7).

  Any failure (no bucket/XID, fileutil error/timeout, nothing matched, no
  parseable row) returns None = 'no disk evidence' -- the caller then falls
  back to the two-sample probe. We NEVER fabricate a recent time on error, so a
  broken lookup can only make reroute MORE careful, never keep a dead job alive.
  """

  def __init__(self, timeout_s: float = 20.0):
    self._timeout_s = timeout_s

  def latest_mtime(self, entry: 'route_lib.QueueEntry') -> Optional[float]:
    xid = entry.xid
    bucket = _bucket_for_entry(entry)
    if not xid or not bucket:
      return None
    # XID-prefixed dir: <bucket>/logs/<project>/xid_<xid>_*  (fileutil ** does
    # NOT recurse across levels, so name the levels explicitly). -R then walks
    # the whole subtree so we see the newest rank/task log, not just top-level.
    pattern = f'{bucket.rstrip("/")}/logs/*/xid_{xid}_*'
    try:
      out = subprocess.run(
          ['fileutil', 'ls', '-l', '-R', pattern],
          capture_output=True, text=True, timeout=self._timeout_s)
    except (subprocess.TimeoutExpired, OSError):
      return None
    if out.returncode != 0 or not out.stdout.strip():
      return None
    newest: Optional[float] = None
    for line in out.stdout.splitlines():
      # Default `-l` row: '<perms> <n> <user> <group> <size> <YYYY/MM/DD> '
      # '<HH:MM:SS> <path>'. Directories (perms start 'd') carry the dir's own
      # mtime too -- fine, a fresh file bumps its dir. Parse date+time fields.
      fields = line.split()
      if len(fields) < 8 or fields[0].startswith('total'):
        continue
      # find the 'YYYY/MM/DD' 'HH:MM:SS' pair (fields 5,6 in 0-index for files;
      # be robust to user/group spacing by scanning for the date-shaped token).
      mt = _parse_fileutil_mtime(fields)
      if mt is None:
        continue
      if newest is None or mt > newest:
        newest = mt
    return newest


# --- Pruned-restart evidence: the CNS I/O that feeds plan_pruned_restart -----
# route_lib owns the DECISION (pure, unit-tested); everything here is the I/O
# that gathers its inputs off CNS. Every failure path returns "no evidence"
# (None), which route_lib.plan_pruned_restart reads as a reason to HOLD -- so a
# broken read can only make auto-resume MORE cautious, never fire it wrongly.


def _new_resume_job_id(power: str) -> str:
  """A fresh local job_id for a warm-restart row, same shape as queue_cli's."""
  return f'{power}-{uuid.uuid4().hex[:6]}'


def _find_newest_rank0_log(bucket: str, xid: str,
                           timeout_s: float) -> Optional[str]:
  """Path of the newest rank-0 attempt log for an XID, or None.

  Logs live at `<bucket>/logs/<project>/xid_<xid>_*/logs/rank_0_attempt<N>.log`;
  the highest attempt number is the last one the run wrote (a preemption bumps
  the attempt), so its tail is where a crash -- if any -- would be.
  """
  pattern = f'{bucket.rstrip("/")}/logs/*/xid_{xid}_*/logs/rank_0_attempt*.log'
  try:
    out = subprocess.run(['fileutil', 'ls', pattern],
                         capture_output=True, text=True, timeout=timeout_s)
  except (subprocess.TimeoutExpired, OSError):
    return None
  if out.returncode != 0 or not out.stdout.strip():
    return None
  best_n, best = -1, None
  for line in out.stdout.splitlines():
    p = line.strip()
    m = re.search(r'rank_0_attempt(\d+)\.log$', p)
    if not m:
      continue
    n = int(m.group(1))
    if n > best_n:
      best_n, best = n, p
  return best


def _read_log_head_tail(path: str, head_bytes: int, tail_bytes: int,
                        timeout_s: float) -> tuple[str, str]:
  """(head, tail) byte-slices of a CNS log, capped so a multi-GB log is never
  pulled whole. The out_dir line lives in the head (boot banner); a crash lives
  in the tail. Any read failure yields ('', ''), i.e. no evidence.
  """
  q = shlex.quote(path)
  def _slice(cmd: str) -> str:
    try:
      out = subprocess.run(f'fileutil cat {q} | {cmd}', shell=True,
                           capture_output=True, text=True, timeout=timeout_s)
    except (subprocess.TimeoutExpired, OSError):
      return ''
    return out.stdout or ''
  return (_slice(f'head -c {int(head_bytes)}'),
          _slice(f'tail -c {int(tail_bytes)}'))


def _ls_cns(path: str, timeout_s: float) -> Optional[list[str]]:
  """`fileutil ls <path>` -> list of entry paths, or None on any failure/empty."""
  try:
    out = subprocess.run(['fileutil', 'ls', path],
                         capture_output=True, text=True, timeout=timeout_s)
  except (subprocess.TimeoutExpired, OSError):
    return None
  if out.returncode != 0 or not out.stdout.strip():
    return None
  return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def _latest_complete_checkpoint(out_dir: str,
                                timeout_s: float) -> Optional[str]:
  """Path of the highest-step COMPLETE checkpoint for `out_dir`, or None.

  Scans BOTH fleet layouts, because one out_dir is one or the other and the
  evidence layer cannot know which in advance:

    * torch port: `<out_dir>/steps/step_<N>.pt` FILES. The saver writes a
      dot-prefixed `.step_<N>.pt.tmp` and atomically `_replace`s it, so a job
      killed mid-write leaves only the tmp -- which fails the `step_` prefix /
      `.pt` suffix test and is never mistaken for a usable checkpoint.
    * ELT/EqR-jax: `<out_dir>/checkpoints/<N>/` DIRS with a BARE-INTEGER leaf.
      orbax writes `<N>.orbax-checkpoint-tmp-<uuid>` and atomically renames to
      `<N>` on finalize, so a bare int is complete by construction and a tmp
      leaf (non-digit) is skipped. Without this branch every ELT auto-resume
      HOLDs for lack of a checkpoint it never scanned for (the layout the
      pruned-restart path was blind to; the whole reason ELT could not
      auto-resume).

  The two live under different subdirs and never collide, so we take the global
  max step across both and return that leaf path verbatim (an ELT leaf comes
  back as `<out_dir>/checkpoints/<N>`, exactly what build_warm_restart_entry
  inverts into restart_from+restart_step).
  """
  base = out_dir.rstrip('/')
  best_step, best = -1, None

  # torch: <out_dir>/steps/step_<N>.pt files
  steps = base + '/steps'
  for p in (_ls_cns(steps, timeout_s) or []):
    name = p.rsplit('/', 1)[-1]
    if not (name.startswith('step_') and name.endswith('.pt')):
      continue
    s = route_lib.checkpoint_step(name)
    if s > best_step:
      best_step = s
      best = p if p.startswith('/cns') else f'{steps}/{name}'

  # ELT/EqR-jax: <out_dir>/checkpoints/<N> dirs (bare-int leaf = complete)
  ckpts = base + '/checkpoints'
  for p in (_ls_cns(ckpts, timeout_s) or []):
    name = p.rstrip('/').rsplit('/', 1)[-1]
    s = route_lib.elt_checkpoint_leaf_step(name)
    if s > best_step:
      best_step = s
      best = p.rstrip('/') if p.startswith('/cns') else f'{ckpts}/{name}'

  return best


class CnsRestartEvidence:
  """Gathers the two CNS facts plan_pruned_restart needs about a dead row: any
  code-bug signature in the log tail, and the newest complete checkpoint. Held
  as an injectable object so run_reconcile stays unit-testable with a fake.
  """

  def __init__(self, timeout_s: float = 30.0,
               head_bytes: int = 32768, tail_bytes: int = 65536):
    self._timeout_s = timeout_s
    self._head_bytes = head_bytes
    self._tail_bytes = tail_bytes

  def code_bug_and_checkpoint(
      self, entry: 'route_lib.QueueEntry'
  ) -> tuple[Optional[str], Optional[str]]:
    bucket = _bucket_for_entry(entry)
    if not bucket or not entry.xid:
      return None, None
    log_path = _find_newest_rank0_log(bucket, str(entry.xid), self._timeout_s)
    if not log_path:
      return None, None
    head, tail = _read_log_head_tail(
        log_path, self._head_bytes, self._tail_bytes, self._timeout_s)
    code_bug = route_lib.looks_like_code_bug(tail)
    out_dir = (route_lib.out_dir_from_log(head)
               or route_lib.out_dir_from_log(tail))
    checkpoint = (_latest_complete_checkpoint(out_dir, self._timeout_s)
                  if out_dir else None)
    return code_bug, checkpoint


def _restart_decision(dead: 'route_lib.QueueEntry',
                      entries: list['route_lib.QueueEntry'],
                      evidence, max_resumes: int):
  """(plan, checkpoint) for a just-failed row: gather CNS evidence, apply the
  pure decision. plan is (verdict, why). Kept separate so both the dry-run and
  the live branch of run_reconcile decide identically.
  """
  cb, ckpt = (evidence.code_bug_and_checkpoint(dead)
              if evidence else (None, None))
  other = route_lib.has_live_config_sibling(dead, entries)
  plan = route_lib.plan_pruned_restart(
      dead, xm_terminal=True, code_bug=cb, checkpoint=ckpt,
      other_live_writer=other, max_auto_resumes=max_resumes)
  return plan, ckpt


# --- Step2: XM-truth reconcile pass (fixes R3 zombie pollution) -------------
def run_reconcile(
    entries: list[route_lib.QueueEntry],
    now: float,
    probe: _StatusProbe,
    dry_run: bool = True,
    *,
    auto_resume_pruned: bool = False,
    auto_resume_max: int = 3,
    restart_evidence=None,
) -> tuple[list[route_lib.QueueEntry], list[str]]:
  """Re-verify every non-terminal (RECONCILABLE_STATES) entry against XManager
  and clean up stale local state. Returns (entries, log_lines).

  This is the reconcile half of the standalone tpu-reroute process. It fixes R3:
  .tpu_local_queue.json keeps state=RUNNING/SUBMITTED for jobs XManager no longer
  tracks (measured 2026-08-27: 13 local-RUNNING, 0 truly running on XM = ~21.7k
  cr/hr phantom occupancy on the ROUTE path). Any headroom/reroute logic reading
  local `state` is poisoned; this pass rewrites zombies to terminal so the route
  path is clean too (the billing path is already XM-truth via budget_check).

  The DECISION is pure (route_lib.decide_reconcile); here we only add the live
  probe and the log. Safety mirrors run_reroute: an UNKNOWN probe NEVER acts, so
  a probe hiccup can never mark a live job dead. Only a definite XM verdict moves
  an entry. In dry_run the decision is logged but the entry is not mutated.
  """
  log: list[str] = []
  targets = [e for e in entries if e.state in route_lib.RECONCILABLE_STATES]
  if not targets:
    log.append('[reconcile] no non-terminal entries to reconcile.')
    return entries, log
  n_zombie = n_promoted = n_unknown = n_noop = n_done = n_resumed = 0
  for e in targets:
    xid = e.xid
    if not xid:
      # A BUILDING/SUBMITTED row with no xid has no XM identity to check.
      continue
    st = probe.status(str(xid))
    tag = f'{e.job_id} (xid={xid}, local {e.state.value}, XM {st})'
    # Age gates the GONE verdict only (a young row is legitimately 0-WU).
    # submitted_at is None on a re-routed row, which reads as 'age unknown'
    # and, like UNKNOWN, actions nothing.
    age_s = (now - e.submitted_at) if e.submitted_at else None
    new_state = route_lib.decide_reconcile(e.state, st, age_s=age_s)
    if new_state is None:
      if st == STATUS_UNKNOWN:
        n_unknown += 1
      else:
        n_noop += 1
      continue
    if dry_run:
      log.append(f'[DRY][reconcile] would set {tag} -> {new_state.value}')
      if new_state == route_lib.JobState.FAILED:
        n_zombie += 1
        if auto_resume_pruned:
          (verdict, why), ckpt = _restart_decision(
              e, entries, restart_evidence, auto_resume_max)
          if verdict == route_lib.RESUME_WARM:
            log.append(f'[DRY][auto-resume] {tag}: would warm-restart from '
                       f'{ckpt} ({why})')
          else:
            log.append(f'[DRY][auto-resume] {tag}: HOLD ({why})')
      elif new_state == route_lib.JobState.DONE:
        n_done += 1
      else:
        n_promoted += 1
      continue
    changed = route_lib.reconcile_entry(e, st, age_s=age_s)
    if changed:
      if e.state == route_lib.JobState.FAILED:
        n_zombie += 1
        if st == STATUS_GONE:
          log.append(f'[reconcile] {tag} -> FAILED '
                     f'(experiment GONE: 0 work units, age {int(age_s or 0)}s)')
        else:
          log.append(f'[reconcile] {tag} -> FAILED (zombie cleaned up)')
        if auto_resume_pruned:
          (verdict, why), ckpt = _restart_decision(
              e, entries, restart_evidence, auto_resume_max)
          if verdict == route_lib.RESUME_WARM and ckpt:
            new = route_lib.build_warm_restart_entry(
                e, ckpt, _new_resume_job_id(e.power))
            entries.append(new)
            n_resumed += 1
            log.append(f'[auto-resume] {tag}: warm-restart queued as '
                       f'{new.job_id} from {ckpt}')
          else:
            log.append(f'[auto-resume] {tag}: HOLD ({why})')
      elif e.state == route_lib.JobState.DONE:
        n_done += 1
        log.append(f'[reconcile] {tag} -> DONE (completed normally)')
      else:
        n_promoted += 1
        log.append(f'[reconcile] {tag} -> RUNNING (placement confirmed)')
  log.append(
      f'[reconcile] {len(targets)} checked: {n_zombie} zombie->FAILED, '
      f'{n_done} completed->DONE, '
      f'{n_promoted} promoted->RUNNING, {n_unknown} UNKNOWN (left alone), '
      f'{n_noop} already-correct'
      + (f', {n_resumed} auto warm-restart(s) queued.' if auto_resume_pruned
         else '.'))
  return entries, log


def adopt_escaped_builds(
    entries: list[route_lib.QueueEntry],
    submitter: '_Submitter',
    dry_run: bool = True,
) -> tuple[list[route_lib.QueueEntry], list[str]]:
  """Resolve rows whose BUILDING claim went stale: adopt the experiment the
  build may have left behind, or clear the flag so the row can build again.

  WHY THIS EXISTS. A stale BUILDING claim has two readings, and the expensive
  one is not the obvious one. Obvious: the worker crashed mid-build, so requeue.
  Expensive: the build SUCCEEDED and the experiment is RUNNING, and only the
  write-back of its xid was lost -- then requeuing puts a SECOND writer on the
  first one's output path. Observed 2026-09-02: elt-dit-50k-fid-v3b was
  reclaimed to QUEUED while xid 285706173 ran; it was adopted by hand.

  reclaim_stale_building cannot make this call itself -- it runs inside the
  queue flock, where a network RPC would block every reader -- so it parks the
  row with `adopt_check_name` set and route_lib's claim selectors refuse to
  build it. This pass, outside the lock, is what unparks it.

  ★Only an EXACT name match adopts, and only a lookup that actually RAN
  clears the flag. A lookup that timed out leaves the row parked: "I could not
  see it" must not read as "it is not there", or the double-write we are
  preventing comes back through the failure path.
  """
  log: list[str] = []
  parked = [e for e in entries if e.adopt_check_name]
  if not parked:
    return entries, log
  n_adopted = n_cleared = n_still_unknown = 0
  for e in parked:
    name = e.adopt_check_name
    if not name:          # narrowed for the type checker; the filter guarantees it
      continue
    found, how = submitter.find_xid_by_name(name)
    if found:
      if dry_run:
        log.append(f'[DRY][adopt] would adopt xid={found} for {e.job_id} '
                   f'(exp_name={name}; {how})')
        n_adopted += 1
        continue
      e.xid = found
      e.state = route_lib.JobState.SUBMITTED   # reconcile promotes it if RUNNING
      e.submitted_at = e.submitted_at or time.time()
      e.adopt_check_name = None
      e.last_reason = (f'adopted escaped build: the stale BUILDING claim had '
                       f'already produced xid={found} ({how}); re-dispatching '
                       f'would have double-written its output path')
      log.append(f'[adopt] {e.job_id} -> xid={found} ({how})')
      n_adopted += 1
      continue
    # No match. Distinguish "the lookup ran and saw nothing" (safe to release)
    # from "the lookup could not run" (must stay parked).
    ran = how.startswith('XM lookup ran')
    if not ran:
      n_still_unknown += 1
      log.append(f'[adopt] {e.job_id} STAYS PARKED: {how}')
      continue
    if dry_run:
      log.append(f'[DRY][adopt] would release {e.job_id} to build ({how})')
      n_cleared += 1
      continue
    e.adopt_check_name = None
    e.last_reason = (f'adopt-check clear: no XManager experiment named {name} '
                     f'({how}), so the stale build left nothing behind; '
                     f'releasing the row to build again')
    log.append(f'[adopt] {e.job_id} released to build ({how})')
    n_cleared += 1
  log.append(f'[adopt] {len(parked)} parked row(s): {n_adopted} adopted, '
             f'{n_cleared} released, {n_still_unknown} still unresolved.')
  return entries, log


# --- the re-route sweep ---------------------------------------------------
def run_reroute(
    entries: list[route_lib.QueueEntry],
    now: float,
    probe: _StatusProbe,
    submitter: Optional[_Submitter] = None,
    reroute_after_s: float = 600.0,
    cooldown_s: float = 1800.0,
    dry_run: bool = True,
    output_probe: Optional[_OutputProbe] = None,
    confirm_gap_s: float = 15.0,
    fresh_output_s: float = 1200.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    history_file: Optional[str] = None,
    borg_probe: Optional['_BorgVmProbe'] = None,
    nominal_running_grace_s: float = 3600.0,
) -> tuple[list[route_lib.QueueEntry], list[str]]:
  """Cancel SUBMITTED jobs stuck PENDING past the deadline and return them to
  QUEUED for the next tick to re-place. Returns (entries, log_lines).

  The clock rule lives in route_lib.needs_reroute; here we add the live check
  and the side effects (cancel + mark_reroute, which cools the stuck cell). A
  job that has meanwhile started RUNNING is promoted; a terminal one is left for
  infra_check to reconcile.

  HARDENING (2026-08-24, after xid 282605596 was wrongly re-routed): a single
  PENDING snapshot is NOT enough to cancel -- a BATCH job's EMA shadow work
  units run in segments, so XManager can read `all pending` in the gap between
  two segments while the job is in fact training. Before cancelling a job the
  first probe called PENDING, we now require BOTH:
    (a) no fresh output on disk -- output_probe.latest_mtime within
        fresh_output_s means it is writing NOW, so it is alive; AND
    (b) a SECOND probe, taken confirm_gap_s later, also PENDING -- the shadow
        gap clears on the second sample.
  Only if both guards still say 'stuck' do we cancel (route_lib.decide_reroute
  owns this pure decision). Every guard can only turn a would-be reroute OFF;
  the reroute path is never widened. The RUNNING/TERMINAL/UNKNOWN branches --
  including the zombie TERMINAL->FAILED cleanup -- are unchanged.
  """
  log: list[str] = []

  candidates = [e for e in entries
                if route_lib.needs_reroute(e, now, reroute_after_s)]
  # ★Rows already promoted to RUNNING are re-checked too, on their own longer
  # clock. needs_reroute() selects SUBMITTED only, so without this the RUNNING
  # branch below is reachable only on the single tick that promotes a row --
  # and a row promoted on a stale XManager RUNNING would never be looked at
  # again. Selection is not a verdict: the branch confirms against Borg and the
  # disk before it touches anything, and promotes on any doubt.
  recheck = [e for e in entries
             if route_lib.needs_liveness_recheck(e, now,
                                                 nominal_running_grace_s)]
  if not candidates and not recheck:
    log.append('[reroute] no SUBMITTED job past the pending deadline, and no '
               'RUNNING row due a liveness re-check.')
    return entries, log

  # ★GLOBAL BRAKE. Counts what THIS process re-routed in the last hour, keyed on
  # nothing, so it survives a churning job being dequeued and re-enqueued under
  # a new id (measured: 7 plates across 2 rows, per-row counters saw 2).
  # Tripping means the fault is the re-router's, and cancelling more cars cannot
  # fix that -- so it does NOTHING and says so loudly.
  hist = _load_reroute_history(history_file or REROUTE_HISTORY_FILE)
  if route_lib.global_reroute_brake(hist, now):
    n = len([t for t in hist if t >= now - 3600.0])
    log.append(
        f'[reroute] ★GLOBAL BRAKE ENGAGED: {n} re-routes in the last hour '
        f'(limit {route_lib.REROUTE_GLOBAL_MAX_PER_HOUR}). Re-routing is '
        f'SUSPENDED this pass and {len(candidates) + len(recheck)} '
        f'candidate(s) left alone. '
        f'A churn rate this high is a fault in placement or in the liveness '
        f'probe, and cancelling more cars cannot fix either.')
    return entries, log

  sub = submitter or Submitter()
  for e in candidates + [r for r in recheck if r not in candidates]:
    # `or now` would read a submitted_at of 0.0 as "no timestamp" and report
    # age 0 -- harmless while every real stamp is an epoch second, and wrong the
    # moment anything (a test, a reset, a hand-edited row) carries a literal 0.
    # The new liveness verdict below gates on this number, so say what is meant.
    age = int(now - (e.submitted_at if e.submitted_at is not None else now))
    xid = e.xid
    state = probe.status(xid) if xid else STATUS_UNKNOWN
    tag = f'{e.job_id} (xid={xid}, {e.cell}, pending {age}s)'
    if state == STATUS_PENDING and xid:
      # GUARD (a): disk evidence of life. A fresh write beats a PENDING snapshot.
      mtime = output_probe.latest_mtime(e) if output_probe else None
      out_fresh = route_lib.output_is_fresh(mtime, now, fresh_output_s)
      if out_fresh:
        age_out = int(now - mtime) if mtime else -1
        e.last_reason = f'alive: output written {age_out}s ago (not re-routed)'
        log.append(f'[reroute] {tag} has FRESH output ({age_out}s ago) '
                   f'-> alive, no action')
        continue
      # GUARD (b): second sample after a gap -- the shadow gap clears on it.
      # (dry-run also takes it, so the log shows the true would-be decision.)
      sleep_fn(confirm_gap_s)
      # A cheap disk re-check inside the window costs nothing and catches a
      # write that landed during the gap (v26: time-staggered second sample).
      mtime2 = output_probe.latest_mtime(e) if output_probe else None
      if route_lib.output_is_fresh(mtime2, time.time(), fresh_output_s):
        age_out = int(time.time() - mtime2) if mtime2 else -1
        e.last_reason = f'alive: output written {age_out}s ago (2nd check)'
        log.append(f'[reroute] {tag} output FRESH on 2nd check '
                   f'-> alive, no action')
        continue
      state2 = probe.status(xid)
      if not route_lib.decide_reroute(state, state2, out_fresh):
        log.append(f'[reroute] {tag} 2nd probe={state2} (not PENDING) '
                   f'-> alive/ambiguous, no action')
        continue
      # Double-confirmed stuck: both probes PENDING and no fresh output.
      if dry_run:
        log.append(f'[DRY][reroute] would cancel + re-route {tag}: '
                   f'PENDING x2, no fresh output')
        continue
      ok, out = sub.cancel(xid)
      if ok:
        hist.append(time.time())
        _save_reroute_history(hist, history_file or REROUTE_HISTORY_FILE)
        route_lib.mark_reroute(e, now, cooldown_s)
        log.append(f'[reroute] cancelled + re-queued {tag}; cell cooled {int(cooldown_s)}s')
      else:
        log.append(f'[reroute] cancel FAILED for {tag}, left SUBMITTED. {_tail(out)}')
    elif state == STATUS_RUNNING:
      # ★XM RUNNING IS NOT "HAS CHIPS", AND PROMOTING ON IT IS A ONE-WAY DOOR.
      # needs_reroute() selects SUBMITTED rows only, so the moment RUNNING is
      # written here the row leaves the re-router's jurisdiction permanently.
      # A job whose Borg VM groups never leave PENDING reads RUNNING at the XM
      # layer indefinitely: measured on xid 286573746, which sat 12 h in `sj`
      # with every VM group PENDING, zero bytes in CNS and zero charged
      # resources, while XManager, `tpu queue-status` and the queue file all
      # said RUNNING -- and no mechanism existed that would ever move it. Its
      # six-metro fallback list was useless because nothing consulted it.
      # A second instance the same night (codi_unroll_loopdist_mm, xid
      # 286454115) sat 32 h the same way, so this is the shape, not a one-off.
      #
      # So promote only when something INDEPENDENT of XManager agrees the job
      # is on hardware. Any one of these promotes:
      #   * Borg says a VM group is RUN -- the direct reading;
      #   * the output dir has ever been written -- it got far enough to write;
      #   * the Borg probe could not tell (None) -- an unreadable probe is not
      #     a verdict, so FAIL OPEN to the old behaviour.
      # Only the full conjunction withholds promotion: Borg ANSWERED, no group
      # is RUN, nothing was ever written, and the grace period has passed. Then
      # it is treated exactly like a stuck PENDING job -- same cancel, same
      # mark_reroute, same global brake -- because that is what it is.
      vm_running = borg_probe.has_running_vmgroup(e) if borg_probe else None
      nominal = False
      if vm_running is False and age >= nominal_running_grace_s and xid:
        # `latest_mtime` conflates "dir missing" with "lookup failed", so this
        # asks the weaker question it can actually answer: has ANYTHING ever
        # been written? A job holding chips writes rank logs within minutes.
        mtime_ever = output_probe.latest_mtime(e) if output_probe else None
        nominal = mtime_ever is None
      # `or not xid` is redundant against the guard above (nominal can only be
      # True when xid is truthy) and is written anyway: it is what narrows xid
      # from `str | None` to `str` for the cancel call below, and a reader
      # should not have to prove that invariant from two places at once.
      if not nominal or not xid:
        e.state = route_lib.JobState.RUNNING
        e.last_reason = f'running in {e.cell} ({e.arch}-{e.chips})'
        why = ('borg vmgroup RUN' if vm_running
               else 'borg unreadable' if vm_running is None
               else 'output written')
        log.append(f'[reroute] {tag} is RUNNING now -> promoted ({why}), '
                   f'no action')
        continue
      if dry_run:
        log.append(f'[DRY][reroute] would cancel + re-route {tag}: XM says '
                   f'RUNNING but NO Borg VM group is RUN and nothing was ever '
                   f'written ({age}s > {int(nominal_running_grace_s)}s grace)')
        continue
      ok, out = sub.cancel(xid)
      if ok:
        hist.append(time.time())
        _save_reroute_history(hist, history_file or REROUTE_HISTORY_FILE)
        route_lib.mark_reroute(e, now, cooldown_s)
        log.append(f'[reroute] cancelled + re-queued {tag}: nominally RUNNING '
                   f'(XM RUNNING, no Borg VM group in RUN, no output in '
                   f'{age}s); cell cooled {int(cooldown_s)}s')
      else:
        log.append(f'[reroute] cancel FAILED for nominally-RUNNING {tag}, '
                   f'left as-is. {_tail(out)}')
    elif state == STATUS_TERMINAL:
      e.state = route_lib.JobState.FAILED
      e.last_reason = 'terminal per XManager (failed/stopped)'
      log.append(f'[reroute] {tag} is TERMINAL -> marked FAILED')
    else:  # STATUS_UNKNOWN
      log.append(f'[reroute] {tag} status UNKNOWN -> no action (never cancel blind)')
  return entries, log


# --- the tick -------------------------------------------------------------
def run_tick(
    entries: list[route_lib.QueueEntry],
    provider: _Provider,
    now: float,
    submitter: Optional[_Submitter] = None,
    dry_run: bool = True,
    group: str = DEFAULT_GROUP,
    max_placements: Optional[int] = None,
    verbose: bool = True,
) -> tuple[list[route_lib.QueueEntry], list[str]]:
  """One router tick. Returns (updated_entries, log_lines).

  Pure orchestration over route_lib + the provider + the submitter seam; the
  entries list is mutated in place (apply_placement) and also returned. In
  dry-run nothing is submitted -- the plan is logged and the queue is unchanged.
  """
  log: list[str] = []
  queued = [e for e in entries if e.state == route_lib.JobState.QUEUED]
  if not queued:
    log.append('[route_check] no QUEUED jobs; nothing to do.')
    return entries, log

  avail_by_cell, arch_price, arch_pool = provider.fetch()
  log.append(f'[route_check] {len(queued)} queued; availability: '
             f'{len(avail_by_cell)} (cell,arch) entries, '
             f'pools={ {a: int(p) for a, p in arch_pool.items()} }')

  placements = route_lib.select_and_plan(
      entries, avail_by_cell, now, max_placements=max_placements,
      arch_price=arch_price, arch_pool=arch_pool)

  if not placements:
    log.append('[route_check] nothing placeable this tick (all candidate '
               'cells oversold/full/cooled-down); jobs stay QUEUED.')
    if verbose:
      for e in queued:
        log.append(f'    - {e.job_id} ({e.power}, archs={e.allowed_archs}) '
                   f'WAITING: {e.last_reason or "no placeable cell"}')
    return entries, log

  by_id = {e.job_id: e for e in entries}
  for p in placements:
    entry = by_id.get(p.job_id)
    if entry is None:
      continue
    # ★Package the ENQUEUE-TIME SNAPSHOT when the row has one, else the live
    # workdir (route_lib.package_dir). The snapshot is a frozen local copy taken
    # at `tpu enqueue`, so a build minutes-to-hours later ships the code as it
    # was enqueued, not whatever the checkout drifted to. A row from before this
    # field, or one enqueued --no_snapshot, has no snapshot_dir and falls back to
    # workdir -- the pre-feature behavior, unchanged.
    workdir = route_lib.package_dir(entry)
    cwd_note = f'  (cwd={workdir})' if workdir else '  (cwd=router process dir -- config must be via --flag)'
    if dry_run:
      argv = build_tpu_queue_cmd(p, entry, group)
      log.append(f'[DRY] would place {p.job_id}: {p.reason}')
      log.append(f'      cmd: {" ".join(argv)}{cwd_note}')
      continue
    # live submit
    sub = submitter or Submitter()
    argv = build_tpu_queue_cmd(p, entry, group)
    log.append(f'[route_check] placing {p.job_id}: {p.reason}')
    log.append(f'      cmd: {" ".join(argv)}{cwd_note}')
    t_submit = time.time()
    xid, out = sub.submit(argv, cwd=workdir)
    if xid:
      # ★`submitted_at` is "epoch when handed to XM", and `submit` BLOCKS for the
      # whole build -- 890-1692 s measured in the field (host build lock, then
      # the build). Passing the round's OPENING `now` backdated it by exactly
      # that much, and reconcile measures its GONE grace period from it, so the
      # grace was already spent before the experiment existed: five live cars
      # were reconciled to FAILED at true ages of 128-938 s while still RUNNING
      # on XM, and a FAILED row does not cancel its xid, so each kept billing
      # invisibly. ADD THE MEASURED BLOCKING TIME to the injected `now` rather
      # than reading the clock afresh, so a caller-supplied `now` (tests,
      # replay) still drives the result instead of being silently ignored.
      submitted_now = now + max(0.0, time.time() - t_submit)
      route_lib.apply_placement(entry, p, xid=xid, now=submitted_now)
      log.append(f'      -> SUBMITTED xid={xid} cell={p.cell}')
    else:
      entry.attempts += 1
      entry.last_reason = 'submit produced no XID (see launch log)'
      log.append(f'      -> FAILED to submit (no XID). tail: '
                 f'{_tail(out)}')
  return entries, log


def _tail(s: str, n: int = 240) -> str:
  """Excerpt a command's output for a one-line `last_reason`.

  ★A pure tail is not a neutral excerpt -- it silently prefers whatever the
  command prints LAST, so a tool that ends every invocation with a fixed banner
  evicts the actual error DETERMINISTICALLY, not just unluckily. Measured
  2026-08-30 (with elt-reproduction-v3, who proved the mechanism): `tpu queue`
  writes a 399-char deprecation banner to stderr on every call, `Submitter.submit`
  concatenated stderr last, and the window was 240 -- so 399 > 240 means a
  tail-only excerpt could return *nothing but* the banner. Every build failure
  fleet-wide recorded that banner instead of its cause from 2026-08-29 onward.

  Three defences, because each alone is fragile:

  1. **A `[[MARKER]]` verdict always wins.** `tpu_wrapper.sh` deliberately prints
     `[[STAGE_SRC_REFUSED]]` / `[[STAGE_RSYNC_TIMEOUT]]` / `[[STAGE_RM_REFUSED]]`
     as the LAST stderr line precisely so the old tail-240 would keep them, and
     `budget_check.py` does the same with `[[BUDGET_DEFERRED]]`. Preferring stdout
     (defence 2) would have thrown those away whenever stdout was non-empty --
     replacing one silent-loss bug with another. So markers are hoisted first,
     whichever stream they came from. ★When you retire a convention, carry the
     things that were built to depend on it.
  2. Otherwise prefer the STDOUT half, where the real error is.
  3. Within the chosen text keep the HEAD as well as the tail: an error appears
     at the start, the exit summary at the end, and the middle is progress
     chatter. State how much was dropped so a reader can tell an excerpt from a
     whole message.
  """
  s = s or ''
  clean_all = _ANSI_RE.sub('', s)
  markers = [ln.strip() for ln in clean_all.splitlines()
             if ln.strip().startswith('[[') and ']]' in ln]
  if _STDERR_MARK in s:
    stdout_part, stderr_part = s.split(_STDERR_MARK, 1)
    # Prefer stdout; fall back to stderr only when stdout carried nothing.
    s = stdout_part if stdout_part.strip() else stderr_part
  s = s.strip().replace('\n', ' | ')
  prefix = ''
  if markers:
    # Hoist the verdict(s) to the front and give them the budget they need.
    prefix = ' | '.join(markers)
    if prefix not in s:
      s = f'{prefix} | {s}' if s else prefix
    elif not s.startswith(prefix):
      s = f'{prefix} | {s}'
  if len(s) <= n:
    return s
  # Never let the excerpt window truncate a hoisted verdict.
  keep = max(n, len(prefix) + 40) if prefix else n
  half = max(1, (keep - 20) // 2)
  if prefix and half < len(prefix):
    head = s[:len(prefix) + 1]
    tail_budget = max(1, keep - len(head) - 20)
    return f'{head} …[{len(s) - len(head) - tail_budget} chars cut]… {s[-tail_budget:]}'
  return f'{s[:half]} …[{len(s) - 2 * half} chars cut]… {s[-half:]}'


# --- the serial build-worker ----------------------------------------------
class _StageHealthProbe(Protocol):
  """Returns the cumulative count of srcfs/CreateSnapshot write failures right
  now. The worker brakes if this jumps between polls (the CitC token bucket is
  draining -- concurrent stage-writes elsewhere). Backed by a log/RPC probe in
  production, scripted in tests."""

  def failure_count(self) -> int:
    ...


def plan_one_entry(
    entry: route_lib.QueueEntry,
    provider: _Provider,
    now: float,
) -> Optional[route_lib.Placement]:
  """Pick a placement for ONE already-claimed entry, using live availability.
  Same policy as the tick (effective-price type, most-placeable cell), but for a
  single entry -- the worker has already chosen WHICH job via the queue order."""
  try:
    avail_by_cell, arch_price, arch_pool = provider.fetch()
  except Exception as e:  # pylint: disable=broad-except
    entry.last_reason = f'availability fetch failed: {e}'
    return None
  return route_lib.plan_one(entry, avail_by_cell, now,
                            arch_price=arch_price, arch_pool=arch_pool)


def run_worker_once(
    queue_file: str,
    provider: _Provider,
    submitter: _Submitter,
    now: float,
    worker_id: str,
    build_stale_s: float = 1800.0,
    group: str = DEFAULT_GROUP,
    stage_probe: Optional[_StageHealthProbe] = None,
    srcfs_fail_brake: int = 20,
    last_fail_count: Optional[int] = None,
    max_build_attempts: int = 3,
    claim_pick: 'Optional[Callable[[list[route_lib.QueueEntry]], Optional[route_lib.QueueEntry]]]' = None,
) -> tuple[str, list[str], Optional[int]]:
  """One worker step. Returns (outcome, log_lines, new_fail_count).

  outcome is one of: 'submitted', 'requeued', 'idle' (nothing to build),
  'busy' (a build already in flight), 'braked' (srcfs failures spiking),
  'held' (the claimed job cannot build as-is and was parked, not churned).

  The single-build invariant is enforced by claim_next_build (atomic, flock'd):
  at most one entry is BUILDING across all worker processes. This step claims
  one, runs `tpu queue` for it (the ONE build), and records the result.
  """
  log: list[str] = []
  new_fail_count = last_fail_count

  # MODE-2 BRAKE: if srcfs write failures jumped since last poll, the CitC token
  # bucket is draining -- do not add a stage-write. Skip this round.
  if stage_probe is not None and srcfs_fail_brake > 0:
    try:
      cur = stage_probe.failure_count()
      new_fail_count = cur
      if last_fail_count is not None and (cur - last_fail_count) >= srcfs_fail_brake:
        log.append(f'[worker] BRAKE: {cur - last_fail_count} new srcfs failures '
                   f'since last poll (>= {srcfs_fail_brake}); skipping this round '
                   'to let the CitC token bucket recover.')
        return 'braked', log, new_fail_count
    except Exception as e:  # pylint: disable=broad-except
      log.append(f'[worker] stage-health probe failed ({e}); proceeding without brake.')

  # ATOMIC CLAIM: reclaim stale, then take the next QUEUED as BUILDING iff no
  # live build is in flight. This is the serial lock.
  claimed = claim_next_build(queue_file, now, worker_id, build_stale_s,
                             pick=claim_pick)
  if claimed is None:
    # Distinguish 'a build is in flight' from 'nothing to do' for the log.
    entries = load_queue(queue_file)
    if route_lib.count_building(entries) > 0:
      return 'busy', log, new_fail_count
    return 'idle', log, new_fail_count

  log.append(f'[worker] claimed {claimed.job_id} (BUILDING); planning + building.')

  # WORKDIR GUARD: a set-but-nonexistent package dir would package the wrong
  # source (or fail). Do NOT churn on it -- park it in HELD for a human to
  # re-enqueue. Check the ACTUAL dir that will be packaged: the enqueue-time
  # snapshot if the row has one, else the live workdir (route_lib.package_dir).
  # This also catches a snapshot reclaimed out from under a row. An EMPTY dir is
  # allowed (a flag-only run); an unbuildable one is caught by max-attempts HOLD.
  pkg_dir = route_lib.package_dir(claimed)
  if pkg_dir and not os.path.isdir(pkg_dir):
    reason = f'package dir does not exist: {pkg_dir} -- re-enqueue from a valid checkout'
    def _hold_bad_workdir(e: route_lib.QueueEntry) -> None:
      route_lib.hold_entry(e, reason)
    update_entry(queue_file, claimed.job_id, _hold_bad_workdir)
    log.append(f'[worker] {claimed.job_id} -> HELD ({reason}); slot released, not churned.')
    return 'held', log, new_fail_count

  # PLAN a cell for it (live availability).
  placement = plan_one_entry(claimed, provider, now)
  if placement is None:
    # Nothing placeable right now -> release the slot, back to QUEUED.
    reason = claimed.last_reason or 'nothing placeable right now'
    def _requeue(e: route_lib.QueueEntry) -> None:
      e.state = route_lib.JobState.QUEUED
      e.build_started_at = None
      e.worker_id = None
      e.last_reason = f'waiting: {reason}'
    update_entry(queue_file, claimed.job_id, _requeue)
    log.append(f'[worker] {claimed.job_id}: {reason}; released slot, back to QUEUED.')
    return 'requeued', log, new_fail_count

  # BUILD + SUBMIT: the one build. build_tpu_queue_cmd + submit(cwd=workdir).
  # ★Prefer the group the ROUTER admitted this job under (g5/g3 before g9).
  # Falling back to the worker's global --group here is what made the whole
  # preference a no-op before: dispatch chose g5, the builder submitted g9.
  # ★A caller PIN outranks both: dispatch already resolves to it, but a row
  # claimed without a dispatch round (the legacy `--worker` claim_pick, or a
  # row hand-moved to BUILD_REQUESTED) would otherwise lose the pin here --
  # exactly the "two leaders" shape that made the g5 preference a no-op.
  argv = build_tpu_queue_cmd(
      placement, claimed,
      route_lib.pinned_group(claimed) or getattr(claimed, 'group', None)
      or group)
  log.append(f'[worker] building {claimed.job_id}: {placement.reason} '
             f'(cwd={pkg_dir or "router dir"})')
  t_submit = time.time()
  xid, out = submitter.submit(argv, cwd=pkg_dir)

  if xid:
    # ★See the same fix in run_tick: `submit` blocks for the whole build, so the
    # round's opening `now` backdates submitted_at by the build duration and
    # burns reconcile's GONE grace before the experiment exists. This is the
    # path that submitted all five cars measured as wrongly reconciled.
    submitted_now = now + max(0.0, time.time() - t_submit)

    def _submitted(e: route_lib.QueueEntry) -> None:
      route_lib.apply_placement(e, placement, xid=xid, now=submitted_now)
      e.build_started_at = None
      e.worker_id = None
    update_entry(queue_file, claimed.job_id, _submitted)
    log.append(f'[worker] {claimed.job_id} -> SUBMITTED xid={xid} cell={placement.cell}')
    return 'submitted', log, new_fail_count

  # R2 FIX: BUDGET-DEFERRAL is NOT a build failure. Before the MODE-1 GUARD
  # counts this no-XID as an attempt, check for the budget marker: budget_check
  # printed `[[BUDGET_DEFERRED]]` because the submit was over the G9 bar. That is
  # a transient FLEET state, not a defect of this job. Park it BUDGET_DEFERRED
  # (soft, auto-promoted next round) WITHOUT touching attempts -- so a job that
  # merely hit a few over-budget rounds is never parked in HELD. Slot released.
  if is_budget_deferral(out):
    def _deferred(e: route_lib.QueueEntry) -> None:
      route_lib.mark_budget_deferred(
          e, f'budget-deferred at build: over the G9 bar; will retry when '
          f'headroom opens (attempts untouched at {e.attempts}). {_tail(out)}')
    update_entry(queue_file, claimed.job_id, _deferred)
    log.append(f'[worker] {claimed.job_id} -> BUDGET_DEFERRED (over bar, NOT a '
               f'build failure; attempts unchanged); slot released.')
    return 'budget_deferred', log, new_fail_count

  # MODE-1 GUARD: no XID / found[] zombie -> NOT submitted. Count the attempt.
  # Requeue for a retry UNLESS it has now failed max_build_attempts times, in
  # which case park it in HELD so one bad job cannot churn the worker forever
  # and starve the rest of the queue (an unattended worker must self-limit).
  attempts_after = claimed.attempts + 1
  if attempts_after >= max_build_attempts:
    reason = (f'build produced no XID after {attempts_after} attempts '
              f'(found[]/crash?); parked. Last: {_tail(out)}')
    def _held(e: route_lib.QueueEntry) -> None:
      e.attempts = attempts_after
      route_lib.hold_entry(e, reason)
    update_entry(queue_file, claimed.job_id, _held)
    log.append(f'[worker] {claimed.job_id} -> HELD after {attempts_after} failed '
               f'attempts; not churning. tail: {_tail(out)}')
    return 'held', log, new_fail_count

  def _failed(e: route_lib.QueueEntry) -> None:
    e.state = route_lib.JobState.QUEUED
    e.build_started_at = None
    e.worker_id = None
    e.attempts = attempts_after
    e.last_reason = (f'build produced no XID (found[]/crash?); retry '
                     f'{attempts_after}/{max_build_attempts}. {_tail(out)}')
  update_entry(queue_file, claimed.job_id, _failed)
  log.append(f'[worker] {claimed.job_id} -> NO XID (found[]/build crash); requeued '
             f'({attempts_after}/{max_build_attempts}). tail: {_tail(out)}')
  return 'requeued', log, new_fail_count


# --- Step3: the budget seam (reuse wiki_agent budget_check --query) ----------
DEFAULT_BUDGET_SCRIPT = os.path.expanduser('~/work/wiki_agents/tools/budget_check.py')


def budget_query(tpu_type: str, tier: str = 'PROD', lo_price: str = '',
                 group: str = '', script: str = DEFAULT_BUDGET_SCRIPT,
                 timeout_s: float = 60.0) -> Optional[dict]:
  """Call budget_check.py --query and return its JSON dict, or None on failure.

  The dict is {income,bar,current,headroom,new_cost,exempt,fits}; `current` is
  XM-truth (budget_check cross-refs the check-cache and age-filters zombies), so
  the router NEVER recomputes cost -- it consumes this. Owner split: billing is
  wiki_agent's; the router only asks. Returns None if the script is missing or
  the call fails/timeouts, so the caller can fail SAFE (treat as no-headroom /
  skip dispatch) rather than guess a number.
  """
  if not os.path.isfile(script):
    return None
  argv = ['python3', script, '--query', tpu_type, tier or 'PROD',
          lo_price or '0', group or '']
  try:
    out = subprocess.run(argv, capture_output=True, text=True,
                         timeout=timeout_s)
    line = _ANSI_RE.sub('', (out.stdout or '').strip()).splitlines()
    if not line:
      return None
    return json.loads(line[-1])   # --query prints ONE json line last
  except Exception:  # pylint: disable=broad-except
    return None


# --- Step3: the router half -- greedy dispatch under backpressure ------------
def run_dispatch_once(
    queue_file: str,
    now: float,
    group: str = DEFAULT_GROUP,
    budget_query_fn: 'Callable[..., Optional[dict]]' = budget_query,
    dry_run: bool = True,
    group_order: Optional[list[str]] = None,
) -> tuple[str, list[str]]:
  """ONE router-dispatch round (the DISPATCH half of the rewritten worker).

  Sequence (design 3.1):
    1. promote BUDGET_DEFERRED -> QUEUED   (top-of-round: everyone re-tests)
    2. BACKPRESSURE: if any BUILD_REQUESTED/BUILDING remains, the builder has
       not drained the prior round -> do NOT dispatch (headroom would be stale).
    3. query XM-truth headroom, then greedy plan_dispatch with in-memory
       pre-debit over the QUEUED entries.
    4. apply: fits -> BUILD_REQUESTED ; over-bar -> BUDGET_DEFERRED.
  Returns (outcome, log). outcome in {'promoted-only','backpressure','dispatched',
  'idle','no-budget'}. All writes go through with_queue_lock. Pure decision is in
  route_lib.plan_dispatch; here we add the queue I/O and the budget seam.
  """
  log: list[str] = []
  # Default preference: the operator's own exempt pools first, g9 floor last.
  group_order = [g for g in (group_order or DEFAULT_GROUP_ORDER) if g]
  if group_order[-1] != group:
    # Always keep the caller's --group as the final fallback, so behaviour with
    # a custom --group is unchanged when none of the preferred pools fit.
    group_order = group_order + [group]

  # 1. promote deferred (under lock) so they re-test this round.
  with with_queue_lock(queue_file):
    entries = load_queue(queue_file)
    promoted = route_lib.promote_deferred(entries)
    if promoted:
      save_queue(queue_file, entries)
  if promoted:
    log.append(f'[dispatch] promoted {len(promoted)} BUDGET_DEFERRED -> QUEUED '
               f'for re-test this round.')

  # 2. BACKPRESSURE: builder still draining -> skip dispatch.
  entries = load_queue(queue_file)
  pending = route_lib.count_build_pending(entries)
  if pending > 0:
    log.append(f'[dispatch] backpressure: {pending} BUILD_REQUESTED/BUILDING '
               f'still draining; no new dispatch this round.')
    return 'backpressure', log

  queued = [e for e in entries if e.state == route_lib.JobState.QUEUED]
  if not queued:
    log.append('[dispatch] no QUEUED jobs to dispatch.')
    return 'idle', log

  # 3. headroom (XM-truth). One query for the round's starting headroom; the
  #    per-candidate cost also comes from the seam, and plan_dispatch pre-debits
  #    in memory so we do not double-count within the round.
  # Headroom is a property of the G9 floor group (the pool with the income/10
  # bar); the exempt pools do not consume it. Probe the LAST group in the order.
  probe0 = budget_query_fn('v6e-16', 'PROD', '', group_order[-1])
  if probe0 is None:
    log.append('[dispatch] budget query unavailable -> fail SAFE: no dispatch '
               'this round (never guess headroom).')
    return 'no-budget', log
  headroom = float(probe0.get('headroom', 0.0))
  log.append(f'[dispatch] headroom={headroom:.1f} cr/hr (bar={probe0.get("bar")} '
             f'current={probe0.get("current")}); {len(queued)} queued candidates.')

  # per-entry cost + exemption via the same seam (cached per type within round).
  # ★GROUP PREFERENCE (operator 2026-08-31 00:05Z / 00:25Z, restated as a
  # standing order): try g5, then g3, then g9 -- IN THAT ORDER, per job. g5/g3
  # are the operator's own dynamic pools: they are EXEMPT from the G9
  # income/10 cap (budget_check._EXEMPT_GROUP_IDS) and self-limit at 100% of
  # their own income, so an overspend there parks itself and nobody has to
  # watch it. G9 is the one with the hard 1/10 ceiling that a human is held to
  # ("G9 超过 10% 我老板会骂我"), so it is the LAST resort, never the default.
  # Before this, group_order existed only on the one-shot place path; the
  # dispatch worker never read it, so all 82 rows went to g9 while g5/g3 sat at
  # 0.0 chips with ~122k credits idle.
  _cost_cache: dict = {}
  def _type_of(e: route_lib.QueueEntry) -> str:
    if e.arch and e.chips:
      return f'{e.arch}-{e.chips}'
    return e.power
  def _probe_group(e: route_lib.QueueEntry, g: str) -> dict:
    key = (_type_of(e), e.tier or 'PROD', g)
    if key not in _cost_cache:
      _cost_cache[key] = budget_query_fn(key[0], key[1], '', g) or {}
    return _cost_cache[key]
  def _pick_group(e: route_lib.QueueEntry) -> tuple:
    """First group in `group_order` whose budget gate admits this job.

    Returns (group, probe). Falls back to the LAST group in the order (the g9
    floor) when none fits, so the job is budget-deferred against g9 exactly as
    before -- the preference can only move a job to a cheaper pool, never make
    a previously-placeable job unplaceable.

    ★A CALLER PIN OVERRIDES THE ORDER ENTIRELY (and is still budget-probed, so
    a pinned-to-g9 job that exceeds the income/10 bar is deferred exactly like
    any other g9 job -- a pin buys a POOL, never a bypass of the bar). The
    preference order is a fleet-wide default; it is right for a job that has no
    opinion and wrong for one that keeps being preempted out of g5's
    vqfree-xm. Without this branch there was no way to express the exception:
    every route into the row (`--group=`, `--launch=group=`) was either eaten
    by absl or dropped at build time."""
    pin = route_lib.pinned_group(e)
    if pin:
      return pin, (_probe_group(e, pin) or {})
    last = (group_order[-1], {})
    for g in group_order:
      pr = _probe_group(e, g)
      if not pr:
        continue          # probe failed for this group: try the next one
      last = (g, pr)
      if pr.get('fits'):
        return g, pr
    return last
  def _probe(e: route_lib.QueueEntry) -> dict:
    return _pick_group(e)[1]
  def cost_of(e: route_lib.QueueEntry) -> float:
    return float(_probe(e).get('new_cost', 0.0))
  def is_exempt(e: route_lib.QueueEntry) -> bool:
    return bool(_probe(e).get('exempt', False))

  # Remember which pool won for each candidate, so the BUILDER submits under it.
  chosen: dict = {e.job_id: _pick_group(e)[0] for e in queued}
  for e in queued:
    g = chosen.get(e.job_id)
    pin = route_lib.pinned_group(e)
    if pin:
      log.append(f'[dispatch] {e.job_id} -> group g{g} (PINNED by caller; '
                 f'preference order {",".join(group_order)} not consulted)')
    elif g and g != group_order[-1]:
      log.append(f'[dispatch] {e.job_id} -> group g{g} (exempt from the g9 '
                 f'income/10 bar)')

  plan = route_lib.plan_dispatch(queued, headroom, cost_of, is_exempt)

  # 4. apply the plan under lock.
  decided = {d.job_id: d for d in plan}
  n_req = n_def = 0
  with with_queue_lock(queue_file):
    live = load_queue(queue_file)
    for e in live:
      d = decided.get(e.job_id)
      if d is None or e.state != route_lib.JobState.QUEUED:
        continue   # only touch entries still QUEUED (a concurrent claim may have moved one)
      if dry_run:
        continue
      if d.decision == route_lib.JobState.BUILD_REQUESTED:
        # ★Carry the admitted group onto the row: dispatch and build are
        # separate processes, so without this the builder falls back to its
        # global --group (9) and the preference silently does nothing.
        e.group = chosen.get(e.job_id) or e.group
        route_lib.mark_build_requested(e, d.reason); n_req += 1
      else:
        route_lib.mark_budget_deferred(e, d.reason); n_def += 1
    if not dry_run and (n_req or n_def):
      save_queue(queue_file, live)
  if dry_run:
    for d in plan:
      log.append(f'[DRY][dispatch] {d.job_id} -> {d.decision.value} '
                 f'[group g{chosen.get(d.job_id, group_order[-1])}] ({d.reason})')
    return 'dispatched', log
  log.append(f'[dispatch] dispatched {n_req} -> BUILD_REQUESTED, '
             f'{n_def} -> BUDGET_DEFERRED.')
  return 'dispatched', log


def run_worker_loop(
    queue_file: str,
    provider_factory: 'Callable[[], _Provider]',
    submitter: _Submitter,
    worker_id: str,
    poll_s: float = 15.0,
    build_stale_s: float = 1800.0,
    group: str = DEFAULT_GROUP,
    stage_probe: Optional[_StageHealthProbe] = None,
    srcfs_fail_brake: int = 20,
    max_build_attempts: int = 3,
    max_iterations: Optional[int] = None,
) -> None:
  """The worker loop: run_worker_once forever, sleeping poll_s when idle/busy/
  braked. `provider_factory` builds a fresh provider per build (one RPC each).
  `max_iterations` bounds the loop for tests."""
  last_fail = None
  it = 0
  while max_iterations is None or it < max_iterations:
    it += 1
    provider = provider_factory()
    outcome, log, last_fail = run_worker_once(
        queue_file, provider, submitter, now=time.time(), worker_id=worker_id,
        build_stale_s=build_stale_s, group=group, stage_probe=stage_probe,
        srcfs_fail_brake=srcfs_fail_brake, last_fail_count=last_fail,
        max_build_attempts=max_build_attempts)
    for line in log:
      print(line, flush=True)
    # After a successful build, immediately try the next (drain fast); otherwise
    # sleep so an empty/busy/braked queue does not spin.
    if outcome not in ('submitted',):
      time.sleep(poll_s)


def run_dispatch_worker_loop(
    queue_file: str,
    provider_factory: 'Callable[[], _Provider]',
    submitter: _Submitter,
    worker_id: str,
    group_order: Optional[list[str]] = None,
    poll_s: float = 15.0,
    build_stale_s: float = 1800.0,
    group: str = DEFAULT_GROUP,
    stage_probe: Optional[_StageHealthProbe] = None,
    srcfs_fail_brake: int = 20,
    max_build_attempts: int = 3,
    budget_query_fn: 'Callable[..., Optional[dict]]' = budget_query,
    max_iterations: Optional[int] = None,
) -> None:
  """Step3 combined loop (the rewritten worker): DISPATCH then BUILD, forever.

  Each round:
    1. run_dispatch_once: promote deferred -> backpressure gate -> greedy
       plan_dispatch with XM-truth headroom -> mark BUILD_REQUESTED/BUDGET_DEFERRED.
    2. drain THIS round's BUILD_REQUESTED serially: claim one (via
       next_build_requested, NOT next_queued), run `tpu queue`, record. The
       single-build invariant (one BUILDING at a time) is unchanged. A no-XID
       with the budget marker parks BUDGET_DEFERRED (R2), a real failure counts
       an attempt as before.
  This is ONE process = router+builder (design 3.1). R1 dies because this is the
  ONLY place that calls `tpu queue`; the daemon's in-lane place pass is gated off
  at go-live (TPU_ROUTE_INLANE_PLACE=0). `max_iterations` bounds it for tests.
  """
  last_fail = None
  it = 0
  while max_iterations is None or it < max_iterations:
    it += 1
    # 1. DISPATCH round.
    _, dlog = run_dispatch_once(
        queue_file, now=time.time(), group=group,
        budget_query_fn=budget_query_fn, dry_run=False,
        group_order=group_order)
    for line in dlog:
      print(line, flush=True)
    # 2. BUILD one BUILD_REQUESTED (serial). Claim from BUILD_REQUESTED so we
    #    never build a job the router has not budget-admitted this round.
    provider = provider_factory()
    outcome, wlog, last_fail = run_worker_once(
        queue_file, provider, submitter, now=time.time(), worker_id=worker_id,
        build_stale_s=build_stale_s, group=group, stage_probe=stage_probe,
        srcfs_fail_brake=srcfs_fail_brake, last_fail_count=last_fail,
        max_build_attempts=max_build_attempts,
        claim_pick=route_lib.next_build_requested)
    for line in wlog:
      print(line, flush=True)
    # Drain fast after a successful build; otherwise sleep so an idle/busy/braked
    # queue does not spin.
    if outcome not in ('submitted',):
      time.sleep(poll_s)


def main(argv):
  del argv

  if _WORKER.value:
    # SERIAL BUILD-WORKER loop. One build at a time, forever. This is the cure
    # for concurrent-build failures (found[] zombies + CitC token exhaustion).
    import socket
    worker_id = f'{socket.gethostname()}:{os.getpid()}'
    print(f'[worker] serial build-worker {worker_id} on {_QUEUE_FILE.value}; '
          f'one build at a time, poll {_WORKER_POLL_S.value}s.', flush=True)
    run_worker_loop(
        _QUEUE_FILE.value,
        provider_factory=lambda: avail_provider.AvailabilityProvider(group=_GROUP.value),
        submitter=Submitter(),
        worker_id=worker_id,
        poll_s=_WORKER_POLL_S.value,
        build_stale_s=_BUILD_STALE_S.value,
        group=_GROUP.value,
        srcfs_fail_brake=_SRCFS_FAIL_BRAKE.value,
        max_build_attempts=_MAX_BUILD_ATTEMPTS.value,
    )
    return

  if _DISPATCH_WORKER.value:
    # Step3 REWRITTEN worker: router-dispatch + serial builder in one loop.
    import socket
    worker_id = f'{socket.gethostname()}:{os.getpid()}'
    print(f'[dispatch-worker] router+builder loop {worker_id} on '
          f'{_QUEUE_FILE.value}; greedy dispatch under XM-truth headroom, serial '
          f'build, poll {_WORKER_POLL_S.value}s.', flush=True)
    run_dispatch_worker_loop(
        _QUEUE_FILE.value,
        provider_factory=lambda: avail_provider.AvailabilityProvider(group=_GROUP.value),
        submitter=Submitter(),
        worker_id=worker_id,
        group_order=([g.strip() for g in _GROUP_ORDER.value.split(',') if g.strip()]
                     if _GROUP_ORDER.value else None),
        poll_s=_WORKER_POLL_S.value,
        build_stale_s=_BUILD_STALE_S.value,
        group=_GROUP.value,
        srcfs_fail_brake=_SRCFS_FAIL_BRAKE.value,
        max_build_attempts=_MAX_BUILD_ATTEMPTS.value,
    )
    return

  if _REROUTE_LOOP.value:
    # STANDALONE tpu-reroute PROCESS (Step2). Its OWN loop, independent of the
    # builder, so the reroute safety net (pending>deadline cancel) always runs
    # even while the serial builder is busy. Each round: (A) XM-truth reconcile
    # to clean zombies off the route path, then (B) the reroute sweep. Both go
    # through the same load/merge-write path as the daemon's one-shot passes, so
    # concurrency is unchanged (with_queue_lock cross-process flock).
    print(f'[reroute-loop] standalone reconcile+reroute on {_QUEUE_FILE.value}; '
          f'poll {_REROUTE_LOOP_POLL_S.value}s.', flush=True)
    while True:
      # (A0) adopt-check pass. Rows whose BUILDING claim went stale are parked
      # (route_lib.reclaim_stale_building sets adopt_check_name, and the claim
      # selectors refuse to build them) until we know whether that build had
      # already escaped to XManager. This runs FIRST so a row carrying a live
      # xid is reconciled in the same round rather than sitting parked for one.
      try:
        snap = load_queue(_QUEUE_FILE.value)
        ad_entries, ad_log = adopt_escaped_builds(
            snap, submitter=Submitter(), dry_run=_DRY_RUN.value)
        for line in ad_log:
          print(f'  [reroute-loop:adopt] {line}', flush=True)
        if ad_log and not _DRY_RUN.value:
          merge_and_save_touched(_QUEUE_FILE.value, ad_entries)
      except Exception as e:  # pylint: disable=broad-except
        print(f'  [reroute-loop:adopt] pass FAILED (non-fatal): {e}',
              flush=True)
      # (A) reconcile pass
      try:
        snap = load_queue(_QUEUE_FILE.value)
        rc_entries, rc_log = run_reconcile(
            snap, now=time.time(), probe=XManagerStatusProbe(),
            dry_run=_DRY_RUN.value,
            auto_resume_pruned=_AUTO_RESUME_PRUNED.value,
            auto_resume_max=_AUTO_RESUME_MAX.value,
            restart_evidence=(CnsRestartEvidence()
                              if _AUTO_RESUME_PRUNED.value else None))
        for line in rc_log:
          print(f'  [reroute-loop:reconcile] {line}', flush=True)
        if not _DRY_RUN.value:
          merge_and_save_touched(_QUEUE_FILE.value, rc_entries)
      except Exception as e:  # pylint: disable=broad-except
        print(f'  [reroute-loop:reconcile] pass FAILED (non-fatal): {e}',
              flush=True)
      # (B) reroute pass
      try:
        snap = load_queue(_QUEUE_FILE.value)
        rr_entries, rr_log = run_reroute(
            snap, now=time.time(), probe=XManagerStatusProbe(),
            reroute_after_s=_REROUTE_AFTER_S.value, cooldown_s=_COOLDOWN_S.value,
            dry_run=_DRY_RUN.value, output_probe=CnsOutputProbe(),
            confirm_gap_s=_CONFIRM_GAP_S.value,
            fresh_output_s=_FRESH_OUTPUT_S.value,
            borg_probe=BorgVmProbe(),
            nominal_running_grace_s=_NOMINAL_RUNNING_GRACE_S.value)
        for line in rr_log:
          print(f'  [reroute-loop:reroute] {line}', flush=True)
        if not _DRY_RUN.value:
          merge_and_save_touched(_QUEUE_FILE.value, rr_entries)
      except Exception as e:  # pylint: disable=broad-except
        print(f'  [reroute-loop:reroute] pass FAILED (non-fatal): {e}',
              flush=True)
      time.sleep(_REROUTE_LOOP_POLL_S.value)
    return  # unreachable; defensive

  # Read an unlocked SNAPSHOT for the pass. The slow work below (status/avail
  # RPCs, cancel/submit, the confirm-gap sleep) runs on this snapshot WITHOUT
  # holding the queue lock, so a concurrent `tpu enqueue` is never blocked for
  # the minutes a tick can take. The lock is taken only for the fast merge-write
  # at the end (merge_and_save_touched), which re-reads the live queue and folds
  # our changes back in by job_id -- so rows enqueued during the RPCs survive.
  entries = load_queue(_QUEUE_FILE.value)

  if _RECONCILE.value:
    # One-shot XM-truth reconcile pass (Step2). Clean zombies off the route path.
    updated, log = run_reconcile(
        entries, now=time.time(), probe=XManagerStatusProbe(),
        dry_run=_DRY_RUN.value,
        auto_resume_pruned=_AUTO_RESUME_PRUNED.value,
        auto_resume_max=_AUTO_RESUME_MAX.value,
        restart_evidence=(CnsRestartEvidence()
                          if _AUTO_RESUME_PRUNED.value else None))
  elif _REROUTE.value:
    # Sweep SUBMITTED jobs stuck PENDING; cancel + return to QUEUED.
    updated, log = run_reroute(
        entries, now=time.time(), probe=XManagerStatusProbe(),
        reroute_after_s=_REROUTE_AFTER_S.value, cooldown_s=_COOLDOWN_S.value,
        dry_run=_DRY_RUN.value, output_probe=CnsOutputProbe(),
        confirm_gap_s=_CONFIRM_GAP_S.value,
        fresh_output_s=_FRESH_OUTPUT_S.value,
        borg_probe=BorgVmProbe(),
        nominal_running_grace_s=_NOMINAL_RUNNING_GRACE_S.value)
  else:
    # Drain QUEUED jobs into the XM queue, trying groups IN PREFERENCE ORDER.
    # `--group_order=5,9` places what the free vqfree pool (g5) can take first
    # and only lets the remainder fall through to the g9 floor -- so we lean on
    # borrowed/free capacity before spending the paid floor. Each group is a
    # normal run_tick against ITS OWN live availability; a job placed under an
    # earlier group is no longer QUEUED, so the next group's tick only sees what
    # is left. With no --group_order this is exactly the old single-group tick.
    group_order = (
        [g.strip() for g in _GROUP_ORDER.value.split(',') if g.strip()]
        if _GROUP_ORDER.value else [_GROUP.value])
    log = []
    updated = entries
    for gi, grp in enumerate(group_order):
      remaining = [e for e in updated if e.state == route_lib.JobState.QUEUED]
      if not remaining and gi > 0:
        log.append(f'[route_check] all jobs placed before group {grp}; '
                   f'skipping remaining groups.')
        break
      if len(group_order) > 1:
        log.append(f'[route_check] === placement pass under group {grp} '
                   f'({gi + 1}/{len(group_order)}) ===')
      # Scope this pass's availability fetch to the archs actually queued (one
      # RPC per arch). Without this, adding the 8 GPU archs to ARCH_PLATFORM
      # would make every tick fire 13 serial RPCs even for an all-TPU queue.
      # GPU archs are in ARCH_PLATFORM (GAP#1a), so a queued GPU job's arch is
      # kept here and its availability IS fetched.
      queued_archs = sorted({
          a.lower() for e in remaining for a in (e.allowed_archs or [])
          if a.lower() in avail_provider.ARCH_PLATFORM
      })
      provider = avail_provider.AvailabilityProvider(
          group=grp, archs=queued_archs or None)
      updated, tick_log = run_tick(
          updated, provider, now=time.time(),
          dry_run=_DRY_RUN.value, group=grp,
          max_placements=_MAX_PLACEMENTS.value, verbose=_VERBOSE.value)
      log.extend(tick_log)

  for line in log:
    print(line)
  if not _DRY_RUN.value:
    # Merge-write instead of a whole-queue overwrite: fold only the rows this
    # pass touched back into the live queue, preserving anything enqueued while
    # the RPCs ran. Neither run_tick nor run_reroute removes entries (they only
    # mutate state in place), so there are no dropped_job_ids to pass.
    merge_and_save_touched(_QUEUE_FILE.value, updated)
    print(f'[route_check] queue saved to {_QUEUE_FILE.value}')
  else:
    print('[route_check] DRY RUN -- queue not modified. Pass --nodry_run to act.')


if __name__ == '__main__':
  app.run(main)
