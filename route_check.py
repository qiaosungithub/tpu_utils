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

import dataclasses
import fcntl
import json
import os
import re
import subprocess
import time
from typing import Callable, Optional, Protocol

from absl import app
from absl import flags

from google3.experimental.users.qiaos.tpu_utils import avail_provider
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


# Live scheduling states for a submitted XID, collapsed to what re-route needs.
STATUS_PENDING = 'PENDING'     # still in the auction -- the re-route trigger
STATUS_RUNNING = 'RUNNING'     # scheduled/coming up/running -- leave it alone
STATUS_TERMINAL = 'TERMINAL'   # failed/completed/cancelled -- stop tracking
STATUS_UNKNOWN = 'UNKNOWN'     # probe failed -- do NOT act (never cancel blind)


class _StatusProbe(Protocol):
  """Returns the collapsed live state of one XID (one of STATUS_*). Backed by
  XManager in production, scripted in tests."""

  def status(self, xid: str) -> str:
    ...


DEFAULT_QUEUE_FILE = os.path.expanduser('~/.tpu_local_queue.json')
# The wrapper defining the `tpu` shell function; we source it, then call `tpu`.
TPU_WRAPPER = os.path.expanduser('~/work/tpu_cmd/tpu_wrapper.sh')
DEFAULT_GROUP = '9'

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
    'cooldown_s', 1800.0, 'After a re-route, avoid the stuck cell for this long '
    'so the job does not bounce straight back into it.')
_WORKER = flags.DEFINE_bool(
    'worker', False, 'Run as the SERIAL build-worker loop: claim one QUEUED job '
    'at a time as BUILDING, run `tpu queue` for it, record the result, repeat. '
    'Only ever one build in flight -- the cure for concurrent-build failures.')
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


def claim_next_build(path: str, now: float, worker_id: str,
                     stale_after_s: float) -> Optional[route_lib.QueueEntry]:
  """Atomically: reclaim stale BUILDING, then IF no live build is in flight,
  mark the next QUEUED entry BUILDING and persist. Returns the claimed entry
  (a copy reflecting the persisted state) or None if nothing was claimed
  (queue empty, or a build already in flight). Serialized by an exclusive
  flock held across the whole read-modify-write."""
  lock_path = _lockfile(path)
  with open(lock_path, 'w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
      entries = load_queue(path)
      route_lib.reclaim_stale_building(entries, now, stale_after_s)
      if not route_lib.can_claim_build(entries, now, stale_after_s):
        save_queue(path, entries)   # persist any reclaim even if we don't claim
        return None
      nxt = route_lib.next_queued(entries)
      if nxt is None:
        save_queue(path, entries)
        return None
      route_lib.claim_for_build(nxt, now, worker_id)
      save_queue(path, entries)
      return nxt
    finally:
      fcntl.flock(lock, fcntl.LOCK_UN)


def update_entry(path: str, job_id: str,
                 mutate: 'Callable[[route_lib.QueueEntry], None]') -> bool:
  """Atomically apply `mutate` to the entry with `job_id` and persist. Returns
  True if the entry was found. Used to write the post-build result (SUBMITTED,
  or back to QUEUED) without clobbering concurrent edits to other entries."""
  lock_path = _lockfile(path)
  with open(lock_path, 'w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
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
    finally:
      fcntl.flock(lock, fcntl.LOCK_UN)


# --- pure helpers ---------------------------------------------------------
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
    flag = k if k.startswith('--') else f'--{k}'
    if v is None or v is True:
      argv.append(flag)
    elif v is False:
      continue
    else:
      argv.append(f'{flag}={v}')
  return argv


def extract_xid(output: str) -> Optional[str]:
  """The XID from `tpu queue` output, ANSI-stripped, or None. Same rule as the
  wrapper: accept both the create line and the resume 'work unit(s)' line."""
  m = _XID_RE.search(_ANSI_RE.sub('', output or ''))
  return m.group(1) if m else None


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
      return None, f'[route_check] tpu queue TIMED OUT after {self.timeout_s}s: {e}'
    out = (proc.stdout or '') + (proc.stderr or '')
    return extract_xid(out), out

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
def classify_wu_states(is_pending: bool, is_running: bool, is_terminal: bool
                       ) -> str:
  """Collapse an XManager work unit's booleans into one STATUS_*. Pure.

  A job scheduled but not yet training (PREPARING/STARTING) counts as RUNNING
  here: it has left the auction, so re-routing it would throw away a placement
  that is about to succeed. Only PENDING -- still bidding -- is the trigger.
  """
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
      return STATUS_UNKNOWN
    # A job is "still pending" only if EVERY work unit is pending; if any WU is
    # running/coming up, the placement took.
    states = []
    for wu in wus:
      is_terminal = bool(getattr(wu, 'is_failed', False)
                         or getattr(wu, 'is_completed', False)
                         or getattr(wu, 'is_stopped', False))
      states.append(classify_wu_states(
          bool(getattr(wu, 'is_pending', False)),
          bool(getattr(wu, 'is_running', False)),
          is_terminal))
    if all(s == STATUS_PENDING for s in states):
      return STATUS_PENDING
    if any(s == STATUS_RUNNING for s in states):
      return STATUS_RUNNING
    if all(s == STATUS_TERMINAL for s in states):
      return STATUS_TERMINAL
    return STATUS_RUNNING


# --- the re-route sweep ---------------------------------------------------
def run_reroute(
    entries: list[route_lib.QueueEntry],
    now: float,
    probe: _StatusProbe,
    submitter: Optional[_Submitter] = None,
    reroute_after_s: float = 600.0,
    cooldown_s: float = 1800.0,
    dry_run: bool = True,
) -> tuple[list[route_lib.QueueEntry], list[str]]:
  """Cancel SUBMITTED jobs stuck PENDING past the deadline and return them to
  QUEUED for the next tick to re-place. Returns (entries, log_lines).

  The clock rule lives in route_lib.needs_reroute; here we add the live check
  (only cancel a job the probe CONFIRMS is still pending -- never on UNKNOWN)
  and the side effects (cancel + mark_reroute, which cools the stuck cell). A
  job that has meanwhile started RUNNING is promoted; a terminal one is left for
  infra_check to reconcile.
  """
  log: list[str] = []
  candidates = [e for e in entries
                if route_lib.needs_reroute(e, now, reroute_after_s)]
  if not candidates:
    log.append('[reroute] no SUBMITTED job past the pending deadline.')
    return entries, log

  sub = submitter or Submitter()
  for e in candidates:
    age = int(now - (e.submitted_at or now))
    xid = e.xid
    state = probe.status(xid) if xid else STATUS_UNKNOWN
    tag = f'{e.job_id} (xid={xid}, {e.cell}, pending {age}s)'
    if state == STATUS_PENDING and xid:
      if dry_run:
        log.append(f'[DRY][reroute] would cancel + re-route {tag}: still PENDING')
        continue
      ok, out = sub.cancel(xid)
      if ok:
        route_lib.mark_reroute(e, now, cooldown_s)
        log.append(f'[reroute] cancelled + re-queued {tag}; cell cooled {int(cooldown_s)}s')
      else:
        log.append(f'[reroute] cancel FAILED for {tag}, left SUBMITTED. {_tail(out)}')
    elif state == STATUS_RUNNING:
      e.state = route_lib.JobState.RUNNING
      e.last_reason = f'running in {e.cell} ({e.arch}-{e.chips})'
      log.append(f'[reroute] {tag} is RUNNING now -> promoted, no action')
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
    workdir = entry.workdir or ''
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
    xid, out = sub.submit(argv, cwd=workdir)
    if xid:
      route_lib.apply_placement(entry, p, xid=xid, now=now)
      log.append(f'      -> SUBMITTED xid={xid} cell={p.cell}')
    else:
      entry.attempts += 1
      entry.last_reason = 'submit produced no XID (see launch log)'
      log.append(f'      -> FAILED to submit (no XID). tail: '
                 f'{_tail(out)}')
  return entries, log


def _tail(s: str, n: int = 240) -> str:
  s = (s or '').strip().replace('\n', ' | ')
  return s[-n:]


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
  claimed = claim_next_build(queue_file, now, worker_id, build_stale_s)
  if claimed is None:
    # Distinguish 'a build is in flight' from 'nothing to do' for the log.
    entries = load_queue(queue_file)
    if route_lib.count_building(entries) > 0:
      return 'busy', log, new_fail_count
    return 'idle', log, new_fail_count

  log.append(f'[worker] claimed {claimed.job_id} (BUILDING); planning + building.')

  # WORKDIR GUARD: a set-but-nonexistent workdir would package the wrong source
  # (or fail). Do NOT churn on it -- park it in HELD for a human to re-enqueue.
  # An EMPTY workdir is allowed here (it may be a flag-only run); if it turns out
  # to be unbuildable it is caught by the max-attempts HOLD below, not guessed at.
  if claimed.workdir and not os.path.isdir(claimed.workdir):
    reason = f'workdir does not exist: {claimed.workdir} -- re-enqueue from a valid checkout'
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
  argv = build_tpu_queue_cmd(placement, claimed, group)
  log.append(f'[worker] building {claimed.job_id}: {placement.reason} '
             f'(cwd={claimed.workdir or "router dir"})')
  xid, out = submitter.submit(argv, cwd=claimed.workdir or '')

  if xid:
    def _submitted(e: route_lib.QueueEntry) -> None:
      route_lib.apply_placement(e, placement, xid=xid, now=now)
      e.build_started_at = None
      e.worker_id = None
    update_entry(queue_file, claimed.job_id, _submitted)
    log.append(f'[worker] {claimed.job_id} -> SUBMITTED xid={xid} cell={placement.cell}')
    return 'submitted', log, new_fail_count

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

  entries = load_queue(_QUEUE_FILE.value)

  if _REROUTE.value:
    # Sweep SUBMITTED jobs stuck PENDING; cancel + return to QUEUED.
    updated, log = run_reroute(
        entries, now=time.time(), probe=XManagerStatusProbe(),
        reroute_after_s=_REROUTE_AFTER_S.value, cooldown_s=_COOLDOWN_S.value,
        dry_run=_DRY_RUN.value)
  else:
    # Drain QUEUED jobs into the XM queue.
    provider = avail_provider.AvailabilityProvider(group=_GROUP.value)
    updated, log = run_tick(
        entries, provider, now=time.time(),
        dry_run=_DRY_RUN.value, group=_GROUP.value,
        max_placements=_MAX_PLACEMENTS.value, verbose=_VERBOSE.value)

  for line in log:
    print(line)
  if not _DRY_RUN.value:
    save_queue(_QUEUE_FILE.value, updated)
    print(f'[route_check] queue saved to {_QUEUE_FILE.value}')
  else:
    print('[route_check] DRY RUN -- queue not modified. Pass --nodry_run to act.')


if __name__ == '__main__':
  app.run(main)
