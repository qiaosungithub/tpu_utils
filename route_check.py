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
from typing import Optional, Protocol

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
  recorder in tests."""

  def submit(self, argv: list[str]) -> tuple[Optional[str], str]:
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

  def submit(self, argv: list[str]) -> tuple[Optional[str], str]:
    # argv[0] is 'tpu' (a shell function); build a sourced-shell command.
    inner = ' '.join(_shquote(a) for a in argv)
    script = f'source {_shquote(self.wrapper_path)} >/dev/null 2>&1; {inner}'
    try:
      proc = subprocess.run(['bash', '-c', script], capture_output=True,
                            text=True, timeout=self.timeout_s)
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
    if dry_run:
      argv = build_tpu_queue_cmd(p, entry, group)
      log.append(f'[DRY] would place {p.job_id}: {p.reason}')
      log.append(f'      cmd: {" ".join(argv)}')
      continue
    # live submit
    sub = submitter or Submitter()
    argv = build_tpu_queue_cmd(p, entry, group)
    log.append(f'[route_check] placing {p.job_id}: {p.reason}')
    log.append(f'      cmd: {" ".join(argv)}')
    xid, out = sub.submit(argv)
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


def main(argv):
  del argv
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
