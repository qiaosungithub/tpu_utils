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
import sys
import time
import uuid

from absl import app
from absl import flags

from google3.experimental.users.qiaos.tpu_utils import avail_provider
from google3.experimental.users.qiaos.tpu_utils import route_check
from google3.experimental.users.qiaos.tpu_utils import route_lib


# `queue_file` and `group` are already defined by route_check (imported above);
# reuse those flag objects rather than redefining them, or absl raises
# DuplicateFlagError the moment both modules load in one binary.
_QUEUE_FILE = route_check._QUEUE_FILE  # pylint: disable=protected-access
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
_JOB_ID = flags.DEFINE_string(
    'job_id', None, 'Local id (enqueue: optional, auto-generated; dequeue: the '
    'id(s), comma-separated, to remove).')
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


def _new_job_id(power: str) -> str:
  return f'{power}-{uuid.uuid4().hex[:6]}'


def _cmd_enqueue(argv: list[str]) -> int:
  if not _POWER.value or not _ARCHS.value:
    print('enqueue: --power and --archs are REQUIRED.\n'
          '  e.g. tpu enqueue --power=v7-32 --archs=v7,v6p '
          '--launch=config=configs/eqr.py', file=sys.stderr)
    return 2
  job_id = _JOB_ID.value or _new_job_id(_POWER.value)
  # workdir default = the CWD at enqueue time, so enqueuing from the right
  # checkout just works. A flag value of '' explicitly opts into the router's
  # own dir (only safe when every difference is an explicit --flag).
  workdir = _WORKDIR.value if _WORKDIR.value is not None else os.getcwd()
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
  )
  # The whole read-modify-write runs under the queue lock: load the LIVE queue,
  # check for a duplicate id, append, save -- atomically. Loading outside the
  # lock (the old bug) let a concurrent route tick's whole-queue overwrite land
  # between our load and save and silently drop this freshly enqueued row.
  with route_check.with_queue_lock(_QUEUE_FILE.value):
    entries = route_check.load_queue(_QUEUE_FILE.value)
    if any(e.job_id == job_id for e in entries):
      print(f'enqueue: job_id {job_id!r} already in the queue; pass a different '
            '--job_id.', file=sys.stderr)
      return 1
    entries.append(entry)
    route_check.save_queue(_QUEUE_FILE.value, entries)
  print(f'enqueued {job_id}: power={entry.power} archs={entry.allowed_archs} '
        f'tier={entry.tier} priority={entry.priority}'
        + (f' metros={entry.allowed_metros}' if entry.allowed_metros else '')
        + (' [topology-locked]' if entry.topology_locked else ''))
  print(f'  packaged from: {entry.workdir or "(router process dir)"}')
  print(f'  queue now holds {len(entries)} job(s). See: tpu queue-status')
  return 0


def _cmd_dequeue(argv: list[str]) -> int:
  ids = set()
  if _JOB_ID.value:
    ids |= {x.strip() for x in _JOB_ID.value.split(',')}
  ids |= {a for a in argv[1:] if not a.startswith('-')}
  if not ids:
    print('dequeue: pass job_id(s): tpu dequeue <job_id> [<job_id> ...]',
          file=sys.stderr)
    return 2
  with route_check.with_queue_lock(_QUEUE_FILE.value):
    entries = route_check.load_queue(_QUEUE_FILE.value)
    keep = [e for e in entries if e.job_id not in ids]
    removed = [e.job_id for e in entries if e.job_id in ids]
    if not removed:
      print(f'dequeue: none of {sorted(ids)} found in the queue.', file=sys.stderr)
      return 1
    route_check.save_queue(_QUEUE_FILE.value, keep)
  for r in removed:
    print(f'dequeued {r}')
  missing = ids - set(removed)
  if missing:
    print(f'  (not found: {sorted(missing)})')
  print(f'  queue now holds {len(keep)} job(s).')
  return 0


def _cmd_requeue(argv: list[str]) -> int:
  """Return HELD job(s) to QUEUED after the operator has fixed the cause. With
  no id, requeues ALL held jobs."""
  ids = set()
  if _JOB_ID.value:
    ids |= {x.strip() for x in _JOB_ID.value.split(',')}
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
    print(f'{e.job_id:22s} {state_disp:18s} {e.power:9s} {archs:14s} '
          f'{e.priority:>4d}  {why}{lock}')
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
