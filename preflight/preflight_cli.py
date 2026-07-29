"""CLI entry point for preflight. Invoked from tpu_wrapper.sh.

Exit codes:
  0 = GREEN or YELLOW (submit allowed; YELLOW prints warnings)
  1 = RED (do not submit; user should fix the request or pass --force)
  2 = internal error (crashed before verdict)
"""

import sys
from absl import app, flags

from google3.experimental.users.qiaos.tpu_utils.preflight import preflight
from google3.experimental.users.qiaos.tpu_utils import group_utils

_TPU_TYPE = flags.DEFINE_string('tpu_type', '', 'e.g. v6e-16')
_GROUP    = flags.DEFINE_string('group', '', 'Group id (1..8) OR full alloc string')
_TIER     = flags.DEFINE_string('tier', 'PROD', 'PROD | BATCH | ""')
_JSON     = flags.DEFINE_bool('json', False, 'Emit JSON verdict (for scripts)')
_OFFLINE  = flags.DEFINE_bool('offline', False, 'Skip capacity RPC')

RED   = '\033[31m'
YELLOW= '\033[33m'
GREEN = '\033[32m'
BOLD  = '\033[1m'
RESET = '\033[0m'


def _color_lines(lines: list[str], status: preflight.Status) -> list[str]:
  color = {'GREEN': GREEN, 'YELLOW': YELLOW, 'RED': RED}[status.value]
  return [f"{color}{lines[0]}{RESET}"] + lines[1:]


def _resolve_alloc(group_arg: str) -> str:
  """Accepts either a group id (int/str) or a full 'group:...' alloc string."""
  if not group_arg:
    return ''
  if group_arg.startswith('group:'):
    return group_arg
  # Try group_utils lookup (accepts '1' or 'g1').
  alloc = group_utils.get_alloc_by_id(group_arg)
  return alloc or group_arg


def main(argv):
  del argv
  if not _TPU_TYPE.value:
    print('Error: --tpu_type required', file=sys.stderr)
    return 2
  if not _GROUP.value:
    print('Error: --group required', file=sys.stderr)
    return 2

  alloc = _resolve_alloc(_GROUP.value)
  if not alloc:
    print(f'Error: cannot resolve group {_GROUP.value!r} to an alloc',
          file=sys.stderr)
    return 2

  try:
    verdict = preflight.run_preflight(
        tpu_type=_TPU_TYPE.value, alloc=alloc, tier=_TIER.value,
        skip_capacity=_OFFLINE.value)
  except Exception as e:
    print(f'preflight internal error: {type(e).__name__}: {e}', file=sys.stderr)
    return 2

  if _JSON.value:
    import json
    from typing import Any
    payload: dict[str, Any] = {
        'status': verdict.status.value,
        'reasons': list(verdict.reasons),
        'tpu_type': _TPU_TYPE.value,
        'alloc': alloc,
        'tier': _TIER.value,
    }
    if verdict.capacity:
      payload['cells_ok'] = [
          {'cell': c.cell, 'chips': c.obtainable}
          for c in verdict.capacity.cells_ok[:10]]
      payload['total_obtainable'] = verdict.capacity.total_obtainable
      payload['total_within_floor'] = verdict.capacity.total_pool_capacity
    print(json.dumps(payload, indent=2))
  else:
    for line in _color_lines(verdict.as_console_lines(), verdict.status):
      print(line)

  return 0 if verdict.status != preflight.Status.RED else 1


if __name__ == '__main__':
  app.run(main)
