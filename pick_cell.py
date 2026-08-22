r"""Pick the cell `tpu queue` should pin RIGHT NOW for one `--tpu_type`.

This is what makes smart cell selection the DEFAULT for `tpu queue`: before a
launch, the wrapper asks this for the best cell that can actually place the
slice (most free chips, not oversold), and pins `--cell=<that>`. The scarce
resource is placeable slices per cell, and the allocator, left alone, kept
landing jobs on an oversold cell while the same accelerator sat idle elsewhere
(13 v7-32 jobs pinned to oversold `yulpptr` while `yukulwh` had 101 free).

Contract with the wrapper is deliberately minimal and FAIL-SAFE:
  * prints ONE line -- the cell name -- and exits 0 when it has a pick;
  * prints NOTHING and exits non-zero when it cannot recommend (no candidate,
    RPC failed, unparseable type). The wrapper then falls back to today's
    behaviour (let the allocator choose), so this can only help, never block.

It reuses avail_provider (free chips decide) + route_lib.best_cell_for_shape
(the same ranking the router uses), restricted to the ONE arch of the requested
type, so it costs a single GetCellAvailability RPC.

    tpu_pick_cell --tpu_type=v7-32 --group=9
"""

from __future__ import annotations

import sys
import time
from typing import Optional, Protocol

from absl import app
from absl import flags

from google3.experimental.users.qiaos.tpu_utils import avail_provider
from google3.experimental.users.qiaos.tpu_utils import route_lib


class _Provider(Protocol):
  """What pick() needs: a fetch() yielding (avail_by_cell, ...). The real
  AvailabilityProvider or a fake in tests."""

  def fetch(self) -> tuple[dict[str, route_lib.CellAvail], dict[str, float],
                           dict[str, float]]:
    ...


_TPU_TYPE = flags.DEFINE_string(
    'tpu_type', None, 'Accelerator type, e.g. v7-32. REQUIRED. A comma list '
    '(v4-64,v5p-32) is rejected -- pin one type per submit.')
_GROUP = flags.DEFINE_string('group', avail_provider.DEFAULT_GROUP,
                             'Alloc group (same numbering as tpu queue).')
_TIER = flags.DEFINE_string('tier', 'PROD', 'PROD | BATCH (unused for the pick '
                            'today; kept so the flag surface matches tpu queue).')
_METROS = flags.DEFINE_list(
    'metros', None, 'Optional metro allow-list; empty = any metro.')
_MAX_PRICE = flags.DEFINE_float(
    'max_price', None, 'Skip cells pricier than this (credits/chip-hr).')


def parse_tpu_type(tpu_type: str) -> Optional[tuple[str, int]]:
  """'v7-32' -> ('v7', 32). None if unparseable or a multi-type list.

  A bare comma list means the caller wants several types; picking one cell for
  that is ambiguous, so we decline (the wrapper keeps its own behaviour).
  """
  s = (tpu_type or '').strip().lower()
  if not s or ',' in s:
    return None
  for sep in ('-', '='):
    if sep in s:
      arch, cores = s.split(sep, 1)
      arch = arch.strip()
      if not arch.isalnum():
        return None
      try:
        chips = int(cores)
      except ValueError:
        return None
      if chips <= 0:
        return None
      return arch, chips
  return None


def pick(tpu_type: str,
         provider: _Provider,
         now: Optional[float] = None,
         metros: Optional[list[str]] = None,
         max_price: Optional[float] = None) -> Optional[str]:
  """The best cell name for `tpu_type`, or None. Pure over an injected provider.

  Builds a throwaway QueueEntry carrying only what the ranking needs (the arch,
  the metro/price constraints) and defers to route_lib.best_cell_for_shape, so
  the pick matches the router exactly: metro filter, price cap, oversold drop,
  >=1 placeable slice, ranked by free slices then price then headroom.
  """
  parsed = parse_tpu_type(tpu_type)
  if parsed is None:
    return None
  arch, chips = parsed
  now = time.time() if now is None else now
  try:
    avail_by_cell, _, _ = provider.fetch()
  except Exception:  # pylint: disable=broad-except
    return None
  entry = route_lib.QueueEntry(
      job_id='_pick', power=f'{arch}-{chips}', allowed_archs=[arch],
      allowed_metros=[m.strip() for m in metros] if metros else None,
      max_price=max_price)
  hit = route_lib.best_cell_for_shape(arch, chips, entry, avail_by_cell, now)
  if hit is None:
    return None
  cell_avail, _ = hit
  return cell_avail.cell


def main(argv):
  del argv
  if not _TPU_TYPE.value:
    return 1
  parsed = parse_tpu_type(_TPU_TYPE.value)
  if parsed is None:
    return 1
  arch, _ = parsed
  # ONE RPC: restrict the provider to just this arch.
  provider = avail_provider.AvailabilityProvider(archs=[arch], group=_GROUP.value)
  cell = pick(_TPU_TYPE.value, provider, metros=_METROS.value,
              max_price=_MAX_PRICE.value)
  if not cell:
    return 1
  print(cell)
  return 0


if __name__ == '__main__':
  sys.exit(app.run(main))
