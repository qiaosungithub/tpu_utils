"""Live availability provider for the local-queue router.

WHAT THIS PRODUCES, and why the router needs exactly these three things:

  avail_by_cell : dict[cell -> CellAvail]   free chips + oversold per cell, the
                  DECIDING placement signal (route_lib.best_cell_for_shape).
  arch_price    : dict[arch -> credits/chip-hr]   the global PROD clearing price
                  per accelerator, for effective-price type selection.
  arch_pool     : dict[arch -> float]   live pool magnitude per accelerator
                  (sum of free chips across its cells), so a big easy-to-get
                  pool earns a price discount (route_lib.pool_weight).

It wraps the SAME `GoodputService.GetCellAvailability` RPC that slice_probe.py
uses -- free chips (`max_available_chips`) is the number that decides, NOT
`obtainable_capacity`, which reported 1616 while a cell held 3 free chips.

The pure functions (metro_of, parse_cell_availability, build_availability,
load_prices) take no google3 dependency and are unit-tested with fakes; the
google3 RPC imports are LAZY, inside AvailabilityProvider.fetch, so this module
imports in a bare interpreter for those tests.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Optional

from google3.experimental.users.qiaos.tpu_utils import route_lib


# --- arch <-> platform / card ---------------------------------------------
# arch -> (scalar_resource platform enum NAME, [market.json card codes]).
# v5e (VIPERLITE_POD 62/60) is intentionally absent: dropped per operator.
# The enum name is resolved lazily against scalar_resource_pb2 so this table
# stays importable without google3.
ARCH_PLATFORM: dict[str, str] = {
    'v4': 'PUFFERFISH',       # 34
    'v5p': 'VIPERFISH',       # 59
    'v6e': 'GHOSTLITE_POD',   # 76 (63 is the same fleet, GHOSTLITE)
    'v6p': 'GHOSTFISH',       # 92
    'v7': 'GHOSTFISHLITE',    # 101
}

# arch -> card codes in market.json price keys (first present wins for price).
ARCH_CARDS: dict[str, list[int]] = {
    'v4': [34],
    'v5p': [59],
    'v6e': [76, 63],
    'v6p': [92],
    'v7': [101],
}

# The alloc/group the router submits under (same as slice_probe --group=9).
DEFAULT_GROUP = '9'
_GROUP_MAP = {
    '1': 'group:deepmind-dynamic/gdm-resources-prod-shared-users-dynamic',
    '2': 'group:deepmind-dynamic/gdm-viscam-goflow-dynamic',
    '3': 'group:deepmind-dynamic/gdm-viscam-interns-dynamic',
    '4': 'group:deepmind-dynamic/viscam-interns',
    '5': 'group:deepmind-dynamic/vqfree-xm',
    '6': 'group:dm/deepmind-large-scale-workshop',
    '7': 'group:dm/dm-resources-prod-shared',
    '8': 'group:gdm-aux/brain-vasp-shared-user-xm',
    '9': 'group:deepmind-dynamic/fr-dna-grand-challenge-team-resource',
}

DEFAULT_MARKET_JSON = os.path.expanduser('~/.tpu_quota_cache_dir/market.json')
DEFAULT_PRICE_POOL = 'deepmind-dynamic-pool'

# Cells whose metro is not recoverable from the `yu<metro>...` name pattern.
# Seeded from xm_launcher._CELL_BUCKETS. Only consulted when a job opts into an
# allowed-metros filter; the default (empty filter) never needs it.
_METRO_OVERRIDES: dict[str, str] = {
    'dl': 'las',   # las -> dl-d  (2nd v4 cell)
    'je': 'cbf',   # cbf neighbour
    'nl': 'tul',   # tul neighbour
    'nk': 'tul',   # tul neighbour
    'el': 'grq',
    'mb': 'ckv',
    'sk': 'sin', 'sn': 'sin', 'so': 'sin',
}


def metro_of(cell: str) -> str:
  """Best-effort metro token for a borg cell name.

  The fleet's full cell names encode the metro as `yu<metro><suffix>`
  (yutulpz -> tul, yulpptr -> lpp, yucbfiv -> cbf, yudfwra -> dfw,
  yuskedq -> ske). Short/legacy names fall back to an override table, then to
  the cell name itself. Metro precision only matters when a job sets
  allowed_metros; by default there is no metro filter.
  """
  c = (cell or '').lower()
  if c in _METRO_OVERRIDES:
    return _METRO_OVERRIDES[c]
  if c.startswith('yu') and len(c) >= 5:
    return c[2:5]
  return c


def load_prices(market_json_path: str = DEFAULT_MARKET_JSON,
                pool: str = DEFAULT_PRICE_POOL) -> dict[str, float]:
  """arch -> global PROD credits/chip-hr, from the money checker's market cache.

  Returns {} if the cache is missing; a caller with no prices falls back to
  ARCH_PREF ordering (route_lib.candidate_shapes handles a None/empty map).
  """
  try:
    with open(market_json_path) as f:
      market = json.load(f)
  except (OSError, ValueError):
    return {}
  prices = market.get('prices', {})
  out: dict[str, float] = {}
  for arch, cards in ARCH_CARDS.items():
    for card in cards:
      key = f'{pool}|{card}|PROD'
      layer = prices.get(key)
      if isinstance(layer, dict) and 'global' in layer:
        try:
          out[arch] = float(layer['global'])
        except (TypeError, ValueError):
          continue
        break
  return out


def parse_cell_availability(resp: Any, platform_int: int) -> dict[str, tuple[int, bool]]:
  """cell -> (free_chips, oversold) for ONE platform, from a GetCellAvailability
  response. Pure: `resp` is the proto (or a duck-typed fake for tests).

  free chips = `max_available_chips` (the deciding number). `obtainable_capacity`
  is deliberately NOT read -- it is a quota-shaped promise that lies.
  Across tiers, a cell's free chips are summed and its oversold flag OR-ed.
  """
  free: dict[str, int] = {}
  oversold: set[str] = set()
  for tiered in getattr(resp, 'tiered_dynamic_pool_availabilities', []):
    av = getattr(tiered, 'availability', None)
    if av is None:
      continue
    for entry in getattr(av, 'max_available_chips', []):
      for pc in getattr(entry, 'platform_to_chip_counts', []):
        if int(getattr(pc, 'platform', -1)) == platform_int:
          free[entry.cell] = free.get(entry.cell, 0) + int(pc.num_chips)
    for entry in getattr(av, 'oversold_statuses', []):
      if platform_int in [int(p) for p in getattr(entry, 'platforms', [])]:
        oversold.add(entry.cell)
  return {cell: (chips, cell in oversold) for cell, chips in free.items()}


def build_availability(
    per_arch: dict[str, dict[str, tuple[int, bool]]],
    arch_price: dict[str, float],
) -> tuple[dict[str, route_lib.CellAvail], dict[str, float], dict[str, float]]:
  """Assemble the router's three inputs from parsed per-arch cell data. Pure.

  `per_arch[arch]` = {cell -> (free_chips, oversold)}.

  A single borg cell can host TWO accelerator generations at once -- `je`
  carries both a v6e and a v7 pod, `nk`/`nl` both v6e and v6p -- confirmed live
  by the RPC. So avail_by_cell is keyed per (cell, arch) as `cell|arch`, NOT by
  bare cell name, or the second generation would silently overwrite the first
  and the router would never see it. route_lib scans .values() filtered by arch
  and matches placements by content, so the key shape is opaque to it.
  arch_pool[arch] = sum of free chips across that arch's cells (live magnitude).
  """
  avail_by_cell: dict[str, route_lib.CellAvail] = {}
  arch_pool: dict[str, float] = {}
  for arch, cells in per_arch.items():
    pool = 0
    price = arch_price.get(arch)
    for cell, (free_chips, oversold) in cells.items():
      pool += max(0, free_chips)
      avail_by_cell[f'{cell}|{arch}'] = route_lib.CellAvail(
          cell=cell, arch=arch, free_chips=free_chips, oversold=oversold,
          price=price, metro=metro_of(cell))
    arch_pool[arch] = float(pool)
  return avail_by_cell, arch_price, arch_pool


class AvailabilityProvider:
  """Fetches live availability via GetCellAvailability, one RPC per arch.

  The RPC stub + alloc lookup are injectable so the fetch path is unit-testable
  with fakes; google3 imports happen lazily in fetch() (and in the default
  factories) so the pure helpers above import in a bare interpreter.
  """

  def __init__(self,
               archs: Optional[list[str]] = None,
               group: str = DEFAULT_GROUP,
               market_json_path: str = DEFAULT_MARKET_JSON,
               stub_factory: Optional[Callable[[], Any]] = None,
               alloc_resolver: Optional[Callable[[str], Any]] = None,
               platform_enum: Optional[Callable[[str], int]] = None,
               request_factory: Optional[Callable[[], Any]] = None,
               deadline_s: float = 60.0):
    self.archs = list(archs) if archs else list(ARCH_PLATFORM.keys())
    self.group = group
    self.market_json_path = market_json_path
    self._stub_factory = stub_factory
    self._alloc_resolver = alloc_resolver
    self._platform_enum = platform_enum
    self._request_factory = request_factory
    self.deadline_s = deadline_s

  # -- default google3-backed factories (lazy imports) ----------------------
  def _default_stub(self) -> Any:
    import datetime
    from google3.borg.xborg.frontend.goodput_optimizer.proto import goodput_optimizer_service_pb2
    from google3.net.rpc.python.contrib import rpc_factory_factory
    from google3.net.rpc2.contrib.smartservice.python import smartservice_util
    return smartservice_util.new_stub(
        goodput_optimizer_service_pb2.GoodputService,
        smartservice_util.parse('blade:xborg-prod-routing-layer'),
        rpc_factory=rpc_factory_factory.new_factory(
            deadline=datetime.timedelta(seconds=self.deadline_s)))

  def _resolve_alloc(self, group: str) -> Any:
    from google3.learning.deepmind.xmanager2.client import resource_service
    spec = (group or '').strip()
    alloc = _GROUP_MAP.get(spec, spec)
    return resource_service.get_resource_alloc(alloc)

  def _platform_int(self, arch: str) -> int:
    from google3.borg.common import scalar_resource_pb2
    name = ARCH_PLATFORM[arch]
    return int(getattr(scalar_resource_pb2.ScalarResource.Key, name))

  def _default_request(self) -> Any:
    from google3.borg.xborg.frontend.goodput_optimizer.proto import goodput_optimizer_service_pb2
    return goodput_optimizer_service_pb2.GetCellAvailabilityRequest()

  # -- the live fetch -------------------------------------------------------
  def fetch(self) -> tuple[dict[str, route_lib.CellAvail], dict[str, float], dict[str, float]]:
    """One RPC per arch -> the router's (avail_by_cell, arch_price, arch_pool)."""
    stub = (self._stub_factory or self._default_stub)()
    resolve = self._alloc_resolver or self._resolve_alloc
    platform_of = self._platform_enum or self._platform_int
    make_request = self._request_factory or self._default_request
    details = resolve(self.group)

    per_arch: dict[str, dict[str, tuple[int, bool]]] = {}
    for arch in self.archs:
      platform_int = platform_of(arch)
      req = make_request()
      req.resource_pool = details.resource_pool_name
      req.allowed_allotments.append(details.xborg_allotment_name)
      req.platforms.append(platform_int)
      try:
        resp = stub.GetCellAvailability(req)
      except Exception as e:  # pylint: disable=broad-except
        print(f'[avail_provider] GetCellAvailability failed for {arch}: {e}')
        continue
      per_arch[arch] = parse_cell_availability(resp, platform_int)

    arch_price = load_prices(self.market_json_path)
    return build_availability(per_arch, arch_price)
