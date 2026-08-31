"""L2 check: per-cell chip availability via GoodputService.GetCellAvailability.

We call `blade:xborg-prod-routing-layer` (verified reachable from citc VM,
see preflight_probe.py). Response gives per-(cell, tier, allotment) chip
counts. We do NOT get topology-level free-slice info here — the only API
that gives that is BorgMaster.ProbeSliceAvailability, which has no Python
stubby wrapper. For the topology-fragmentation case, L2 only guarantees
'enough total chips exist somewhere'; the actual reject on fragmentation
still has to be caught by the daemon retry loop.
"""

import dataclasses
import datetime
import time
from typing import Optional

from google3.borg.common import scalar_resource_pb2
from google3.borg.xborg.frontend.goodput_optimizer.proto import (
    goodput_optimizer_service_pb2)
from google3.learning.deepmind.xmanager2.client import resource_service
from google3.net.rpc.python.contrib import rpc_factory_factory
from google3.net.rpc2.contrib.smartservice.python import smartservice_util

_GOODPUT_ADDRESS = 'blade:xborg-prod-routing-layer'
_DEFAULT_DEADLINE = datetime.timedelta(seconds=30)

# Map user-facing tier string -> GetCellAvailabilityResponse.ResourceTier enum
# (see goodput_optimizer_service.proto:770).
_TIER_ENUM = {
    'PROD': 1,   # XBORG_HIGHLY_AVAILABLE
    'BATCH': 5,  # XBORG_NON_PROD
}

# Simple in-process cache: (pool, allotment_tuple, platform_key) -> (ts, response).
# TTL kept short (30 s) because market state changes ~1 min.
_CACHE_TTL_S = 30
_response_cache: dict[tuple, tuple[float, object]] = {}


@dataclasses.dataclass(frozen=True)
class CellCapacity:
  """Chip availability in one cell."""
  cell: str
  tier: str                    # 'PROD' or 'BATCH'
  within_floor: int            # guaranteed chips (allotment's floor)
  max_available: int           # opportunistic ceiling (>= within_floor)
  # obtainable_capacity = xborg's estimate of chips schedulable now, netting
  # out others' usage. NOTE: response uses per-cell aggregation across
  # allotments; we sum per (allotment, cell) here.
  obtainable: int = 0


@dataclasses.dataclass(frozen=True)
class CapacityResult:
  """Outcome of the L2 check."""
  ok: bool
  cells_ok: tuple[CellCapacity, ...]     # cells with >= requested chips
  cells_insufficient: tuple[CellCapacity, ...]  # cells with any chips, but < requested
  total_pool_capacity: int               # sum across all cells (within-floor at requested tier)
  total_obtainable: int                  # sum of obtainable across all cells
  # Alloc-scoped quota (from ResourceAllocationDetails.floor_v2, aggregated at
  # the tier we care about across all cells). This is the number that shows
  # up in `tpu quota` for the user's own group, and is what the user usually
  # thinks of as 'my quota'.
  alloc_scoped_quota: int = 0
  alloc_scoped_used: int = 0
  # The GQM ResourcePool this alloc lives in (e.g. 'deepmind-dynamic-pool').
  # Carried out of the check because market prices and limit orders are BOTH
  # keyed by pool: the same (cell, chip, tier) cleared at 20.20 credits in
  # deepmind-dynamic-pool and 32.89 in gemini-dynamic-pool at the same instant,
  # so a price join without the pool silently mixes markets.
  pool: str = ''
  hard_error: Optional[str] = None
  warnings: tuple[str, ...] = ()
  raw_response_repr: str = ''            # truncated repr for debugging


def _get_stub():
  return smartservice_util.new_stub(
      goodput_optimizer_service_pb2.GoodputService,
      smartservice_util.parse(_GOODPUT_ADDRESS),
      rpc_factory=rpc_factory_factory.new_factory(deadline=_DEFAULT_DEADLINE))


_alloc_meta_cache: dict[str, tuple[str, str]] = {}


def _resolve_pool_allotment(alloc: str) -> tuple[str, str]:
  """alloc string -> (resource_pool, xborg_allotment) via resource_service.

  Cached forever within a process; alloc metadata is essentially static.
  """
  hit = _alloc_meta_cache.get(alloc)
  if hit is not None:
    return hit
  details = resource_service.get_resource_alloc(alloc)
  hit = (details.resource_pool_name, details.xborg_allotment_name)
  _alloc_meta_cache[alloc] = hit
  return hit


def _fetch_cell_availability(pool: str, allotments: tuple[str, ...],
                             platform_key_enum: int):
  """Call GetCellAvailability with caching. Returns the raw response proto."""
  cache_key = (pool, allotments, platform_key_enum)
  now = time.time()
  cached = _response_cache.get(cache_key)
  if cached and now - cached[0] < _CACHE_TTL_S:
    return cached[1]

  req = goodput_optimizer_service_pb2.GetCellAvailabilityRequest()
  req.resource_pool = pool
  for a in allotments:
    req.allowed_allotments.append(a)
  # pytype: disable=bad-argument-type  # int is accepted for enum field.
  req.platforms.append(platform_key_enum)
  # pytype: enable=bad-argument-type

  stub = _get_stub()
  resp = stub.GetCellAvailability(req)
  _response_cache[cache_key] = (now, resp)
  return resp


def _extract_cells(resp, tier: str, platform_key_enum: int) -> list[CellCapacity]:
  """Walk the response and pull out per-cell chip counts at the target tier.

  Response layout (see goodput_optimizer_service.proto):
    resp.tiered_dynamic_pool_availabilities: repeated {tier, availability}
      availability.allotment_availability: repeated {allotment, obtainable_capacity}
        obtainable_capacity: repeated CellAvailability {cell, platform_to_chip_counts}
    resp.tiered_static_pool_availabilities: same shape but for static pools
    resp.dynamic_pool_availability: pool-level (no allotment filter)
  """
  want_tier_enum = _TIER_ENUM.get(tier.upper())
  cells: dict[str, CellCapacity] = {}
  # Live opportunistic ceiling per cell, read from the pool-level
  # `max_available_chips` (DynamicPoolAvailability field 2 / StaticPool field 3).
  # This is the market-level free-chip count (within-floor + acquirable), which
  # for a free-pool arch like GB200 is far larger than the per-allotment
  # `obtainable_capacity` forecast -- folding it in stops a healthy free pool
  # from being flagged RED purely because the allotment's obtainable is low.
  free_by_cell: dict[str, int] = {}
  def _walk_tiered(tiered_list):
    for tier_bucket in tiered_list:
      if tier_bucket.tier != want_tier_enum:
        continue
      av = tier_bucket.availability
      # Different pool types populate different subfields; be defensive.
      allotment_list = getattr(av, 'allotment_availability', None) or []
      for alloc_av in allotment_list:
        obtainable_list = getattr(alloc_av, 'obtainable_capacity', None) or []
        for obtain_cap in obtainable_list:
          cell = obtain_cap.cell
          for ptcc in obtain_cap.platform_to_chip_counts:
            if ptcc.platform != platform_key_enum:
              continue
            chips = int(ptcc.num_chips)
            prev = cells.get(cell)
            if prev is None:
              cells[cell] = CellCapacity(
                  cell=cell, tier=tier, within_floor=chips,
                  max_available=chips, obtainable=chips)
            else:
              cells[cell] = dataclasses.replace(
                  prev,
                  within_floor=prev.within_floor + chips,
                  obtainable=prev.obtainable + chips)
      # Pool-level max_available_chips: same CellAvailability shape, but hangs
      # directly off `availability` (not per-allotment). Sum matching-platform
      # chips per cell into free_by_cell.
      for cell_av in (getattr(av, 'max_available_chips', None) or []):
        cell = cell_av.cell
        for ptcc in cell_av.platform_to_chip_counts:
          if ptcc.platform != platform_key_enum:
            continue
          free_by_cell[cell] = free_by_cell.get(cell, 0) + int(ptcc.num_chips)

  _walk_tiered(resp.tiered_dynamic_pool_availabilities)
  _walk_tiered(resp.tiered_static_pool_availabilities)

  # Fold the live free-chip ceiling into each cell's max_available. A cell that
  # only shows up in max_available_chips (no allotment obtainable) still becomes
  # a candidate; a cell present in both takes the larger of the two.
  for cell, free in free_by_cell.items():
    prev = cells.get(cell)
    if prev is None:
      cells[cell] = CellCapacity(
          cell=cell, tier=tier, within_floor=0,
          max_available=free, obtainable=0)
    else:
      cells[cell] = dataclasses.replace(
          prev, max_available=max(prev.max_available, free))
  return list(cells.values())


def check_capacity(alloc: str, tier: str, xm_accelerator_key: str,
                   borg_platform_key: str, chips_required: int
                   ) -> CapacityResult:
  """L2 capacity check for a single (alloc, tier, platform, chips).

  Returns a CapacityResult; `ok=False` when no cell has >= chips_required.
  """
  # Resolve alloc.
  try:
    pool, allotment = _resolve_pool_allotment(alloc)
  except Exception as e:
    return CapacityResult(
        ok=False, cells_ok=(), cells_insufficient=(),
        total_pool_capacity=0, total_obtainable=0,
        hard_error=f"Cannot resolve alloc '{alloc}': {type(e).__name__}: {e}")

  # Translate platform string to enum.
  try:
    platform_enum = getattr(scalar_resource_pb2.ScalarResource.Key,
                            borg_platform_key)
  except AttributeError:
    return CapacityResult(
        ok=False, cells_ok=(), cells_insufficient=(),
        total_pool_capacity=0, total_obtainable=0,
        hard_error=f"Unknown Borg platform key: {borg_platform_key}")

  # Call the RPC.
  try:
    resp = _fetch_cell_availability(pool, (allotment,), platform_enum)
  except Exception as e:
    return CapacityResult(
        ok=False, cells_ok=(), cells_insufficient=(),
        total_pool_capacity=0, total_obtainable=0,
        hard_error=(f"GoodputService.GetCellAvailability failed: "
                    f"{type(e).__name__}: {e}"))

  cells = _extract_cells(resp, tier, platform_enum)
  if not cells and tier.upper() != 'PROD':
    # GetCellAvailability only ever returns the XBORG_HIGHLY_AVAILABLE tier
    # bucket for our dynamic pool -- verified live: the response carries a
    # single `tiered_dynamic_pool_availabilities` entry with tier=1 and no
    # tier=5 entry at all, so a BATCH query comes back empty everywhere.
    #
    # Reading that as "no BATCH capacity" is wrong, and it is what made every
    # BATCH route degenerate into an unranked list of RED-ish rows. The
    # alloc-scoped forecast does carry per-cell NonProd numbers (8,379 v5p
    # chips across 13 cells at the same instant), so fall back to it.
    #
    # It is a fallback rather than the primary source because it costs a much
    # slower RPC (~25 s vs ~1 s) and is a forecast rather than a live figure.
    cells = _fetch_forecast_cells(alloc, tier, xm_accelerator_key)
  # Effective placeable chips per cell = max of the allotment obtainable-capacity
  # forecast and the live pool-level free ceiling (max_available). A free-pool
  # arch (e.g. GB200) can have a tiny allotment obtainable but a large live free
  # pool; using only obtainable would flag it RED even though a job would place.
  def _eff(c: CellCapacity) -> int:
    return max(c.obtainable, c.max_available)

  cells_ok = tuple(sorted([c for c in cells if _eff(c) >= chips_required],
                          key=lambda c: -_eff(c)))
  cells_insufficient = tuple(sorted([c for c in cells if 0 < _eff(c) < chips_required],
                                    key=lambda c: -_eff(c)))
  total_cap = sum(c.within_floor for c in cells)
  total_obt = sum(_eff(c) for c in cells)

  # Alloc-scoped quota (via floor_v2): what the user sees in tpu quota.
  # This is stricter than the pool-wide obtainable_capacity from GoodputService.
  alloc_quota, alloc_used = _fetch_alloc_scoped_quota(
      alloc, tier, xm_accelerator_key)

  warnings = []
  if not cells_ok:
    hard_error = (
        f"No cell in {alloc} at {tier} has {chips_required} available "
        f"(max of obtainable-forecast and live-free) "
        f"{xm_accelerator_key} chips. Total available across "
        f"{len(cells)} cells: {total_obt}.")
    if cells_insufficient:
      top_hint = ', '.join(f'{c.cell}:{_eff(c)}' for c in cells_insufficient[:5])
      hard_error += f" Top cells (chips available): {top_hint}."
    return CapacityResult(
        ok=False, cells_ok=(), cells_insufficient=cells_insufficient,
        total_pool_capacity=total_cap, total_obtainable=total_obt,
        alloc_scoped_quota=alloc_quota, alloc_scoped_used=alloc_used,
        pool=pool, hard_error=hard_error, warnings=tuple(warnings))

  # Heuristic: warn if the user's own alloc quota is thin vs the request.
  # This is more meaningful than the pool ceiling because that's what
  # actually caps a PROD job at admission time.
  if tier.upper() == 'PROD' and alloc_quota > 0:
    remaining = max(0, alloc_quota - alloc_used)
    if remaining < 2 * chips_required:
      warnings.append(
          f"PROD quota headroom is thin: your alloc has quota={alloc_quota}, "
          f"used={alloc_used}, remaining={remaining}, request={chips_required}. "
          f"If quota was granted per cell, submission may still fail on a "
          f"single-cell shortage even though the sum is enough.")
  elif tier.upper() == 'PROD' and alloc_quota == 0:
    warnings.append(
        f"Could not read PROD quota for {xm_accelerator_key} in {alloc} "
        f"(floor_v2 reported 0). Cannot verify headroom.")

  return CapacityResult(
      ok=True, cells_ok=cells_ok, cells_insufficient=cells_insufficient,
      total_pool_capacity=total_cap, total_obtainable=total_obt,
      alloc_scoped_quota=alloc_quota, alloc_scoped_used=alloc_used,
      pool=pool, warnings=tuple(warnings))


# Cache the whole forecast per alloc: one call answers every (tier, platform)
# question, and the router asks ~5 archs x 9 groups in one run.
_forecast_cache: dict[str, tuple[float, object]] = {}
_FORECAST_TTL_S = 120

_TIER_TO_PRIORITY = {'PROD': 'HighlyAvailable', 'BATCH': 'NonProd',
                     'SPOT': 'BestEffort'}


def _fetch_forecast_cells(alloc: str, tier: str,
                          xm_accelerator_key: str) -> list[CellCapacity]:
  """Per-cell obtainable chips from the resource-service forecast.

  Used only where GetCellAvailability has no data for the tier (BATCH).

  Despite taking an alloc name, the numbers this returns are POOL-WIDE, not
  alloc-scoped: three different allocs in deepmind-dynamic-pool returned byte
  identical forecasts (NonProd v5p: 13 cells, 5,699 chips, same per-cell
  split). That is not a bug to route around -- it matches how BATCH is actually
  admitted, since the BATCH pass tests ``DemandFitsInRootPoolCapacity`` against
  the shared root pool and never consults a per-alloc floor. Two allocs in one
  pool genuinely do have identical BATCH prospects.

  The consequence for callers is that BATCH capacity cannot discriminate
  between same-pool groups, and a ranking built on it will legitimately tie.

  Returns [] on any failure; the caller then reports the tier as empty exactly
  as before, so this can only add signal.
  """
  priority = _TIER_TO_PRIORITY.get(tier.upper())
  if not priority:
    return []
  now = time.time()
  hit = _forecast_cache.get(alloc)
  if hit and now - hit[0] < _FORECAST_TTL_S:
    forecast = hit[1]
  else:
    try:
      forecast = resource_service.get_forecast_info(
          resource_alloc_name=alloc,
          with_global_batch_accelerators_availability=True,
          with_full_availability=True)
    except Exception:  # pylint: disable=broad-except
      # A forecast failure is not fatal: this is a best-effort enrichment of a
      # tier the primary RPC does not cover at all.
      return []
    _forecast_cache[alloc] = (now, forecast)

  out: list[CellCapacity] = []
  for cell, by_priority in getattr(forecast, 'forecast', {}).items():
    res_set = by_priority.priorities.get(priority)
    if not res_set:
      continue
    chips = int(getattr(res_set, xm_accelerator_key, 0) or 0)
    if chips <= 0:
      continue
    # within_floor is left at 0 deliberately: the forecast reports what is
    # obtainable, and at BATCH there is no floor to report -- that pass never
    # consults one.
    out.append(CellCapacity(cell=cell, tier=tier, within_floor=0,
                            max_available=chips, obtainable=chips))
  return out


_quota_cache: dict[tuple[str, str, str], tuple[float, tuple[int, int]]] = {}


def _fetch_alloc_scoped_quota(alloc: str, tier: str,
                              xm_accelerator_key: str) -> tuple[int, int]:
  """Reads this alloc's own guaranteed floor and its live usage.

  Quota comes from ``ResourceAllocationDetails.floor_v2``, which is scoped to
  the alloc. It must NOT be read from ``get_pool_capacity`` /
  ``list_resources`` capacities: those describe the whole shared resource
  pool (hundreds of cells, shared by many groups), so they overstate a single
  alloc's quota by orders of magnitude and are nearly identical across every
  group sharing that pool.

  Returns (quota, used) as integer chip counts. Returns (0, 0) if the call
  fails or the alloc/type has no quota.
  """
  tier_map = {'PROD': 'HighlyAvailable', 'BATCH': 'NonProd', 'SPOT': 'BestEffort'}
  p_name = tier_map.get(tier.upper())
  if not p_name:
    return (0, 0)
  ck = (alloc, p_name, xm_accelerator_key)
  now = time.time()
  hit = _quota_cache.get(ck)
  if hit and now - hit[0] < _CACHE_TTL_S:
    return hit[1]
  try:
    details = resource_service.get_resource_alloc(resource_alloc_name=alloc)
    floor = details.floor_v2.priorities.get(p_name)
    # Raw proto values are direct chip counts; no milli-unit scaling.
    quota = int(getattr(floor, xm_accelerator_key, 0) or 0) if floor else 0

    # Try to also get 'used' via resource_service.get_resource_usage.
    used = 0
    try:
      u = resource_service.get_resource_usage(alloc, [p_name])
      chips_u = getattr(u, xm_accelerator_key, 0) if u else 0
      used = int(chips_u)
    except Exception:
      pass
    result = (quota, used)
    _quota_cache[ck] = (now, result)
    return result
  except Exception:
    return (0, 0)
