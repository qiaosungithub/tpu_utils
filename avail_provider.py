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

import importlib
import json
import os
import re
import sys
from typing import Any, Callable, Optional

from google3.experimental.users.qiaos.tpu_utils import metro_util
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
    # NVIDIA GPUs. Enum names verified against borg/common/scalar_resource.proto;
    # values (46/66/70/86/87/112/89/100) match the proto's ScalarResource.Key.
    'a100': 'GPU_TESLA_A100_40GIB',   # 46
    'a100_80gib': 'GPU_TESLA_A100_80GIB',  # 66
    'h100': 'GPU_NVIDIA_H100',        # 70
    'h200': 'GPU_NVIDIA_H200',        # 86
    'b200': 'GPU_NVIDIA_B200',        # 87
    'b300': 'GPU_NVIDIA_B300',        # 112
    # gb200 (89) / gb300 (100) are NOT resolvable: the operator's directive forbids this
    # group from using them, so the router must not be able to name a GB slice at all.
    # ★Removing them here makes an unknown-arch lookup fail; it does NOT relax a cap.
    # Contrast tpu_wrapper.sh's `gb200) echo "20"`, which is a limit-PRICE and whose
    # deletion would leave the family UNCAPPED -- that one stays.
}

# arch -> card codes in market.json price keys (first present wins for price).
ARCH_CARDS: dict[str, list[int]] = {
    'v4': [34],
    'v5p': [59],
    'v6e': [76, 63],
    'v6p': [92],
    'v7': [101],
    'a100': [46], 'a100_80gib': [66], 'h100': [70], 'h200': [86],
    'b200': [87], 'b300': [112],   # gb200/gb300 withdrawn: see above
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

# Cell -> metro resolution lives in the dependency-free ``metro_util`` leaf,
# itself a facade over the MEASURED ``cell_locality`` snapshot, so the --power
# router (``preflight.router``) and this smart-cell path agree on it exactly.
# Re-exported under the historical names for existing callers/tests.
_METRO_OVERRIDES = metro_util.METRO_OVERRIDES
metro_of = metro_util.metro_of

# The string a CellAvail carries when the cell's metro was never measured.
#
# WHY A STRING AND NOT THE SENTINEL. ``CellAvail.metro`` is typed ``str`` and
# ``route_lib.best_cell_for_shape`` calls ``.lower()`` on it unconditionally, so
# putting the sentinel object in the field would turn an unknown cell into an
# AttributeError deep inside the placement loop -- a crash, not a decision. This
# marker is a string that can never equal a real metro (metros are three lowercase
# letters), so an ``--metro`` allow-list DROPS the cell, which is the fail-closed
# direction: an unmeasurable cell is never silently treated as in-metro.
#
# It is deliberately NOT '' -- an empty metro would read as "no constraint" to a
# future filter written the other way round, and '' is what a missing field looks
# like. This value announces itself in any log line that prints it.
UNMEASURED_METRO = '__unmeasured__'


def metro_str(cell: str) -> str:
  """``metro_of`` coerced to a str for ``CellAvail.metro``, never guessing.

  Unknown -> ``UNMEASURED_METRO``, which no allow-list can match.
  """
  m = metro_of(cell)
  return UNMEASURED_METRO if m is metro_util.UNKNOWN else str(m)


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


def load_cell_prices(market_json_path: str = DEFAULT_MARKET_JSON,
                     pool: str = DEFAULT_PRICE_POOL) -> dict[str, dict[str, float]]:
  """arch -> {cell -> credits/chip-hr}, the PER-CELL prices in the same layer.

  ★THE PRICES WERE ALWAYS THERE; the router just never read them. Each market
  layer holds `global` PLUS one entry per cell, and inside one arch they differ
  by up to 3.2x -- v6e measured 15.999 (x102 cells) and 51.923 (x13 cells) in
  the same snapshot, v6p 14.404/28.261, v5p 15.475/33.692. `load_prices` takes
  only `global`, so every cell of an arch reached the router with an identical
  price and the cell-level sort had nothing to rank on.

  Invalid entries are DROPPED, not defaulted:
    * `None`   -- GQM quotes no price for that cell (v7's yuphxrp). A cell with
                  no price is one we cannot cost, and guessing the global value
                  for it is how a cell you cannot actually get ends up looking
                  like the cheapest option.
    * non-numeric -- same reasoning.
  A price of 0.0 is KEPT: a free pool is a real state (v4/v6e whole layers sit
  at 0.0), not missing data.

  Returns {} if the cache is missing; callers fall back to the global price,
  i.e. exactly today's behaviour.
  """
  try:
    with open(market_json_path) as f:
      market = json.load(f)
  except (OSError, ValueError):
    return {}
  prices = market.get('prices', {})
  out: dict[str, dict[str, float]] = {}
  for arch, cards in ARCH_CARDS.items():
    for card in cards:
      layer = prices.get(f'{pool}|{card}|PROD')
      if not isinstance(layer, dict) or 'global' not in layer:
        continue
      per_cell: dict[str, float] = {}
      for cell, val in layer.items():
        if cell == 'global' or not isinstance(val, (int, float)):
          continue          # drops None and any non-numeric quote
        per_cell[cell] = float(val)
      if per_cell:
        out[arch] = per_cell
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
    cell_price: Optional[dict[str, dict[str, float]]] = None,
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

  `cell_price[arch][cell]` (from `load_cell_prices`) gives each CellAvail its
  OWN price. A cell missing from that map falls back to the arch's global price
  -- which is the pre-2026-08-31 behaviour for every cell, so an absent or
  stale market cache degrades to exactly what the router did before.
  """
  avail_by_cell: dict[str, route_lib.CellAvail] = {}
  arch_pool: dict[str, float] = {}
  cell_price = cell_price or {}
  for arch, cells in per_arch.items():
    pool = 0
    price = arch_price.get(arch)
    by_cell = cell_price.get(arch, {})
    for cell, (free_chips, oversold) in cells.items():
      pool += max(0, free_chips)
      avail_by_cell[f'{cell}|{arch}'] = route_lib.CellAvail(
          cell=cell, arch=arch, free_chips=free_chips, oversold=oversold,
          price=by_cell.get(cell, price), metro=metro_str(cell))
    arch_pool[arch] = float(pool)
  return avail_by_cell, arch_price, arch_pool


# --- half-initialised module recovery ---------------------------------------
# ★A transient gRPC failure inside a LAZY import pins a long-lived worker
# FOREVER, and it disguises itself as "waiting for capacity": the router logs
# `availability fetch failed`, keeps the job QUEUED, never increments attempts,
# never bills, and the fleet stalls with nothing marked broken.
#
# MECHANISM, measured against the live RPC (not inferred):
# A generated `*_pb_stubby.py` sets, at MODULE level,
#     try:  _client_stub_base_class = proto_python_api_2_stub.Stub
#     except ImportError: _client_stub_base_class = object
# When the RPC stack is half-imported, that line raises AttributeError, which
# the `except ImportError` does NOT catch -- so the stubby module itself dies
# mid-body and every later call raises
#     NameError: name '_client_stub_base_class' is not defined
# Python never re-runs an import that is already in sys.modules, so the process
# is poisoned for good.
#
# ★THE FIX IS RELOAD-IN-PLACE, NOT EVICTION. Dropping the modules from
# sys.modules and re-importing MEASURABLY DOES NOT WORK: the cached
# `*_pb2.GoodputService` class holds `__globals__` pointing at the OLD module
# dict, so a fresh import builds a second dict nobody references and the stale
# class keeps raising. `importlib.reload()` re-executes the body in the SAME
# dict, which is the one the cached class reads. Verified end to end against
# blade:xborg-prod-routing-layer: poison 711 attrs -> evict+reimport still
# NameError -> reload-in-place recovers 206 cells.
#
# ★THE DISCRIMINATOR: AttributeError or NameError => the module EXISTS but is
#   INCOMPLETE, i.e. a poisoned PROCESS, not a version mismatch -- do NOT go
#   chasing library versions. An ImportError means the dependency is genuinely
#   absent and a reload cannot help; let it propagate.
# Corroborating signal: the first failure differs from every later one, and a
# FRESH process succeeds.
_HALF_INIT_MODULE_HINTS = (
    'stubby',
    'grpc',
    'rpc',
    'net.rpc',
)


_HALF_INIT_RE = re.compile(r"module '([A-Za-z0-9_.]+)' has no attribute")
# The stubby module's own body died partway, so a module-level name it was
# supposed to bind is missing. This is the shape actually observed live.
_HALF_INIT_NAME_RE = re.compile(r"name '([A-Za-z0-9_]+)' is not defined")


def _looks_half_initialised(exc: BaseException) -> bool:
  """True iff `exc` is the 'module exists but is incomplete' shape.

  ONLY AttributeError (attribute never populated) and NameError (module body
  died before binding a module-level name) qualify. An ImportError means the
  module is genuinely absent -- reloading and retrying would just burn a second
  RPC deadline and hide the real error.
  """
  if isinstance(exc, ImportError):  # ModuleNotFoundError included
    return False
  if isinstance(exc, NameError):
    return _HALF_INIT_NAME_RE.search(str(exc)) is not None
  if not isinstance(exc, AttributeError):
    return False
  return _poisoned_module_name(exc) is not None


def _poisoned_module_name(exc: BaseException) -> Optional[str]:
  """The module named by an AttributeError, e.g. `...base_stubby_api`, or None."""
  m = _HALF_INIT_RE.search(str(exc))
  return m.group(1) if m else None


def _is_generated_proto(name: str) -> bool:
  """Generated `_pb2` protos are NEVER reloaded: re-executing one duplicates
  descriptor-pool entries and raises, turning a recoverable stall into a hard
  crash. Their `_pb_stubby` siblings are safe and ARE reloaded -- that is where
  the poison actually sits."""
  return '_pb2' in name.lower()


def _reload_candidates(seed: Optional[str]) -> list[str]:
  """Live RPC-stack modules to re-execute in place.

  Wider than just the module named in the message, because a module that DID
  finish importing still holds a reference to the broken one:
    1. the seed module named by the error, plus its submodules;
    2. every RPC-stack module (`_HALF_INIT_MODULE_HINTS`);
  both excluding generated protos and `None` tombstones (a tombstone has no
  module object to reload; it is dropped separately).
  """
  victims: set[str] = set()
  for name, mod in list(sys.modules.items()):
    if mod is None or _is_generated_proto(name):
      continue
    if seed and (name == seed or name.startswith(seed + '.')):
      victims.add(name)
      continue
    lowered = name.lower()
    if any(h in lowered for h in _HALF_INIT_MODULE_HINTS):
      victims.add(name)
  return sorted(victims)


def _tombstone_names() -> list[str]:
  """`None` entries left in sys.modules by a failed import; safe to drop."""
  return sorted(n for n, m in list(sys.modules.items()) if m is None)


def _heal_half_initialised_modules(seed: Optional[str] = None) -> list[str]:
  """Re-execute poisoned RPC modules IN PLACE. Returns the names healed.

  In place (`importlib.reload`) rather than pop+reimport: the cached generated
  service class reads the ORIGINAL module dict via `__globals__`, so a fresh
  module object would leave it reading the stale one. Scoped to the RPC stack
  on purpose -- reloading the world under a live worker is not recoverable.
  """
  healed: list[str] = []
  for name in _tombstone_names():
    sys.modules.pop(name, None)
    healed.append(name)
  for name in _reload_candidates(seed):
    mod = sys.modules.get(name)
    if mod is None:
      continue
    try:
      importlib.reload(mod)
      healed.append(name)
    except Exception:  # pylint: disable=broad-except
      # A module that refuses to reload is not fatal: the others may still
      # restore the stack, and the retry will show whether it worked.
      pass
  return healed


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
    """One RPC per arch -> the router's (avail_by_cell, arch_price, arch_pool).

    Retries ONCE after re-executing half-initialised modules in place (see
    `_looks_half_initialised`): a transient gRPC failure during a lazy import
    otherwise pins a long-lived worker forever.
    """
    try:
      return self._fetch_once()
    except Exception as e:  # pylint: disable=broad-except
      if not _looks_half_initialised(e):
        raise
      healed = _heal_half_initialised_modules(_poisoned_module_name(e))
      print(f'[avail_provider] half-initialised module detected '
            f'({type(e).__name__}: {e}); reloaded {len(healed)} module(s) in '
            f'place and retrying once: {healed[:8]}', flush=True)
      if not healed:
        raise
      return self._fetch_once()

  def _fetch_once(self) -> tuple[dict[str, route_lib.CellAvail], dict[str, float], dict[str, float]]:
    """The unguarded fetch. One RPC per arch."""
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
    # Two reads of the same file, deliberately: the ARCH score wants one price
    # per arch, the CELL score wants each cell's own. Conflating them is what
    # made the cell-level sort price-blind.
    cell_price = load_cell_prices(self.market_json_path)
    return build_availability(per_arch, arch_price, cell_price)

