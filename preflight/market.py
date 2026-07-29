"""L3 check: GQM market data (per-cell clearing prices + limit orders).

WHY this module exists
----------------------
Preflight's L1/L2 answer "is this slice shape legal" and "do enough chips
exist".  Neither question is the one that actually strands jobs on a dynamic
(GQM) pool.  The real #1 hard blocker is a **limit order**: a per
``(ResourcePool, Mdb, XManagerExperimentId, ScuId, ResourceType, Priority)``
price cap.  When the auction clears ABOVE the cap the SCU is moved into
``BUCKET_ID_TRIGGERED_LIMIT_ORDER``, which bypasses the main scheduling
process -- the job is pulled from the queue *before* any capacity check runs.
Free floor and idle chips do not help.  A router that only knows about quota
and obtainable chips will therefore cheerfully recommend a combination that is
100% blocked.

WHICH price the cap is compared against (this one is counter-intuitive)
-----------------------------------------------------------------------
GQM stores prices per ``(resource_pool, resource_type, cell, priority)``, and
the per-cell spread is real: v6e PROD cleared at 22.23 credits/chip-hour in 117
cells and 48.93 in nine others on 2026-07-29, inside one pool.  It is therefore
very tempting to conclude that pinning a cheap cell can rescue a job whose
limit order has triggered.  **It cannot.**  Verified in source:

* Every production market cycle runs the V2 auction --
  ``cron_trigger/trigger_server.cc:222`` hardcodes ``set_run_v2_auction(true)``.
* The V2 path calls ``TransfromMarketSpecsToSingleCell`` BEFORE the auction
  (``cron_service/utils/quota_auction_utils.cc:438``), which rewrites the cell
  of every capacity spec, demand spec, bid and queued SCU to the synthetic
  constant ``kSingleGlobalCell`` = ``"single-global-layer"`` (``:132``,
  ``data/consts.h:45``).
* The trigger test then matches ``spec.cell() == scu_info.cell()``
  (``market_algorithm/limit_order.cc:265``) -- with both sides equal to that
  same synthetic constant.  **The cell you pinned never enters the comparison.**
* Post-auction the synthetic cell is renamed to ``global``
  (``quota_auction_utils.cc:236``), which is what lands in Spanner.

So the ``Cell='global'`` row is not a convenience aggregate -- it *is* the
number the limit order is compared against.  Corroboration: the ``LimitOrders``
table has no Cell column at all, so a per-cell cap is not even expressible.

This module therefore keeps both, for two different jobs:

* the **pool-level (global) price** decides BLOCKED -- see ``is_blocked``;
* the **per-cell price** decides COST, because charging really does read the
  per-cell hourly rows (``mdb_charges_utils.cc:55-65``).  A cheap cell saves
  credits; it just cannot unblock a triggered limit order.

The POOL dimension is not cosmetic either: the same (cell, type, tier) cleared
at 20.20 in ``deepmind-dynamic-pool`` and 32.89 in ``gemini-dynamic-pool`` at
the same instant, and ``LimitOrders`` is keyed by ``ResourcePool``.  Collapsing
pools would import another team's market into your decision.

WHY a JSON cache instead of querying Spanner here
-------------------------------------------------
``money_check.py`` already fetches exactly this data every daemon round, and a
fresh Spanner round-trip inside ``tpu route`` would add ~10 s to an interactive
command.  ``money_check`` therefore also dumps the structured data to
``~/.tpu_quota_cache_dir/market.json`` (built by ``build_payload``, written by
``write_snapshot``); this module only reads it.  The rendered ``money.txt`` is
NOT a usable source: it keeps four sample cells per card, which cannot support
a per-cell routing decision.

Everything here degrades gracefully.  If the daemon is not running the JSON is
missing, ``load_snapshot`` returns an empty snapshot carrying a warning, and
the router must fall back to price-blind behaviour rather than crash -- a
missing market file is an ordinary state, not an error.  It must not be silent
either, or the router quietly goes back to recommending blocked combinations.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from typing import Any, Optional

from google3.borg.common import scalar_resource_pb2
from google3.experimental.users.qiaos.tpu_utils.preflight import topology

# Written by money_check.py, read by router.py. Lives next to the other
# ~/.tpu_quota_cache_dir artifacts the daemon maintains.
CACHE_DIR = os.path.expanduser('~/.tpu_quota_cache_dir')
MARKET_JSON_NAME = 'market.json'
MARKET_JSON_PATH = os.path.join(CACHE_DIR, MARKET_JSON_NAME)

# Bumped whenever the on-disk shape changes incompatibly. A reader that sees an
# unknown version treats the file as missing rather than mis-parsing it.
SCHEMA_VERSION = 2

# Match the 180 s freshness threshold tpu_wrapper.sh already applies to the
# other cache files, so "stale" means the same thing everywhere.
DEFAULT_MAX_AGE_S = 180.0

_INT64_MAX = 9223372036854775807
_INT64_MIN = -9223372036854775808

# Tier strings as they appear in the GQM Spanner tables' Priority column.
PROD = 'PROD'
BATCH = 'BATCH'
KNOWN_TIERS = (PROD, BATCH)

# The synthetic pool-level row. Post-auction rename of 'single-global-layer'
# (quota_auction_utils.cc:236). This is the price a limit order is actually
# compared against, and it must never be offered as a schedulable cell.
GLOBAL_CELL = 'global'


def decode_price(milli_credits: int) -> Optional[float]:
  """Decode ``ResourcePrices.MilliCreditsPerUnitHour`` into credits/chip-hour.

  GQM encodes two non-numeric states in this column, and collapsing them loses
  real signal:

  * ``INT64_MAX``           -- the resource is UNOBTAINABLE in that cell this
                               cycle.  Returned as ``None``.
  * ``INT64_MIN`` (or any
    negative value)         -- no bid was recorded; GQM itself sanitizes this
                               to zero, so we do too.
  * ``0``                   -- a genuine free-pool clearing price: supply met
                               demand and the auction cleared at zero cost.

  The last two both surface as ``0.0``; that collapse is GQM's, not ours.  The
  distinction this function protects is ``None`` (no chips at any price) versus
  ``0.0`` (chips for free) -- the one that changes a routing decision.
  """
  if milli_credits == _INT64_MAX:
    return None
  if milli_credits == _INT64_MIN or milli_credits < 0:
    return 0.0
  return float(milli_credits) / 1000.0


def arch_resource_type(arch: str) -> Optional[int]:
  """Marketing arch name ('v6e') -> ``borg.ScalarResource.Key`` int (76).

  Resolved through ``topology.BORG_PLATFORM_KEY`` -- the wrapper's single owner
  of the arch -> Borg platform mapping, and the same table ``capacity.py`` uses
  for GetCellAvailability -- and then through the proto enum, so no integer is
  hardcoded here and the market join can never disagree with the capacity join.

  That matters: the pod and non-pod SKUs of one TPU generation are *different*
  resource types with different markets.  ``GHOSTLITE_POD`` (76) is v6e, while
  ``GHOSTLITE`` (63) is a separate SKU with no rows at all in
  ``deepmind-dynamic-pool``; hardcoding 63 for v6e yields a silent "no price
  data" instead of the real blocker.
  """
  key_name = topology.BORG_PLATFORM_KEY.get(arch.lower())
  if not key_name:
    return None
  try:
    return int(scalar_resource_pb2.ScalarResource.Key.Value(key_name))
  except ValueError:
    return None


@dataclasses.dataclass(frozen=True)
class LimitOrder:
  """One MDB-level price cap, in credits/chip-hour."""
  cap: float
  user: str          # LimitOrderUser: who set it. Often a teammate, not you.
  mdb: str
  resource_type: int
  tier: str
  pool: str = ''

  def blocks(self, price: Optional[float]) -> bool:
    return is_blocked(price, self.cap)

  def as_tuple(self) -> tuple[float, str]:
    """The ``(cap, user)`` pair, for callers that only need those two."""
    return (self.cap, self.user)


@dataclasses.dataclass(frozen=True)
class MarketSnapshot:
  """Per-cell prices + limit orders as of one daemon round.

  ``prices`` is ``{(pool, resource_type_int, tier): {cell: price_or_None}}``
  where a ``None`` price means UNOBTAINABLE this cycle (see ``decode_price``).
  ``limit_orders`` is ``{(pool, mdb, resource_type_int, tier): LimitOrder}``.
  Both are plain dicts so the whole snapshot stays trivially picklable and
  cheap to pass to worker threads.
  """
  prices: dict[tuple[str, int, str], dict[str, Optional[float]]] = (
      dataclasses.field(default_factory=dict))
  limit_orders: dict[tuple[str, str, int, str], LimitOrder] = (
      dataclasses.field(default_factory=dict))
  generated_unix: float = 0.0
  pools: tuple[str, ...] = ()
  # Non-empty when the data could not be used as-is; the CLI prints it verbatim
  # so a price-blind run is never silent.
  warning: str = ''

  @property
  def available(self) -> bool:
    return bool(self.prices) and self.generated_unix > 0

  @property
  def age_seconds(self) -> Optional[float]:
    if self.generated_unix <= 0:
      return None
    return max(0.0, time.time() - self.generated_unix)

  def is_stale(self, max_age_s: float = DEFAULT_MAX_AGE_S) -> bool:
    age = self.age_seconds
    return age is None or age > max_age_s

  def age_str(self) -> str:
    """Short human rendering of the data's age, for CLI headers."""
    age = self.age_seconds
    if age is None:
      return 'n/a'
    if age < 90:
      return f'{age:.0f}s'
    if age < 5400:
      return f'{age / 60:.0f}m'
    return f'{age / 3600:.1f}h'

  # ---- lookups ----------------------------------------------------------

  def get_prices(self, resource_type: int, tier: str,
                 pool: Optional[str] = None,
                 include_global: bool = False
                 ) -> dict[str, Optional[float]]:
    """Cell -> clearing price for one (resource type, tier). ``{}`` if unknown.

    The synthetic ``global`` row is excluded by default: it is a pool-level
    aggregate, not a place a job can be pinned, and letting it into a per-cell
    map would make it look like a (usually very cheap) schedulable cell.  Use
    ``get_pool_price`` to read it deliberately.

    ``pool=None`` merges every pool in the snapshot, keeping the cheapest quote
    per cell.  That is only correct when the caller genuinely has access to all
    of them; routing code should pass the alloc's own ``CapacityResult.pool``
    so another team's market cannot leak in.
    """
    rtype = int(resource_type)
    tier = tier.upper()
    if pool is not None:
      cells = self.prices.get((pool, rtype, tier), {})
      return {c: p for c, p in cells.items()
              if include_global or c != GLOBAL_CELL}
    merged: dict[str, Optional[float]] = {}
    for (_p, r, t), cells in self.prices.items():
      if r != rtype or t != tier:
        continue
      for cell, price in cells.items():
        if not include_global and cell == GLOBAL_CELL:
          continue
        if cell not in merged or merged[cell] is None:
          merged[cell] = price
        elif price is not None:
          merged[cell] = min(merged[cell], price)  # type: ignore[type-var]
    return merged

  def get_pool_price(self, resource_type: int, tier: str,
                     pool: Optional[str] = None) -> Optional[float]:
    """The pool-level (``Cell='global'``) clearing price -- the one that blocks.

    This is the value the GQM limit-order trigger compares against, because the
    V2 auction collapses every cell into one synthetic layer before evaluating
    caps (see the module docstring).  Returns ``None`` when there is no global
    row, or when it carries the UNOBTAINABLE sentinel.

    Falls back to the MAXIMUM per-cell price if the global row is missing.  Max,
    not min: with no pool-level number to compare against, over-reporting
    "blocked" costs the user one extra ``--explain``, while under-reporting it
    sends them to submit a job that will be silently pulled from the queue.
    """
    with_global = self.get_prices(resource_type, tier, pool,
                                  include_global=True)
    if GLOBAL_CELL in with_global:
      return with_global[GLOBAL_CELL]
    real = [p for c, p in with_global.items()
            if c != GLOBAL_CELL and p is not None]
    return max(real) if real else None

  def get_pool_price_for_arch(self, arch: str, tier: str,
                              pool: Optional[str] = None) -> Optional[float]:
    """``get_pool_price`` keyed by the marketing arch name ('v6e')."""
    rtype = arch_resource_type(arch)
    if rtype is None:
      return None
    return self.get_pool_price(rtype, tier, pool)

  def get_prices_for_arch(self, arch: str, tier: str,
                          pool: Optional[str] = None
                          ) -> dict[str, Optional[float]]:
    """``get_prices`` keyed by the marketing arch name ('v6e')."""
    rtype = arch_resource_type(arch)
    if rtype is None:
      return {}
    return self.get_prices(rtype, tier, pool)

  def get_limit_order(self, mdb_or_alloc: str, resource_type: int, tier: str,
                      pool: Optional[str] = None) -> Optional[LimitOrder]:
    """MDB-level cap for one (resource type, tier), or ``None`` if unset.

    Accepts either a bare MDB name or a full ``group:pool/mdb`` alloc string.
    Only MDB-scoped rows are modelled: per-XID and per-SCU orders are strictly
    more specific (resolution order is SCU > XID > MDB) but cannot exist before
    the job is submitted, which is exactly when this router runs.

    ``None`` genuinely means "no cap", not "unknown": with no row there is no
    ``milli_credit_limit_price``, so the reason can never fire.

    When several caps match (``pool=None``, or a wildcard row written without a
    pool) the LOWEST cap wins -- the one most likely to block, which is the
    safe direction for a pre-submit check.
    """
    mdb = mdb_or_alloc.rstrip('/').split('/')[-1]
    rtype = int(resource_type)
    tier = tier.upper()
    hits = [lo for (p, m, r, t), lo in self.limit_orders.items()
            if m == mdb and r == rtype and t == tier
            # '' is the wildcard pool: a row whose pool the writer could not
            # determine. Assume it applies rather than ignoring a real cap.
            and (pool is None or p == pool or not p)]
    if not hits:
      return None
    return min(hits, key=lambda lo: lo.cap)

  def get_limit_order_for_arch(self, mdb_or_alloc: str, arch: str, tier: str,
                               pool: Optional[str] = None
                               ) -> Optional[LimitOrder]:
    """``get_limit_order`` keyed by the marketing arch name ('v6e')."""
    rtype = arch_resource_type(arch)
    if rtype is None:
      return None
    return self.get_limit_order(mdb_or_alloc, rtype, tier, pool)


def is_blocked(price: Optional[float], cap: Optional[float]) -> bool:
  """True when a limit order would strand a job at this clearing price.

  * ``cap is None``   -> no limit-order row exists, so the reason can never
                         fire.  Never blocked.
  * ``price is None`` -> UNOBTAINABLE this cycle.  That is a *capacity*
                         problem, not a limit-order one; calling it "blocked by
                         your price cap" would send the user to fix the wrong
                         thing.  Not blocked here.
  * otherwise         -> blocked iff ``price > cap``.  Strict: clearing exactly
                         at the cap is still affordable.
  """
  if cap is None or price is None:
    return False
  return price > cap


# --------------------------------------------------------------------------
# Serialization. money_check.py owns the fetch; this module owns the format, so
# writer and reader cannot drift.
# --------------------------------------------------------------------------


def build_payload(
    prices_by_key: dict[
        tuple[int, str], list[tuple[str, Optional[float], str]]],
    limit_orders: dict[tuple[str, int, str], tuple[int, str]],
    pools: Optional[set[str]] = None,
    mdbs: Optional[set[str]] = None,
    limit_order_pools: Optional[dict[tuple[str, int, str], str]] = None,
    generated_unix: Optional[float] = None,
) -> dict[str, Any]:
  """Build the on-disk payload from money_check's already-fetched structures.

  Args:
    prices_by_key: exactly ``money_check.fetch_prices_from_spanner`` output --
      ``{(resource_type_int, tier): [(cell, price_or_None, pool)]}``, with the
      sentinel decoding already applied by ``decode_price``.  The per-entry
      pool is what re-establishes the pool dimension here.
    limit_orders: exactly ``money_check.fetch_limit_orders`` output --
      ``{(mdb, resource_type_int, tier): (milli_credits, user)}``.
    pools: pools the caller participates in, recorded for provenance.
    mdbs: if given, limit orders are filtered to these MDBs.  The global table
      carries hundreds of other teams' rows that cannot affect our routing.
    limit_order_pools: optional ``{key: ResourcePool}`` companion map for
      ``limit_orders``.  Absent, rows are recorded under the wildcard pool
      ``''`` and matched in any pool -- deliberately conservative, since a cap
      we cannot place is more safely assumed to apply than to be ignored.
    generated_unix: override for tests; defaults to now.

  Returns a JSON-serializable dict.  Prices are emitted as ``null`` when
  UNOBTAINABLE so the three-way distinction survives the round-trip.
  """
  price_out: dict[str, dict[str, Optional[float]]] = {}
  for (rtype, tier), entries in prices_by_key.items():
    if tier not in KNOWN_TIERS:
      continue
    for cell, price, pool in entries:
      cells = price_out.setdefault(f'{pool}|{int(rtype)}|{tier}', {})
      prev = cells.get(cell, ...)
      if prev is ... or prev is None:
        # First quote wins; any real quote beats the UNOBTAINABLE sentinel,
        # since keeping None would hide a usable cell.
        cells[cell] = price
      elif price is not None:
        cells[cell] = min(prev, price)

  lo_out: dict[str, dict[str, Any]] = {}
  for key, value in limit_orders.items():
    mdb, rtype, tier = key
    if mdbs is not None and mdb not in mdbs:
      continue
    if tier not in KNOWN_TIERS:
      continue
    milli, user = value
    pool = (limit_order_pools or {}).get(key, '')
    lo_out[f'{pool}|{mdb}|{int(rtype)}|{tier}'] = {
        'cap': float(milli) / 1000.0,
        'user': user or '',
    }

  return {
      'schema_version': SCHEMA_VERSION,
      'generated_unix': float(generated_unix
                              if generated_unix is not None else time.time()),
      'pools': sorted(pools or ()),
      'prices': price_out,
      'limit_orders': lo_out,
  }


def write_snapshot(payload: dict[str, Any],
                   path: str = MARKET_JSON_PATH) -> None:
  """Atomically write the payload. Called by money_check on every round.

  Temp file + rename, because ``tpu route`` may read this file at any instant
  and a half-written JSON would drop the router into price-blind mode for no
  reason.
  """
  os.makedirs(os.path.dirname(path), exist_ok=True)
  tmp = f'{path}.tmp.{os.getpid()}'
  with open(tmp, 'w', encoding='utf-8') as f:
    json.dump(payload, f, separators=(',', ':'), sort_keys=True)
  os.replace(tmp, path)


def _empty(warning: str) -> MarketSnapshot:
  return MarketSnapshot(warning=warning)


def load_snapshot(path: str = MARKET_JSON_PATH) -> MarketSnapshot:
  """Read the market cache. Never raises; returns an empty snapshot instead.

  A missing file is the normal state when the background daemon is not running,
  so it must not be an error -- but the reason is always carried in
  ``MarketSnapshot.warning`` so the CLI can say why it went price-blind.
  """
  try:
    with open(path, 'r', encoding='utf-8') as f:
      raw = json.load(f)
  except FileNotFoundError:
    return _empty(f'no market data at {path}; is the tpu-daemon running? '
                  'Routing without prices or limit orders.')
  except (OSError, ValueError) as e:
    return _empty(f'market data at {path} is unreadable ({type(e).__name__}: '
                  f'{e}); routing without prices or limit orders.')

  if not isinstance(raw, dict):
    return _empty(f'market data at {path} is not a JSON object; ignoring.')

  version = raw.get('schema_version')
  if version != SCHEMA_VERSION:
    return _empty(
        f'market data at {path} has schema_version={version!r}, expected '
        f'{SCHEMA_VERSION}; rebuild + rerun money_check. Routing without '
        'prices.')

  prices: dict[tuple[str, int, str], dict[str, Optional[float]]] = {}
  for key, cells in (raw.get('prices') or {}).items():
    parts = str(key).split('|')
    if len(parts) != 3 or not isinstance(cells, dict):
      continue
    pool, rtype_s, tier = parts
    try:
      rtype = int(rtype_s)
    except ValueError:
      continue
    prices[(pool, rtype, tier.upper())] = {
        str(cell): (None if p is None else float(p))
        for cell, p in cells.items()
    }

  limit_orders: dict[tuple[str, str, int, str], LimitOrder] = {}
  for key, value in (raw.get('limit_orders') or {}).items():
    parts = str(key).split('|')
    if len(parts) != 4 or not isinstance(value, dict):
      continue
    pool, mdb, rtype_s, tier = parts
    raw_cap = value.get('cap')
    if raw_cap is None:
      continue
    try:
      rtype = int(rtype_s)
      cap = float(raw_cap)
    except (TypeError, ValueError):
      continue
    tier = tier.upper()
    limit_orders[(pool, mdb, rtype, tier)] = LimitOrder(
        cap=cap, user=str(value.get('user') or ''), mdb=mdb,
        resource_type=rtype, tier=tier, pool=pool)

  snap = MarketSnapshot(
      prices=prices,
      limit_orders=limit_orders,
      generated_unix=float(raw.get('generated_unix') or 0.0),
      pools=tuple(raw.get('pools') or ()),
  )
  if not snap.prices:
    return _empty(f'market data at {path} contains no price rows; ignoring.')
  return snap
