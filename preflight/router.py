"""Local TPU router: from a desired power class to a (group, tpu_type, CELL).

Power classes are compute-equivalence buckets, derived from Borg's published
per-chip `vle` compute unit (see ``_V5P_MULTIPLIER`` below for the table and
its two independent derivations):
  1 v6p chip = 1 v7 chip ~ 4.34 v5p chips ~ 2.17 v6e chips
  => v6p-8 ~ v7-8 ~ v6e-16 ~ v5p-32 (roughly)

We express a "power class" as an integer in a canonical unit (v5p-chip units).
The user either names a concrete type (``--power=v5p-32``) or a numeric class
(``--power=32``, meaning "32 v5p-equivalent chips").

This is a COMPUTE ratio. It does not predict memory-bound throughput, where the
ordering can even invert (v6e has less HBM bandwidth than v5p). ``_V5P_MULTIPLIER``
documents the caveats; surface them to the user rather than treating the score
as a universal speed number.

WHY there is a cell dimension
-----------------------------
Admission on a dynamic (GQM) pool is an ordered multi-pass pipeline, not a
single quota test. The pass that kills jobs first is the limit order: a price
cap per (pool, mdb, resource_type, tier). When the auction clears above the cap
the SCU is dropped before it is ever bucketized, so neither free floor nor idle
chips help.

The cell matters for COST, not for unblocking. GQM stores prices per cell and
the spread is real (v6e PROD: 22.23 credits/chip-hour in 117 cells, 48.93 in
nine others, same pool, same minute), and charging reads those per-cell rows --
so pinning the cheap cell genuinely halves the bill. But it CANNOT rescue a
triggered limit order: the V2 auction collapses every cell into one synthetic
layer before evaluating caps, so the cap is compared against the pool-level
price and the cell a job pinned never enters the comparison. See the
``preflight.market`` module docstring for the full chain of source citations.

That asymmetry is exactly why blocking and cell selection are computed from two
different numbers below, and why the CLI must not advise "try a cheaper cell"
as a fix for a triggered cap.

Algorithm:
  1. Expand the power class -> concrete (arch, chips) options.
  2. Cross with the allowed groups (default: every group in GROUP_MAP).
  3. Run preflight (L1 topology + L2 per-cell capacity) on each; drop RED.
  4. Decide BLOCKED from the pool-level price vs the alloc's limit order.
  5. Join `cap.cells_ok` against per-cell prices to pick the cheapest cell and
     cost the run. A blocked combo is KEPT, flagged, with a reason -- silence
     would turn "your price cap blocks everything" into "no results", which is
     the exact confusion this router exists to remove.
  6. Rank and return top-K.
"""

import concurrent.futures
import dataclasses
from typing import Optional

from google3.experimental.users.qiaos.tpu_utils import metro_util
from google3.experimental.users.qiaos.tpu_utils.preflight import market
from google3.experimental.users.qiaos.tpu_utils.preflight import preflight

# Compute equivalence: (arch, chips) -> canonical v5p-chip units.
#
# Source of truth is the `vle` (v5e-equivalent) field that Borg publishes per
# accelerator in //borg/util/reports/gxu/gxus_by_platform_ga.textproto. It is
# the officially normalized per-chip bf16 compute unit; dividing every entry by
# v5p's 2.33 gives the numbers below. Cross-checked against the per-chip MXU
# bf16 FLOPs in the ART system models
# (//platforms/deepsea/ffds/art/performance/systems/configs/*.textproto):
#
#   arch  chip           vle     MXU bf16   vle/2.33   flops/459e12
#   v5e   viperlite      1.00     197 TF      0.43        0.43
#   v4    pufferfish     1.40     275 TF      0.60        0.60
#   v5p   viperfish      2.33     459 TF      1.00        1.00
#   v6e   ghostlite_pod  4.66     918 TF      2.00        2.00
#   v6p   ghostfish     10.11    1992 TF      4.34        4.34
#   v7    ghostfishlite 10.11    1992 TF      4.34        4.34
#
# The two derivations agree to three digits, so these are real ratios rather
# than a guess. NOTE the previous table claimed v6p = 2.0 and v4 = 1.0; both
# understated the spread badly (v6p by >2x), which made `tpu route --power=`
# recommend roughly twice the v6p/v7 hardware a request actually needs.
#
# Caveats this single scalar cannot express, so read them before trusting it:
#  - It is a *compute* ratio only. HBM bandwidth does not track it: v6e has
#    1.61 TB/s vs v5p's 2.77 TB/s, so despite scoring 2x on compute a v6e chip
#    is SLOWER than v5p on memory-bound work (long-context attention, small
#    batch decode). v6p/v7 are 7.37 TB/s, i.e. 2.7x v5p, well under their 4.34x
#    compute ratio -- memory-bound jobs will not see the full speedup either.
#  - Low-precision paths differ: v4/v5e/v5p/v6e accelerate int8 (2x) and int4
#    (4x); v6p/v7 accelerate fp8 (2x) and give int8 NO speedup at all (1x). An
#    int8-tuned model ported from v5p to v6p/v7 must move to fp8 to gain.
_V5P_MULTIPLIER: dict[str, float] = {
    'v5e': 0.43,
    'v4': 0.60,
    'v5p': 1.0,
    'v6e': 2.0,
    'v6p': 4.34,
    'v7': 4.34,     # ghostfishlite: same GFC chip as v6p, identical per-chip perf.
}

# Preference between equally-good architectures: newer first. v7 leads because
# it matches v6p chip-for-chip yet repeatedly clears at the 0.00 free-pool price
# when v6p has zero availability; it is capped at 32 chips in _LOCUS_TABLE, so
# it simply drops out of the candidate set for larger requests.
_ARCH_PREF: dict[str, int] = {
    'v7': 0, 'v6p': 1, 'v6e': 2, 'v5p': 3, 'v4': 4, 'v5e': 5}

# Preference between groups (allocs): which one to SPEND FIRST. Lower = preferred.
# g3 (gdm-viscam-interns-dynamic) and g5 (vqfree-xm) are small dynamic pools
# with their OWN credit balance and -- crucially -- NO share of the G9 income/10
# cap that the budget gate enforces. Spending them first therefore preserves the
# regulated G9 budget for when it is actually needed, at no extra credit cost.
# g9 holds the big floor but every chip-hour there is billed against that 1/10
# cap, so it is the LAST resort among otherwise-equal options, not the first.
#
# This is only a preference among candidates that are ALREADY equally runnable
# and equal-status (it sits below `blocked` and GREEN/YELLOW in the sort key),
# so it never routes a job somewhere it cannot actually run -- it only decides
# WHOSE budget to draw when more than one group could take the job. Groups not
# listed share the default and are chosen after g3/g5 but before nothing in
# particular; the remaining tie-breaks (headroom, cost, arch) still decide.
_GROUP_PREF: dict[int, int] = {
    3: 0,   # gdm-viscam-interns-dynamic -- own balance, exempt from G9 1/10 cap
    5: 0,   # vqfree-xm                  -- own balance, exempt from G9 1/10 cap
}
_GROUP_PREF_DEFAULT = 1   # everyone else, incl. g9 (billed against the 1/10 cap)


@dataclasses.dataclass(frozen=True)
class CellOffer:
  """One cell's price and obtainable capacity for a given (arch, chips, tier).

  ``price is None`` means the market reported the cell UNOBTAINABLE this cycle,
  or we simply have no quote for it. Those are different, and ``price_known``
  tells them apart: an unpriced cell is still usable (we just cannot cost it),
  whereas an unobtainable one should not be recommended.

  There is deliberately no per-cell ``blocked`` flag. Limit orders are decided
  at the pool level, so blocking is a property of the whole (group, arch, tier)
  combination -- it lives on ``Candidate``.
  """
  cell: str
  obtainable: int
  price: Optional[float]          # credits per chip-hour
  price_known: bool               # False = no quote at all for this cell
  unobtainable: bool              # True = quoted, but as INT64_MAX
  cost_per_hour: Optional[float]  # chips * price


@dataclasses.dataclass(frozen=True)
class Candidate:
  """A concrete (group, arch, chips, cell) offer with its preflight verdict."""
  group_id: int
  alloc: str
  arch: str
  chips: int
  tpu_type: str       # e.g. 'v6e-16'
  power_score: float  # in v5p-equivalent chips (higher = more compute)
  verdict: preflight.Verdict
  # --- cell + market dimension (empty/None when market data is unavailable) --
  cell: str = ''
  price: Optional[float] = None         # credits per chip-hour in `cell`
  cost_per_hour: Optional[float] = None # chips * price
  obtainable: int = 0                   # chips obtainable in `cell`
  # The pool-level ('global') price -- the number the limit order is actually
  # compared against. Usually differs from `price`, which is cell-specific.
  pool_price: Optional[float] = None
  blocked: bool = False                 # pool_price > cap
  block_reason: str = ''                # human-readable, only when blocked
  limit_order: Optional[market.LimitOrder] = None
  # Every cell we evaluated, cheapest first. Kept so --explain can show the
  # full picture rather than only the winner.
  offers: tuple[CellOffer, ...] = ()

  @property
  def remaining_quota(self) -> int:
    cap = self.verdict.capacity
    if not cap:
      return 0
    return max(0, cap.alloc_scoped_quota - cap.alloc_scoped_used)

  @property
  def headroom_ratio(self) -> float:
    """PROD quota headroom in units of the request. Meaningless for BATCH."""
    if self.chips <= 0:
      return 0.0
    return self.remaining_quota / self.chips

  @property
  def obtainable_ratio(self) -> float:
    if self.chips <= 0:
      return 0.0
    return self.obtainable / self.chips


def to_power(arch: str, chips: int) -> float:
  """Return the v5p-equivalent chip count for the given (arch, chips)."""
  return _V5P_MULTIPLIER.get(arch.lower(), 1.0) * float(chips)


def parse_power_input(power: str) -> float:
  """Parse a power spec like 'v5p-32', 'v6e-16', or bare number '32'.

  Returns power in v5p-equivalent chips.
  """
  s = power.strip().lower()
  # bare integer
  if s.isdigit():
    return float(int(s))
  # 'v6e-16', 'v5p-32', ...
  for sep in ('-', '='):
    if sep in s:
      arch, cores = s.split(sep, 1)
      try:
        return to_power(arch, int(cores))
      except (ValueError, KeyError):
        raise ValueError(f"power spec '{power}' has non-integer chip count")
  raise ValueError(f"Cannot parse power spec: {power!r}. "
                   f"Expected 'v6e-16', 'v5p-32', or a bare integer.")


def _candidate_options(target_power: float,
                       tolerance: float = 0.5
                       ) -> list[tuple[str, int]]:
  """Enumerate (arch, chips) options whose power is within tolerance of target.

  tolerance=0.5 means we accept anything with power in [0.75*target, 1.5*target].
  """
  low = target_power * (1.0 - tolerance / 2.0)
  high = target_power * (1.0 + tolerance / 2.0)
  # For each arch, iterate legal sizes and keep those in range.
  from google3.experimental.users.qiaos.tpu_utils.preflight import topology
  out: list[tuple[str, int]] = []
  for arch in ['v7', 'v6p', 'v6e', 'v5p', 'v4', 'v5e']:
    for chips in topology.legal_sizes_for(arch):
      p = to_power(arch, chips)
      if low <= p <= high:
        out.append((arch, chips))
  return out


def _evaluate_cells(arch: str, chips: int, tier: str,
                    verdict: preflight.Verdict,
                    snapshot: market.MarketSnapshot,
                    metros: Optional[list[str]] = None) -> list[CellOffer]:
  """Join this combo's viable cells against the market, cheapest first.

  Only ``cap.cells_ok`` is considered: those are the cells that already hold
  enough obtainable chips, so a price on any other cell is not actionable.

  ``metros`` (data-locality) is a HARD filter applied FIRST: a cell whose metro
  is not in the allow-list is dropped before pricing, so the router can only
  ever recommend an in-metro cell. When it empties the list the whole combo
  yields no offer and drops out of the ranking -- that is what makes ``--power
  --metro`` fail closed instead of silently roaming to a no-data cell. Metro is
  resolved by the shared ``metro_util`` leaf, the same mapping the smart-cell
  default path uses, so both agree on which cell sits in which metro.

  These prices drive COST and cell choice only. Blocking is decided once per
  combo from the pool-level price -- see ``_evaluate_block``.
  """
  cap = verdict.capacity
  if not cap or not cap.cells_ok:
    return []

  allowed = {m.strip().lower() for m in (metros or []) if m.strip()}

  # Pool-scoped lookup. capacity.py resolved the alloc's real pool; passing it
  # keeps another pool's market (which can be 60% dearer for the same cell) out
  # of the decision.
  prices = snapshot.get_prices_for_arch(arch, tier, pool=cap.pool or None)

  offers: list[CellOffer] = []
  for cell_cap in cap.cells_ok:
    if allowed and metro_util.metro_of(cell_cap.cell) not in allowed:
      continue
    known = cell_cap.cell in prices
    price = prices.get(cell_cap.cell)
    offers.append(CellOffer(
        cell=cell_cap.cell,
        obtainable=cell_cap.obtainable,
        price=price,
        price_known=known,
        unobtainable=known and price is None,
        cost_per_hour=None if price is None else chips * price,
    ))

  # Cheapest first; unpriced cells sort last but stay selectable, because "we
  # have no quote" is not evidence that the cell is bad. Prices tie constantly
  # (GQM clears a whole price tier at once -- 119 v6e cells shared one price on
  # 2026-07-29), so the tie-breaks carry most of the decision: prefer real
  # production cells over staging/test ones, then the deepest pool of chips.
  offers.sort(key=lambda o: (o.price is None, o.price or 0.0,
                             _is_nonprod_cell(o.cell), -o.obtainable))
  return offers


# Cells whose names carry these suffixes are cloud-staging or test cells. They
# do appear in GetCellAvailability with real chip counts, but pinning a job to
# one is almost never what the user meant, so they lose every tie. They are not
# dropped: if a staging cell is the only place with capacity, saying so beats
# saying nothing.
_NONPROD_CELL_MARKERS = ('-c-staging', '-c-test', '-staging', '-test')


def _is_nonprod_cell(cell: str) -> bool:
  return any(m in cell for m in _NONPROD_CELL_MARKERS)


def _evaluate_block(alloc: str, arch: str, tier: str,
                    verdict: preflight.Verdict,
                    snapshot: market.MarketSnapshot
                    ) -> tuple[bool, str, Optional[float],
                               Optional[market.LimitOrder]]:
  """Decide whether a limit order blocks this combo, pool-wide.

  Returns ``(blocked, reason, pool_price, limit_order)``.

  The comparison uses the POOL-level price, not any cell's, because the V2
  auction collapses every cell into one synthetic layer before evaluating caps.
  A blocked combo is blocked everywhere; there is no cheap cell to escape to.
  """
  pool = (verdict.capacity.pool or None) if verdict.capacity else None
  limit_order = snapshot.get_limit_order_for_arch(alloc, arch, tier, pool=pool)
  pool_price = snapshot.get_pool_price_for_arch(arch, tier, pool=pool)
  if limit_order is None:
    # No row means no `milli_credit_limit_price`, so the reason can never fire.
    return (False, '', pool_price, None)
  if not market.is_blocked(pool_price, limit_order.cap):
    return (False, '', pool_price, limit_order)
  assert pool_price is not None  # is_blocked() is False when price is None
  return (True,
          f'{arch} {tier} clears at {pool_price:.2f} credits/chip-hr pool-wide, '
          f'above the {limit_order.cap:.2f} cap set by '
          f'{limit_order.user or "?"}',
          pool_price, limit_order)


def _pick_offer(offers: list[CellOffer]) -> Optional[CellOffer]:
  """Choose the cell to recommend: the cheapest one that has chips.

  Price is a cost decision only -- it cannot lift a limit-order block -- so
  this never filters, it only orders.
  """
  if not offers:
    return None
  # An unobtainable quote means no chips at any price in that cell this cycle.
  # If that is true of every cell we fall back to the full list rather than
  # returning nothing: L2 said the chips are there, so a stale price row should
  # not veto the whole combo.
  usable = [o for o in offers if not o.unobtainable] or offers
  return usable[0]


def rank(candidates: list[Candidate], tier: str = 'PROD') -> list[Candidate]:
  """Rank candidates. Sort key, in order:

    1. blocked ascending   -- never put a limit-order-blocked combo on top.
    2. status  ascending   -- GREEN before YELLOW (RED never reaches here).
    2c. group preference    -- PROD only; spend g3/g5 (own balance, exempt from
                              the G9 income/10 cap) before g9 and the rest. See
                              ``_GROUP_PREF``. Neutral at BATCH (free pool).
    3. headroom descending -- see the PROD/BATCH split below.
    3b. unverified asc     -- PROD only; quota==0 sinks. See below.
    4. cost_per_hour asc   -- chips * per-chip-hour price. Prefer cheap cells:
                              it saves credits and, more importantly, keeps you
                              under a future cap.
    5. arch preference     -- newer first (v6e > v6p > v5p > v4 > v5e).
    6. chips ascending     -- smaller footprint wins ties.
    7. floor descending    -- PROD only; a bigger claim breaks a cost tie.
    8. obtainable desc, then group id -- final tie-breaks; see below.

  Step 2c (group preference) is placed ABOVE the economics (headroom, cost,
  floor) on purpose: g3/g5 draw on their own credit balance and do NOT count
  against the G9 income/10 budget the launch gate enforces, so spending them
  first is strictly cheaper in the resource that is actually scarce. It sits
  BELOW blocked+status so it can never promote a non-runnable or lower-
  confidence (YELLOW-over-GREEN) placement just to save budget -- correctness
  of placement still outranks whose budget pays. It is neutral at BATCH, where
  every group draws from the same free pool and no floor/credit is spent.

  The PROD/BATCH difference in step 3 is deliberate and load-bearing:

  * PROD enters the lease and market buckets, where the alloc's floor
    (``floor_v2``) really does gate admission, so `remaining_quota / chips` is
    a meaningful measure of how comfortably the request fits.
  * BATCH lands in ``BUCKET_ID_BATCH_TIER``, the LAST bucket processed, and
    that pass never consults the floor at all -- its only test is
    ``DemandFitsInRootPoolCapacity``. A BATCH job runs fine with floor_v2 == 0,
    which is why "BATCH quota" is a meaningless number to rank on. For BATCH we
    therefore rank on obtainable chips in the chosen cell, which is the thing
    that actually decides admission -- and step 3b is skipped entirely, since
    holding a floor buys a BATCH job nothing.

    A consequence worth stating: BATCH capacity is a property of the POOL, so
    every group sharing a pool ties on steps 3-7 and the order among them is
    arbitrary but not wrong -- their prospects really are identical. Step 8
    settles it on group id purely so the output is stable between runs.

  Steps 3b and 7 exist because `remaining == 0` is produced by two very
  different states that step 3 alone cannot tell apart:

  * the alloc holds a real floor for this chip and has spent it, versus
  * ``floor_v2`` came back 0, i.e. the alloc holds no floor for this chip at
    all -- usually because it does not participate in that market. Preflight
    already says so out loud ("Could not read PROD quota ... Cannot verify
    headroom").

  Ranking those equal is what used to fill the top of the table with
  unverifiable allocs while the one group that genuinely holds the chips sat
  below the fold. Step 3b sinks the unverifiable ones. It is only a flag, not a
  magnitude, so it cannot outvote price: comparing floor SIZES happens at step
  7, after cost, so a free cell still beats a bigger-but-dearer claim.
  Both are neutral at BATCH, where holding a floor buys nothing.
  """
  is_batch = tier.upper() == 'BATCH'
  status_rank = {preflight.Status.GREEN.value: 0,
                 preflight.Status.YELLOW.value: 1}

  def key(c: Candidate):
    headroom = c.obtainable_ratio if is_batch else c.headroom_ratio
    quota = (c.verdict.capacity.alloc_scoped_quota
             if c.verdict.capacity else 0)
    unverified = 0 if (is_batch or quota > 0) else 1
    floor_rank = 0 if is_batch else -quota
    # Whose budget to spend first. Neutral at BATCH (one free pool, nothing is
    # spent), active at PROD where g3/g5 are exempt from the G9 income/10 cap.
    group_pref = (0 if is_batch
                  else _GROUP_PREF.get(c.group_id, _GROUP_PREF_DEFAULT))
    # Unknown price sorts after every known one, so a priced cheap cell always
    # beats an unpriced guess.
    cost = float('inf') if c.cost_per_hour is None else c.cost_per_hour
    return (
        1 if c.blocked else 0,                        # ascending: 0 = runnable
        status_rank.get(c.verdict.status.value, 99),  # ascending: 0 = GREEN
        group_pref,                                   # ascending: g3/g5 first
        -headroom,                                    # descending
        unverified,                                   # ascending: verified 1st
        cost,                                         # ascending: cheap first
        _ARCH_PREF.get(c.arch, 99),
        c.chips,
        floor_rank,                                   # ascending: bigger floor
        # Final tie-breaks. With a shared clearing price and identical
        # verdicts, the cell holding more spare chips is the safer landing spot
        # (fragmentation is invisible to L2, so depth is a weak proxy). Group
        # id last so that genuinely-equivalent rows -- which is every
        # same-pool group at BATCH -- come out in a stable order.
        -c.obtainable,
        c.group_id,
    )

  return sorted(candidates, key=key)


def route(power: str,
          tier: str = 'PROD',
          groups: Optional[list[int]] = None,
          tolerance: float = 0.5,
          top_k: int = 3,
          progress_fn=None,
          snapshot: Optional[market.MarketSnapshot] = None,
          metros: Optional[list[str]] = None
          ) -> tuple[list[Candidate], market.MarketSnapshot]:
  """Main entry point.

  Args:
    power:     e.g. 'v5p-32', 'v6e-16', or bare int '32'.
    tier:      'PROD' or 'BATCH'.
    groups:    subset of group ids to consider. Default: every id in
               ``group_utils.GROUP_MAP``.
    tolerance: fractional slack around target power (0.5 = +-25%).
    top_k:     number of top candidates to return.
    progress_fn: optional (str) -> None for streaming progress messages.
    snapshot:  pre-loaded market data; loaded from the daemon cache if omitted.
               Injectable so tests can pin prices without a running daemon.
    metros:    data-locality allow-list of metros (e.g. ['cbf', 'tul']). When
               set, ONLY cells in those metros are eligible; a combo with no
               in-metro cell drops out entirely, so the result is empty rather
               than out-of-metro when the whole allow-list is full. Default
               None/[] = any metro (roams the fleet, today's behaviour).

  Returns ``(candidates, snapshot)``. Candidates are sorted best first and may
  include entries with ``blocked=True`` when nothing is runnable -- the caller
  decides how loudly to say so. The snapshot is returned so the CLI can report
  the data's age (and whether there was any).
  """
  from google3.experimental.users.qiaos.tpu_utils import group_utils

  if snapshot is None:
    snapshot = market.load_snapshot()
  if progress_fn:
    if snapshot.available:
      progress_fn(f'market data: {snapshot.age_str()} old, '
                  f'{len(snapshot.limit_orders)} limit orders, pools='
                  f'{",".join(snapshot.pools) or "?"}')
    else:
      progress_fn(f'market data UNAVAILABLE: {snapshot.warning}')

  target = parse_power_input(power)
  if progress_fn:
    progress_fn(f"target power ~ {target} v5p-equivalent chips "
                f"(tolerance +-{tolerance*50:.0f}%)")

  options = _candidate_options(target, tolerance)
  if progress_fn:
    progress_fn(f"expanded to {len(options)} (arch, chips) options: {options}")

  if groups is None:
    # Every group we know about. Deriving this from GROUP_MAP rather than a
    # literal range is the whole fix for the g9 bug: the old `range(1, 9)`
    # stopped at 8 and silently skipped the user's only real dynamic alloc.
    groups = sorted(group_utils.GROUP_MAP)

  # Build the full task list first (arch, chips, gid, alloc, tpu_type).
  tasks: list[tuple[str, int, int, str, str]] = []
  for arch, chips in options:
    for gid in groups:
      alloc = group_utils.get_alloc_by_id(gid)
      if not alloc:
        if progress_fn:
          progress_fn(f'  skipping unknown group id {gid}')
        continue
      tasks.append((arch, chips, gid, alloc, f'{arch}-{chips}'))

  candidates: list[Candidate] = []
  total = len(tasks)

  def _probe_one(task):
    arch, chips, gid, alloc, tpu_type = task
    try:
      verdict = preflight.run_preflight(tpu_type, alloc, tier)
    except Exception as e:  # pylint: disable=broad-except
      # One bad alloc must not take down the whole fan-out; the error is
      # surfaced per-task through progress_fn instead of being swallowed.
      return (task, None, e)
    return (task, verdict, None)

  # Fan out concurrently; the underlying resource_service + smartservice calls
  # release the GIL during stubby RPC waits.
  with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    completed = 0
    for task, verdict, err in pool.map(_probe_one, tasks):
      completed += 1
      arch, chips, gid, alloc, tpu_type = task
      if err is not None:
        if progress_fn:
          progress_fn(f"    ERROR probing g{gid}/{tpu_type}: "
                      f"{type(err).__name__}: {err}")
        continue
      if verdict is None or verdict.status == preflight.Status.RED:
        if progress_fn:
          s = verdict.status.value if verdict else '?'
          progress_fn(f"  [{completed}/{total}] g{gid} {tpu_type} -> {s}")
        continue

      # Per-cell offers, with the metro allow-list (data-locality) applied as a
      # HARD filter first. When ``metros`` is set and empties the list, the
      # combo has no in-metro cell: drop it entirely so it can neither rank nor
      # be recommended. That is the fail-closed half of --power --metro -- an
      # all-full allow-list yields an empty RESULT rather than an out-of-metro
      # (no-data) recommendation the dataloader would crash on.
      offers = _evaluate_cells(arch, chips, tier, verdict, snapshot,
                               metros=metros)
      if metros and not offers:
        if progress_fn:
          progress_fn(f"  [{completed}/{total}] g{gid} {tpu_type} -> "
                      f"no cell in metro(s) {','.join(metros)}")
        continue

      # Two independent questions, deliberately answered from two different
      # numbers: the pool-level price decides whether a cap blocks the combo,
      # the per-cell prices decide where to run and what it costs.
      blocked, block_reason, pool_price, limit_order = _evaluate_block(
          alloc, arch, tier, verdict, snapshot)
      best = _pick_offer(offers)

      if progress_fn:
        s = verdict.status.value
        where = f' @ {best.cell}' if best else ''
        cost = ('' if not best or best.cost_per_hour is None
                else f' {best.cost_per_hour:.0f} cr/hr')
        flag = ' BLOCKED' if blocked else ''
        progress_fn(f"  [{completed}/{total}] g{gid} {tpu_type} -> "
                    f"{s}{where}{cost}{flag}")

      candidates.append(Candidate(
          group_id=gid, alloc=alloc, arch=arch, chips=chips, tpu_type=tpu_type,
          power_score=to_power(arch, chips), verdict=verdict,
          cell=best.cell if best else '',
          price=best.price if best else None,
          cost_per_hour=best.cost_per_hour if best else None,
          obtainable=best.obtainable if best else 0,
          pool_price=pool_price,
          blocked=blocked, block_reason=block_reason,
          limit_order=limit_order,
          offers=tuple(offers)))

  ranked = rank(candidates, tier=tier)
  return (ranked[:top_k], snapshot)


def route_all(power: str, tier: str = 'PROD',
              groups: Optional[list[int]] = None,
              tolerance: float = 0.5,
              progress_fn=None,
              snapshot: Optional[market.MarketSnapshot] = None,
              metros: Optional[list[str]] = None
              ) -> tuple[list[Candidate], market.MarketSnapshot]:
  """``route`` with no top-K truncation. Used by --explain, which must be able
  to list the combos that were excluded as well as the ones that survived."""
  return route(power=power, tier=tier, groups=groups, tolerance=tolerance,
               top_k=1 << 30, progress_fn=progress_fn, snapshot=snapshot,
               metros=metros)
