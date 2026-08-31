"""Pure scheduling core for the local job queue + router.

WHY THIS FILE HAS NO I/O
------------------------
Everything here is a pure function of its arguments: queue entries in, a
placement decision out. The availability snapshot, the wall-clock time, and the
random seed are all INJECTED, never read from the world. That is what makes the
scheduler unit-testable without a TPU pool, a daemon, or a clock -- the hard
part (oversold cells, fragmentation, priority ties, tpu-type fallback, the
10-minute re-route) is decided here and pinned by tests. The binary
(`route_check.py`) is the only place that touches RPCs, the queue file, and
`tpu queue`/`tpu cancel`; it calls into this module for every decision.

THE MODEL
---------
Two queues. The XManager queue is not ours: admitting a job commits it to one
cell, and if that cell is oversold it sits PENDING for hours. On top of it we
keep a LOCAL queue -- a durable list of desired runs the operator submits
freely (pending costs no group credit; balance bills on usage, and a pending
Borg gang has zero tasks). The router drains the local queue into the XM queue
one placement at a time, choosing a cell that can actually place the slice NOW.

A job names what it will ACCEPT, not one concrete target:
  - a power level (v5p-equivalent compute), so the router may satisfy it with
    any equivalent (arch, chips) pair;
  - the tpu families it allows (v7, v6p, ...), in preference order;
  - a tier (PROD/BATCH);
  - the metros it allows (data-locality / latency);
  - a router-level priority (UNRELATED to XM priority): higher goes first, and
    among equal priority the router picks RANDOMLY each tick so no single job
    starves the others.

The scarce resource is per-cell PLACEABLE SLICES, and the deciding signal is
`GetCellAvailability`'s max_available_chips (NOT obtainable, which lies) minus
any oversold flag. Free chips can still be fragmented with no contiguous slice,
so a placement is a TRY: the binary submits, watches, and if the job is still
PENDING after the re-route deadline it cancels and returns the job here to be
placed somewhere else. That watch/re-route loop is what makes fragmentation
survivable -- we cannot predict it, only react to it.
"""

from __future__ import annotations

import dataclasses
import enum
import random
from typing import Callable, Optional


# --------------------------------------------------------------------------
# Compute-equivalence: v5p-equivalent chips per (arch, chips). Mirrors
# preflight/router.py::_V5P_MULTIPLIER so the queue speaks the same "power"
# language as `tpu route --power`. Kept as a local copy so this module has NO
# import that drags in RPC stubs -- the router binary reconciles the two.
# --------------------------------------------------------------------------
V5P_MULTIPLIER: dict[str, float] = {
    'v5e': 0.43,
    'v4': 0.60,
    'v5p': 1.0,
    'v6e': 2.0,
    'v6p': 4.34,
    'v7': 4.34,
    # NVIDIA GPUs, per-chip compute vs v5p from the internal `vle` rate card
    # (borg/util/reports/gxu/gxus_by_platform_ga.textproto): A100 1.58/2.33,
    # H100/H200 5.02/2.33, B200/B300/GB200/GB300 11.42/2.33. This lets
    # `--power` compare a GPU ask against a TPU one -- but a GPU only enters a
    # candidate set when the job's OWN `--archs` lists it, so a TPU run is
    # never silently handed a GPU.
    'a100': 0.68, 'a100_80gib': 0.68,
    'h100': 2.15, 'h200': 2.15,
    'b200': 4.90, 'b300': 4.90, 'gb200': 4.90, 'gb300': 4.90,
}

# Newer-first preference among equally-good architectures (mirrors _ARCH_PREF).
# GPUs sort AFTER all TPUs: when a job lists both a TPU and a GPU arch and both
# clear, the TPU wins the tie unless price/pool ordering (below) says otherwise.
ARCH_PREF: dict[str, int] = {
    'v7': 0, 'v6p': 1, 'v6e': 2, 'v5p': 3, 'v4': 4, 'v5e': 5,
    'gb300': 10, 'gb200': 11, 'b300': 12, 'b200': 13,
    'h200': 14, 'h100': 15, 'a100_80gib': 16, 'a100': 17}

# Legal chip counts per arch (mirrors topology._LOCUS_TABLE keys). The router
# binary can override this with the authoritative topology module; the copy
# keeps the core self-contained and testable.
LEGAL_SIZES: dict[str, list[int]] = {
    'v4': [8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    'v5p': [8, 16, 32, 64, 128, 256, 512, 1024],
    'v6p': [8, 16, 32, 64, 128, 256, 512],
    'v7': [4, 8, 16, 32],
    'v6e': [8, 16, 32, 64, 128, 256],
    'v5e': [8, 16, 32, 64],
    # NVIDIA GPUs -- non-torus; the size is a device count capped at the card's
    # NVLink domain (mirrors topology._GPU_LEGAL). GPUs have NO GEOMETRY entry
    # below, so a topology-locked job can never route ONTO or OFF a GPU (its
    # geometry match always fails), which is the correct safe default for a
    # mesh-sharded checkpoint.
    'a100': [1, 2, 4, 8, 16],
    'a100_80gib': [1, 2, 4, 8],
    'h100': [1, 2, 4, 8],
    'h200': [1, 2, 4, 8],
    'b200': [1, 2, 4, 8],
    'b300': [1, 2, 4, 8],
    'gb200': [1, 2, 4, 8, 16, 32, 64, 72],
    'gb300': [1, 2, 4, 8, 16, 32, 64, 72],
}

# Mesh GEOMETRY per (arch, chips) -- mirrors topology._LOCUS_TABLE values. This
# is what "same topology" MEANS: a checkpoint sharded for a 2x4x4 mesh can only
# resume onto another 2x4x4 mesh. Two shapes are topology-equivalent iff their
# geometry strings match, which is a stronger and more correct test than "same
# arch" or "same chip count":
#   v6p-32 and v7-32 are BOTH 2x4x4  -> a topology-locked paligemma job may move
#                                       between them (the operator's example);
#   v6p-32 (2x4x4) and v6e-32 (4_8)  -> different mesh, NOT interchangeable even
#                                       though the chip COUNT is equal.
# The 3-D torus family (v4/v5p/v6p/v7) shares geometry at every legal size; the
# 2-D pod family (v6e/v5e) has its own. A job that can retrain from scratch is
# NOT locked and ignores this table entirely.
GEOMETRY: dict[str, dict[int, str]] = {
    'v4': {8: '2x2x2', 16: '2x2x4', 32: '2x4x4', 64: '4x4x4', 128: '4x4x8',
           256: '4x8x8', 512: '4x8x16', 1024: '8x8x16', 2048: '8x16x16'},
    'v5p': {8: '2x2x2', 16: '2x2x4', 32: '2x4x4', 64: '4x4x4', 128: '4x4x8',
            256: '4x8x8', 512: '4x8x16', 1024: '8x8x16'},
    'v6p': {8: '2x2x2', 16: '2x2x4', 32: '2x4x4', 64: '4x4x4', 128: '4x4x8',
            256: '4x8x8', 512: '4x8x16'},
    'v7': {4: '2x2x1', 8: '2x2x2', 16: '2x2x4', 32: '2x4x4'},
    'v6e': {8: '2_4', 16: '4_4', 32: '4_8', 64: '8_8', 128: '8_16_wrap_y',
            256: '16_16_wrap_xy'},
    'v5e': {8: '2_4', 16: '4_4', 32: '4_8', 64: '8_8'},
}


def geometry_of(arch: str, chips: int) -> Optional[str]:
  """Mesh geometry string for (arch, chips), or None if not a legal shape."""
  return GEOMETRY.get(arch.lower(), {}).get(chips)


def same_topology(arch_a: str, chips_a: int, arch_b: str, chips_b: int) -> bool:
  """True iff the two shapes share a mesh geometry (checkpoint-resume-safe).

  This is the test a topology-locked job uses to decide whether it may move to a
  different (arch, chips). v6p-32 <-> v7-32 is True; v6p-32 <-> v6e-32 is False.
  """
  ga = geometry_of(arch_a, chips_a)
  gb = geometry_of(arch_b, chips_b)
  return ga is not None and ga == gb


def to_power(arch: str, chips: int) -> float:
  """v5p-equivalent chip count for (arch, chips)."""
  return V5P_MULTIPLIER.get(arch.lower(), 1.0) * float(chips)


def parse_power(power: str) -> float:
  """Parse 'v5p-32' / 'v6e-16' / bare '32' into v5p-equivalent chips."""
  s = str(power).strip().lower()
  if s.isdigit():
    return float(int(s))
  for sep in ('-', '='):
    if sep in s:
      arch, cores = s.split(sep, 1)
      return to_power(arch, int(cores))
  raise ValueError(f"cannot parse power spec {power!r}")


def power_geometry(power: str) -> Optional[str]:
  """The mesh geometry a power spec names directly, e.g. 'v6p-32' -> '2x4x4',
  or None for a bare chip count (no single arch, so no single geometry).

  This is the ANCHOR geometry for a topology-locked job that has not been placed
  yet: the checkpoint was sharded for the shape the operator asked for, so the
  first placement must match that mesh -- not merely any shape within the power
  tolerance. Without this, a locked 'v6p-32' (2x4x4) job could first land on a
  v6e-64 (8_8) slice that happened to fall inside the power window.
  """
  s = str(power).strip().lower()
  for sep in ('-', '='):
    if sep in s:
      arch, cores = s.split(sep, 1)
      try:
        return geometry_of(arch, int(cores))
      except ValueError:
        return None
  return None


# --------------------------------------------------------------------------
# TYPE SELECTION: cheapest-effective-first, with a big-pool bonus.
#
# The operator's rule: pick the tpu type by price (cheaper ~= easier to get),
# BUT a type whose pool is large is more schedulable and more stable, so it may
# win even when nominally a bit more expensive -- worth up to ~20% price
# forgiveness for a big pool. Pool size MOVES DAILY, so it is never hardcoded:
# the binary passes the LIVE per-arch pool magnitude (sum of obtainable/free
# chips across the job's allowed metros) computed from the same availability
# data the cell step uses.
# --------------------------------------------------------------------------
POOL_BONUS = 0.20   # max fraction of price a large pool is forgiven
# Pool magnitude (chips) at/above which the FULL bonus applies. Below it the
# bonus scales in log-proportion, so a tiny pool earns almost nothing. ~4096
# chips = a genuinely deep pool (yukulwh-class); a one-cell 500-chip pool earns
# only a sliver.
POOL_FULL_BONUS_CHIPS = 4096.0


def pool_weight(pool_chips: float) -> float:
  """Multiplier (1.0 .. 1+POOL_BONUS) rewarding a bigger live pool.

  A pool at/above POOL_FULL_BONUS_CHIPS earns the full POOL_BONUS; smaller pools
  earn a log-scaled fraction so the reward tapers smoothly to ~0 for a thin
  one-cell pool. Returns 1.0 for an empty/unknown pool (no reward, no penalty).
  """
  import math
  if pool_chips <= 0:
    return 1.0
  frac = math.log1p(pool_chips) / math.log1p(POOL_FULL_BONUS_CHIPS)
  frac = max(0.0, min(1.0, frac))
  return 1.0 + POOL_BONUS * frac


def effective_price(raw_price: float, pool_chips: float) -> float:
  """Price discounted by the pool-size bonus: a big pool reads as cheaper.

  effective = raw / pool_weight, so a full-bonus pool makes a type read as if it
  were 1/(1.2) ~= 17% cheaper, floating a large-pool type above a marginally
  cheaper but thin one. Lower is better.
  """
  return raw_price / pool_weight(pool_chips)


class JobState(str, enum.Enum):
  """Lifecycle of a local-queue entry. Strings so the JSON file is readable."""
  QUEUED = 'QUEUED'        # waiting for the router to place it
  BUILD_REQUESTED = 'BUILD_REQUESTED'  # the router decided this round to dispatch
                           # it; the serial builder will pick it up next. Transient,
                           # QUEUED -> BUILD_REQUESTED -> BUILDING. Counts (with
                           # BUILDING) as "builder not yet drained" for backpressure:
                           # the router does not open a new dispatch round while any
                           # BUILD_REQUESTED/BUILDING remains, so headroom math is
                           # not stale. NOT picked by next_queued (only next_build
                           # _requested claims it).
  BUILDING = 'BUILDING'    # a serial worker is running `tpu queue` for it NOW
  HELD = 'HELD'            # parked: cannot build as-is (bad workdir / too many
                           # failed attempts). NOT picked by the worker until a
                           # human fixes it (re-enqueue) -- prevents churn.
  BUDGET_DEFERRED = 'BUDGET_DEFERRED'  # over the G9 credit bar THIS round: a soft,
                           # non-terminal, auto-recoverable park. NOT HELD, NOT a
                           # build failure -- attempts is NOT incremented. Re-tested
                           # every round (promote_deferred -> QUEUED at top of round);
                           # flows back automatically when headroom opens. A budget
                           # refusal is a transient FLEET state, not a per-job defect.
  SUBMITTED = 'SUBMITTED'  # handed to XM, watching for RUNNING vs re-route
  RUNNING = 'RUNNING'      # confirmed running; the router is done with it
  DONE = 'DONE'            # finished (terminal)
  FAILED = 'FAILED'        # gave up / user cancelled (terminal)


TERMINAL_STATES = frozenset({JobState.RUNNING, JobState.DONE, JobState.FAILED})

# States whose XID the XM-truth reconcile pass re-verifies against XManager.
# NOTE: this is deliberately NOT TERMINAL_STATES -- RUNNING is "terminal" there
# only in the sense of "router stops tracking", but a local RUNNING entry is
# exactly what turns into a zombie when XM has since dropped it, so reconcile
# MUST re-check it. BUILD_REQUESTED has no XID yet (builder hasn't run) so it is
# not reconcilable; BUDGET_DEFERRED/QUEUED/HELD have no live XID either.
RECONCILABLE_STATES = frozenset(
    {JobState.RUNNING, JobState.SUBMITTED, JobState.BUILDING})


@dataclasses.dataclass
class QueueEntry:
  """One desired run in the local queue.

  Only the scheduling-relevant fields live here; launch details (config, bucket,
  exp_name, ...) ride along in `launch_kwargs` untouched and are handed to
  `tpu queue` verbatim when the job is placed.
  """
  job_id: str                       # local id, our own (NOT an XID)
  power: str                        # 'v5p-32', 'v6e-16', or a bare int
  allowed_archs: list[str]          # ['v7', 'v6p'] -- families the job accepts
  tier: str = 'PROD'                # PROD | BATCH
  allowed_metros: Optional[list[str]] = None   # None/[] = any metro
  priority: int = 0                 # router priority; higher first
  power_tolerance: float = 0.5      # accept power in [0.75x, 1.5x] of target
  max_price: Optional[float] = None # limit-order cap, credits/chip-hr
  # TOPOLOGY LOCK. False (default) = the job can retrain from scratch, so the
  # router may pick any equivalent (arch, chips). True = the job resumes from a
  # checkpoint sharded for one mesh (e.g. paligemma), so it may ONLY move to a
  # shape with the SAME geometry. v6p-32 and v7-32 are both 2x4x4 and thus
  # interchangeable even when locked; v6e-32 (4_8) is not. When locked and
  # already placed once, `locked_geometry` pins the mesh; if unset, the geometry
  # of the first candidate shape is adopted and frozen.
  topology_locked: bool = False
  locked_geometry: Optional[str] = None   # e.g. '2x4x4'; None until first placed
  launch_kwargs: dict = dataclasses.field(default_factory=dict)
  # WORKDIR the submit runs `tpu queue` FROM. `tpu queue` rsyncs the source tree
  # from its CWD into the stagedir, so a run whose config/edits live in a
  # particular checkout (not passed via --config) MUST be packaged from that
  # directory -- otherwise the router ships a copy of the WRONG source. Empty
  # means "wherever the router process runs", which is only safe for a run whose
  # every difference is passed as an explicit flag. `tpu enqueue` defaults this
  # to the CWD at enqueue time, so enqueuing from the right checkout just works.
  workdir: str = ''
  # ---- mutable state ----
  state: JobState = JobState.QUEUED
  xid: Optional[str] = None         # set once submitted
  cell: Optional[str] = None        # cell it was placed into
  arch: Optional[str] = None        # concrete arch chosen
  chips: Optional[int] = None       # concrete chip count chosen
  submitted_at: Optional[float] = None   # epoch when handed to XM
  build_started_at: Optional[float] = None  # epoch a worker claimed it (BUILDING)
  worker_id: Optional[str] = None   # which worker claimed it (BUILDING); for debug
  attempts: int = 0                 # BUILD failures so far -- the 3-strikes
                                    # brake reads THIS. Never bump it for a
                                    # re-route: a re-route is the router moving
                                    # a healthy job to another cell, not the job
                                    # failing to build (infra-v17).
  reroutes: int = 0                 # how many times the router moved this job
  # ★Which alloc group the ROUTER admitted this job under (operator 00:05Z:
  # "我反复要求过优先用 G5 / G3"). The dispatch and build stages are separate
  # processes, so the group chosen while checking budget must be carried ON THE
  # ROW -- otherwise the builder falls back to its global --group (9) and every
  # car lands on the one pool that has a hard 1/10-of-income ceiling, while g5/g3
  # sit idle. None = not yet decided; the builder then uses its own default.
  group: Optional[str] = None
  # ★EVERY XID THIS ROW HAS EVER HELD, oldest first, excluding the current one.
  # `xid` is overwritten on each placement, so without this a resubmitted row
  # leaves its earlier experiment with NO local record anywhere -- which reads
  # exactly like an orphan to any XM->local audit, and is unfindable when you
  # need to stop it. A row with attempts>0 and an empty prior_xids is itself a
  # signal: the history predates this field.
  prior_xids: list[str] = dataclasses.field(default_factory=list)
  cooldown_cells: dict = dataclasses.field(default_factory=dict)  # cell -> until-epoch
  last_reason: str = ''             # why it is where it is (for status view)

  def to_dict(self) -> dict:
    d = dataclasses.asdict(self)
    d['state'] = self.state.value
    return d

  @classmethod
  def from_dict(cls, d: dict) -> 'QueueEntry':
    d = dict(d)
    if 'state' in d and not isinstance(d['state'], JobState):
      d['state'] = JobState(d['state'])
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in known})


@dataclasses.dataclass(frozen=True)
class CellAvail:
  """Per-cell availability for ONE arch, from GetCellAvailability.

  `free_chips` is max_available_chips -- what is actually free now, the number
  that decides. `oversold` is the cell's oversold flag. `price` is the pool
  clearing price for the arch (one global layer; per-cell only for cost). We do
  NOT store `obtainable` for decisions -- it is a quota-shaped promise that
  reported 1616 while a cell held 3 free chips.
  """
  cell: str
  arch: str
  free_chips: int
  oversold: bool
  price: Optional[float] = None      # credits/chip-hr (pool price)
  metro: str = ''


def slices_for(free_chips: int, chips_per_slice: int) -> int:
  """Upper bound on placeable slices: free // per-slice. UPPER BOUND because
  free chips may be fragmented with no contiguous slice among them."""
  if chips_per_slice <= 0:
    return 0
  return free_chips // chips_per_slice


@dataclasses.dataclass(frozen=True)
class Placement:
  """The router's decision for one job: submit `arch`-`chips` into `cell`."""
  job_id: str
  arch: str
  chips: int
  cell: str
  price: Optional[float]
  reason: str
  geometry: Optional[str] = None   # mesh of the chosen shape; frozen if locked


def candidate_shapes(entry: QueueEntry,
                     legal_sizes: Optional[dict[str, list[int]]] = None,
                     arch_price: Optional[dict[str, float]] = None,
                     arch_pool: Optional[dict[str, float]] = None,
                     ) -> list[tuple[str, int]]:
  """(arch, chips) options this job accepts, best-first.

  Filters to: the job's allowed archs; legal chip counts for each; power within
  tolerance of the target.

  ORDERING -- two modes:
    * With market data (`arch_price` = credits/chip-hr per arch, `arch_pool` =
      live pool chips per arch): rank by EFFECTIVE price ascending, i.e. raw
      price discounted by the pool-size bonus, so the cheapest type that is also
      easy to get floats up, and a big pool can outrank a marginally cheaper
      thin one (the operator's ~20% rule). Ties fall back to ARCH_PREF then
      fewer chips. This is the SELECT-TYPE-THEN-CELL policy: pick the type here,
      pick the cell in `best_cell_for_shape`.
    * Without market data (tests, offline): fall back to ARCH_PREF (newer first)
      then fewer chips -- the original deterministic order.

  TOPOLOGY LOCK: if the job is topology_locked, shapes are additionally filtered
  to those whose mesh geometry matches. If `locked_geometry` is already pinned
  (the job has been placed before), only that geometry is allowed -- so a
  paligemma checkpoint on 2x4x4 may re-route between v6p-32 and v7-32 but never
  onto v6e-32. If NOT yet pinned, the anchor geometry is the one the job's own
  `power` spec names (v6p-32 -> 2x4x4); only matching shapes are eligible, so a
  locked job cannot first land on a same-power but different-mesh slice (v6e-64
  is 8_8, not 2x4x4). A bare-int power names no geometry, so a locked job MUST
  specify an arch in its power spec (enforced below).
  """
  legal_sizes = legal_sizes or LEGAL_SIZES
  target = parse_power(entry.power)
  tol = entry.power_tolerance
  low, high = target * (1.0 - tol / 2.0), target * (1.0 + tol / 2.0)
  allowed = [a.lower() for a in entry.allowed_archs]
  # The geometry a locked job must match: the pinned one if already placed,
  # else the one its power spec names.
  lock_geom = None
  if entry.topology_locked:
    lock_geom = entry.locked_geometry or power_geometry(entry.power)
  out: list[tuple[str, int]] = []
  for arch in allowed:
    for chips in legal_sizes.get(arch, []):
      p = to_power(arch, chips)
      if not (low <= p <= high):
        continue
      if entry.topology_locked:
        # A locked job with no resolvable anchor geometry (bare-int power and
        # not yet placed) is unsafe to place at all -- skip everything rather
        # than guess a mesh for a sharded checkpoint.
        if lock_geom is None or geometry_of(arch, chips) != lock_geom:
          continue
      out.append((arch, chips))

  if arch_price:
    def eff(ac):
      arch, chips = ac
      raw = arch_price.get(arch)
      if raw is None:
        return (float('inf'), ARCH_PREF.get(arch, 99), chips)
      pool = (arch_pool or {}).get(arch, 0.0)
      return (effective_price(raw, pool), ARCH_PREF.get(arch, 99), chips)
    out.sort(key=eff)
  else:
    out.sort(key=lambda ac: (ARCH_PREF.get(ac[0], 99), ac[1]))
  return out


def best_cell_for_shape(
    arch: str,
    chips: int,
    entry: QueueEntry,
    avail_by_cell: dict[str, CellAvail],
    now: float,
) -> Optional[tuple[CellAvail, int]]:
  """Best placeable cell for one (arch, chips), or None if none can place it.

  Applies, in order: metro filter, price cap, oversold drop, cooldown drop,
  and >=1 placeable slice. Ranks survivors by placeable slices desc, then price
  asc, then free chips desc (headroom). Returns (cell_avail, n_slices)."""
  allowed_metros = [m.lower() for m in (entry.allowed_metros or [])]
  ranked: list[tuple[int, float, int, CellAvail]] = []
  for ca in avail_by_cell.values():
    if ca.arch.lower() != arch.lower():
      continue
    if allowed_metros and ca.metro.lower() not in allowed_metros:
      continue
    if entry.max_price is not None and ca.price is not None and ca.price > entry.max_price:
      continue
    if ca.oversold:
      continue
    if entry.cooldown_cells.get(ca.cell, 0) > now:
      continue
    n = slices_for(ca.free_chips, chips)
    if n < 1:
      continue
    price_key = ca.price if ca.price is not None else float('inf')
    ranked.append((n, price_key, ca.free_chips, ca))
  if not ranked:
    return None
  # slices desc, price asc, free desc
  ranked.sort(key=lambda r: (-r[0], r[1], -r[2]))
  best = ranked[0]
  return best[3], best[0]


def plan_one(entry: QueueEntry,
             avail_by_cell: dict[str, CellAvail],
             now: float,
             legal_sizes: Optional[dict[str, list[int]]] = None,
             arch_price: Optional[dict[str, float]] = None,
             arch_pool: Optional[dict[str, float]] = None,
             ) -> Optional[Placement]:
  """Decide where to place ONE job, trying its accepted shapes best-first.

  Shapes are ordered by `candidate_shapes` -- effective-price-first when market
  data is supplied (select type by price+pool bonus), ARCH_PREF otherwise. For
  each shape in that order, the first cell that can actually place it wins.

  Returns a Placement, or None if nothing can place it right now (it stays
  QUEUED and is retried next tick -- we NEVER submit into an oversold/full
  cell, which is exactly the bug this whole system exists to avoid)."""
  for arch, chips in candidate_shapes(entry, legal_sizes, arch_price, arch_pool):
    hit = best_cell_for_shape(arch, chips, entry, avail_by_cell, now)
    if hit is not None:
      ca, n = hit
      geom = geometry_of(arch, chips)
      lock_note = ''
      if entry.topology_locked:
        lock_note = (f" [locked {entry.locked_geometry}]"
                     if entry.locked_geometry else f" [locking {geom}]")
      reason = (f"{arch}-{chips} -> {ca.cell} "
                f"({n} free slice(s)"
                + (f", {ca.price:.2f} cr/chip-hr" if ca.price is not None else "")
                + ")" + lock_note)
      return Placement(job_id=entry.job_id, arch=arch, chips=chips,
                       geometry=geom,
                       cell=ca.cell, price=ca.price, reason=reason)
  return None


def select_and_plan(
    entries: list[QueueEntry],
    avail_by_cell: dict[str, CellAvail],
    now: float,
    rng: Optional[random.Random] = None,
    max_placements: Optional[int] = None,
    legal_sizes: Optional[dict[str, list[int]]] = None,
    arch_price: Optional[dict[str, float]] = None,
    arch_pool: Optional[dict[str, float]] = None,
) -> list[Placement]:
  """Plan placements for a whole QUEUED batch, honoring priority + fairness.

  Order: higher priority first; among equal priority, RANDOM order (seeded rng
  for tests) so no job in a priority band starves. As each placement is chosen,
  its slices are DECREMENTED from the local availability copy so two jobs in one
  tick do not both get routed to the same cell and re-create the oversold
  (thundering herd). Only QUEUED entries are considered."""
  rng = rng or random.Random()
  # local mutable copy of availability so we can decrement within the tick
  local: dict[str, CellAvail] = dict(avail_by_cell)

  queued = [e for e in entries if e.state == JobState.QUEUED]
  # group by priority desc; shuffle within each band
  by_prio: dict[int, list[QueueEntry]] = {}
  for e in queued:
    by_prio.setdefault(e.priority, []).append(e)
  ordered: list[QueueEntry] = []
  for prio in sorted(by_prio.keys(), reverse=True):
    band = by_prio[prio][:]
    rng.shuffle(band)
    ordered.extend(band)

  placements: list[Placement] = []
  for entry in ordered:
    if max_placements is not None and len(placements) >= max_placements:
      break
    p = plan_one(entry, local, now, legal_sizes, arch_price, arch_pool)
    if p is None:
      continue
    placements.append(p)
    # Decrement the chosen (cell, arch)'s free chips so the next job in this
    # tick sees the draw-down. Matched by CONTENT, not by dict key: a cell can
    # host two accelerator generations (je carries both v6e and v7 pods), so
    # avail is keyed per (cell, arch) and we must not decrement the wrong one.
    for k, ca in local.items():
      if ca.cell == p.cell and ca.arch.lower() == p.arch.lower():
        local[k] = dataclasses.replace(
            ca, free_chips=max(0, ca.free_chips - p.chips))
        break
  return placements


def apply_placement(entry: QueueEntry, placement: Placement, xid: str,
                    now: float) -> QueueEntry:
  """Record a successful submit on the entry: SUBMITTED, with the concrete
  shape/cell/xid and the submit clock started. For a topology-locked job placed
  for the FIRST time, freeze `locked_geometry` from the chosen shape so every
  later re-route stays on the same mesh. The binary calls this right after
  `tpu queue` returns an XID.

  ★The previous XID is PRESERVED in `prior_xids`, never just overwritten: a row
  that is resubmitted (build retry, re-route) would otherwise erase the only
  local trace of an experiment that may still be running and billing."""
  entry.state = JobState.SUBMITTED
  if entry.xid and entry.xid != xid:
    if not entry.prior_xids:
      entry.prior_xids = []
    if entry.xid not in entry.prior_xids:
      entry.prior_xids.append(entry.xid)
  entry.xid = xid
  entry.cell = placement.cell
  entry.arch = placement.arch
  entry.chips = placement.chips
  entry.submitted_at = now
  if entry.topology_locked and entry.locked_geometry is None:
    entry.locked_geometry = placement.geometry
  entry.last_reason = placement.reason
  return entry


def needs_reroute(entry: QueueEntry, now: float, reroute_after_s: float) -> bool:
  """True if a SUBMITTED job has been pending past the re-route deadline.

  The core does NOT know the job's live XM status -- the binary passes that in
  by only calling this for jobs it has confirmed are still PENDING. Here we just
  own the CLOCK part of the rule: submitted, and older than the deadline."""
  if entry.state != JobState.SUBMITTED:
    return False
  if entry.submitted_at is None:
    return False
  return (now - entry.submitted_at) >= reroute_after_s


def output_is_fresh(latest_mtime: Optional[float], now: float,
                    fresh_within_s: float) -> bool:
  """True if the job's output dir shows a write within `fresh_within_s` of now.

  A job that is writing checkpoints/metrics RIGHT NOW is alive, whatever a
  single XManager snapshot momentarily says. This is the disk-evidence guard
  that stops a BATCH job -- whose EMA shadow work-units run in segments, so the
  XM layer can read `all pending` in the gap between two segments -- from being
  re-routed while it is in fact training (xid 282605596, 2026-08-24: XM showed
  all-pending in a shadow gap, but pass@2=0.5025 had just been written).

  Pure: the caller does the CNS stat and passes the newest mtime in (or None if
  the dir is missing / the stat failed -- in which case there is NO evidence of
  life here and we fall back to the two-sample probe, never judging alive on a
  failed lookup). Boundary is exclusive-of-stale: a write EXACTLY fresh_within_s
  ago is treated as stale (not fresh), so the deadline cannot make reroute a
  no-op."""
  if latest_mtime is None:
    return False
  return (now - latest_mtime) < fresh_within_s


def decide_reroute(first_state: str, second_state: Optional[str],
                   output_fresh: bool,
                   pending_const: str = 'PENDING') -> bool:
  """Pure decision: should this candidate ACTUALLY be re-routed?

  The safety invariant: this only ever returns True when EVERY guard has failed
  to find life. It never widens the reroute path -- each guard can only turn a
  would-be reroute OFF.

    1. first_state != PENDING  -> not reroute (RUNNING/TERMINAL/UNKNOWN handled
       by the caller's own branches; we never reroute a non-PENDING first read).
    2. output_fresh            -> not reroute (disk says it is alive NOW).
    3. second_state != PENDING -> not reroute (the shadow-gap cleared on the
       second sample; UNKNOWN also protects -- never cancel on ambiguity).
    4. both PENDING and no fresh output -> RE-ROUTE (double-confirmed stuck).

  `second_state` is None when the caller has not taken a second sample (e.g. the
  first read was already non-PENDING, or output was fresh so we short-circuited);
  in that case guards 1/2 must have decided, and reaching here with None means
  'do not reroute' (defensive)."""
  if first_state != pending_const:
    return False
  if output_fresh:
    return False
  if second_state is None:
    return False
  if second_state != pending_const:
    return False
  return True


def mark_reroute(entry: QueueEntry, now: float, cooldown_s: float) -> QueueEntry:
  """Return the entry reset to QUEUED after a failed placement, with the cell
  it was stuck in put on cooldown so the next plan avoids it for a while."""
  if entry.cell:
    entry.cooldown_cells[entry.cell] = now + cooldown_s
  # ★Preserve the XID before clearing it (infra-v17). A re-routed row gets
  # xid=None, and without this the cancelled experiment becomes invisible to
  # every audit that enumerates known XIDs (xid_recon/recon.py reads
  # prior_xids explicitly) -- i.e. a ghost car. record_submit already does
  # this on the re-submit path; the reroute path was missing it.
  if entry.xid:
    if not entry.prior_xids:
      entry.prior_xids = []
    if entry.xid not in entry.prior_xids:
      entry.prior_xids.append(entry.xid)
  entry.state = JobState.QUEUED
  entry.last_reason = (f"re-routed after pending in {entry.cell} "
                       f">{int((now - (entry.submitted_at or now)))}s")
  entry.xid = None
  entry.cell = None
  entry.arch = None
  entry.chips = None
  entry.submitted_at = None
  # ★NOT attempts: that counter feeds the 3-strikes build brake (route_check
  # parks a row HELD at max_build_attempts). A re-route is not a build failure
  # -- counting it there let an oversold-cell rotation park elt's cars with
  # zero real build failures (measured 22:1xZ: v3e had attempts=3, all from
  # re-routes). Track re-routes separately.
  entry.reroutes += 1
  return entry


# --- serial build-worker invariant ----------------------------------------
# The whole point of the worker: on THIS machine, two `tpu queue` builds that
# share a checkout race on the blaze output_base (-> found[] zombie work units)
# and a burst of concurrent stage-writes drains the CitC snapshot token bucket
# (-> truncated stagedir, .par crash). Both are cured by never running two
# builds at once. BUILDING is that lock, held IN the durable queue file so it
# survives across worker restarts and is visible to `tpu queue-status`.

def count_building(entries: list['QueueEntry']) -> int:
  """How many entries are currently BUILDING. The serial invariant is that this
  never exceeds 1; the worker checks it before claiming the next job."""
  return sum(1 for e in entries if e.state == JobState.BUILDING)


def building_is_stale(entry: QueueEntry, now: float, stale_after_s: float) -> bool:
  """True if a BUILDING claim is older than stale_after_s -- i.e. the worker that
  claimed it died mid-build (a build is minutes, never an hour). Such a claim
  must be reclaimed or the queue wedges forever holding the single-build slot."""
  if entry.state != JobState.BUILDING:
    return False
  started = entry.build_started_at
  if started is None:
    return True   # BUILDING with no timestamp is already corrupt -- reclaim it
  return (now - started) >= stale_after_s


def reclaim_stale_building(entries: list['QueueEntry'], now: float,
                           stale_after_s: float) -> list['QueueEntry']:
  """Reset any stale BUILDING entry back to QUEUED (worker crashed mid-build).
  Returns the list of entries reclaimed. Frees the single-build slot."""
  reclaimed = []
  for e in entries:
    if building_is_stale(e, now, stale_after_s):
      e.state = JobState.QUEUED
      e.build_started_at = None
      e.worker_id = None
      e.attempts += 1
      e.last_reason = f'reclaimed: BUILDING claim went stale (>{int(stale_after_s)}s)'
      reclaimed.append(e)
  return reclaimed


def next_queued(entries: list['QueueEntry']) -> Optional['QueueEntry']:
  """The next QUEUED entry to build, highest priority first then FIFO-ish by
  list order. Returns None if nothing is queued. Does NOT consider whether a
  build is already in flight -- the caller enforces the single-build invariant."""
  queued = [e for e in entries if e.state == JobState.QUEUED]
  if not queued:
    return None
  # highest priority wins; ties keep insertion order (stable sort)
  return max(queued, key=lambda e: e.priority) if len(queued) > 1 else queued[0]


def claim_for_build(entry: QueueEntry, now: float, worker_id: str) -> QueueEntry:
  """Mark an entry BUILDING (the worker is about to run `tpu queue` for it).
  Records who claimed it and when, so a crashed claim can be detected as stale."""
  entry.state = JobState.BUILDING
  entry.build_started_at = now
  entry.worker_id = worker_id
  entry.last_reason = f'building (worker {worker_id})'
  return entry


def can_claim_build(entries: list['QueueEntry'], now: float,
                    stale_after_s: float) -> bool:
  """True iff the worker may start a new build: no LIVE (non-stale) BUILDING
  entry holds the single-build slot. A stale claim does not count (it will be
  reclaimed first). This is the serial invariant, enforced on the durable
  queue so even two worker processes cannot both build at once."""
  for e in entries:
    if e.state == JobState.BUILDING and not building_is_stale(e, now, stale_after_s):
      return False
  return True


# --- HELD: park a job that cannot build as-is, instead of churning ----------
# An unattended serial worker must not spin on a bad job. Two causes park a job
# in HELD (skipped by next_queued until a human re-enqueues): a workdir that is
# set but does not exist (definitely broken -- the wrong source would be
# packaged or the build fails), and too many failed build attempts (the
# empty-workdir-that-fails case, caught without guessing whether the config is
# flag-resolvable). HELD is recoverable, not terminal.

def hold_entry(entry: QueueEntry, reason: str) -> QueueEntry:
  """Park an entry in HELD with a human-readable reason. Frees the build slot."""
  entry.state = JobState.HELD
  entry.build_started_at = None
  entry.worker_id = None
  entry.last_reason = f'HELD: {reason}'
  return entry


def requeue_held(entry: QueueEntry) -> QueueEntry:
  """Return a HELD entry to QUEUED (a human fixed it / wants a retry). Resets the
  attempt counter so it gets a fresh run of tries."""
  if entry.state == JobState.HELD:
    entry.state = JobState.QUEUED
    entry.attempts = 0
    entry.last_reason = 'requeued from HELD'
  return entry


# --- BUILD_REQUESTED + backpressure (router-dispatch / serial-builder split) --
# The rewritten worker separates the DISPATCH decision (router half: pick jobs
# that fit headroom this round, mark them BUILD_REQUESTED) from the BUILD action
# (builder half: serially claim each BUILD_REQUESTED, run `tpu queue`). The two
# halves live in ONE process but are distinct phases; BUILD_REQUESTED is the
# handoff token between them, held in the durable queue so it survives a restart.

def mark_build_requested(entry: QueueEntry, reason: str = '') -> QueueEntry:
  """Router dispatch: mark a QUEUED entry BUILD_REQUESTED (this round's builder
  will claim it). Transient; does NOT touch attempts or timestamps (the builder
  stamps build_started_at when it claims). Idempotent for an already-requested
  entry."""
  entry.state = JobState.BUILD_REQUESTED
  entry.last_reason = reason or 'dispatch: fits headroom this round; queued for build'
  return entry


def count_build_pending(entries: list['QueueEntry']) -> int:
  """How many entries are BUILD_REQUESTED or BUILDING = 'the builder has not yet
  drained this round'. Backpressure gate: the router must NOT open a new dispatch
  round while this is > 0, or its headroom query would double-count jobs that are
  dispatched-but-not-yet-in-the-billing-registry."""
  return sum(1 for e in entries
             if e.state in (JobState.BUILD_REQUESTED, JobState.BUILDING))


def next_build_requested(entries: list['QueueEntry']) -> Optional['QueueEntry']:
  """The next BUILD_REQUESTED entry for the serial builder to claim, highest
  priority first then insertion order (mirrors next_queued). Returns None if the
  builder has drained this round. Does NOT enforce the single-build invariant --
  the caller checks can_claim_build first."""
  reqd = [e for e in entries if e.state == JobState.BUILD_REQUESTED]
  if not reqd:
    return None
  return max(reqd, key=lambda e: e.priority) if len(reqd) > 1 else reqd[0]


# --- BUDGET_DEFERRED (soft, auto-recovering budget park) --------------------
# A budget refusal is a transient FLEET state, not a per-job defect. Instead of
# counting it as a build failure (attempts++ -> 3-strikes -> permanent HELD), the
# worker parks the job BUDGET_DEFERRED and re-tests it every round. When headroom
# opens (a running job ends / income rises / a big job ahead clears) it flows back
# to QUEUED automatically -- zero human action. attempts is NEVER touched here.

def mark_budget_deferred(entry: QueueEntry, reason: str = '') -> QueueEntry:
  """Park an entry BUDGET_DEFERRED (over the credit bar this round). NOT HELD,
  NOT a build failure: attempts is deliberately NOT incremented. Frees the build
  slot. Re-evaluated next round by promote_deferred."""
  entry.state = JobState.BUDGET_DEFERRED
  entry.build_started_at = None
  entry.worker_id = None
  entry.last_reason = reason or 'budget-deferred: over the G9 credit bar this round; will retry when headroom opens'
  return entry


def promote_deferred(entries: list['QueueEntry']) -> list['QueueEntry']:
  """Top-of-round: return every BUDGET_DEFERRED entry to QUEUED so it is re-tested
  against the CURRENT headroom this round. Returns the list of promoted entries.
  attempts is untouched (a deferral was never a failure). This makes the deferral
  a pure per-round re-test: if it is still over the bar, the dispatch step lands
  it back in BUDGET_DEFERRED; if headroom opened, it dispatches."""
  promoted = []
  for e in entries:
    if e.state == JobState.BUDGET_DEFERRED:
      e.state = JobState.QUEUED
      e.last_reason = 'promoted from budget-deferred: re-testing headroom this round'
      promoted.append(e)
  return promoted


# --- XM-truth reconcile (fixes R3 zombie pollution on the ROUTE path) --------
# .tpu_local_queue.json keeps state=RUNNING/SUBMITTED for jobs XManager no longer
# tracks (measured 2026-08-27: 13 local-RUNNING, 0 truly running on XM). Any
# headroom/reroute logic that reads local `state` is poisoned. The reconcile pass
# (in the tpu-reroute process) re-verifies every RECONCILABLE_STATES entry's XID
# against XM and rewrites zombies to terminal. This decision is PURE (XM status is
# an injected string, so route_lib stays RPC-free); the binary's probe supplies it.

def decide_reconcile(local_state: 'JobState', xm_status: str,
                     terminal_const: str = 'TERMINAL',
                     running_const: str = 'RUNNING',
                     pending_const: str = 'PENDING',
                     unknown_const: str = 'UNKNOWN',
                     completed_const: str = 'COMPLETED'
                     ) -> Optional['JobState']:
  """Pure reconcile decision for ONE entry, given its local state and XM's live
  status. Returns the NEW JobState to write, or None for 'leave unchanged'.

  Safety rule (mirrors decide_reroute): NEVER act on missing data. xm_status
  UNKNOWN (probe failed) -> None: we do not mark a job dead because a probe
  hiccuped. Only a definite XM verdict moves an entry.

    XM COMPLETED -> DONE       (it RAN TO COMPLETION -- a success, not a zombie)
    XM TERMINAL  -> FAILED     (zombie cleanup: XM dropped it / it failed/stopped)
    XM RUNNING   & local SUBMITTED -> RUNNING  (placement took; promote)
    XM RUNNING   & local RUNNING   -> None      (already correct)
    XM PENDING   -> None       (genuinely still queued in the auction; the reroute
                                step -- not reconcile -- owns pending>deadline)
    XM UNKNOWN   -> None       (never act blind)
  A local entry not in RECONCILABLE_STATES is never passed here (caller filters).

  COMPLETED vs TERMINAL: these were a single status until it was measured that
  105 of 227 queue rows had been reconciled to FAILED, 100% of them, including
  jobs whose results were already in use. A finished job is DONE. Only the probe
  can tell the two apart, so this function is only as correct as the status it
  is handed -- an unrecognised status still falls through to no-op, and UNKNOWN
  is still never actioned.
  """
  if xm_status == completed_const:
    return JobState.DONE
  if xm_status == terminal_const:
    return JobState.FAILED
  if xm_status == running_const and local_state == JobState.SUBMITTED:
    return JobState.RUNNING
  # RUNNING+RUNNING, any PENDING, any UNKNOWN, or an unrecognised status: no-op.
  return None


def reconcile_entry(entry: QueueEntry, xm_status: str, reason: str = '') -> bool:
  """Apply decide_reconcile to one entry in place. Returns True if the entry's
  state changed (a zombie was cleaned up or a placement promoted), False if left
  unchanged. Only touches entries currently in RECONCILABLE_STATES."""
  if entry.state not in RECONCILABLE_STATES:
    return False
  new_state = decide_reconcile(entry.state, xm_status)
  if new_state is None or new_state == entry.state:
    return False
  old = entry.state
  entry.state = new_state
  if new_state == JobState.DONE:
    entry.last_reason = reason or f'reconciled: XM reports COMPLETED (was local {old.value}); finished normally'
  elif new_state == JobState.FAILED:
    entry.last_reason = reason or f'reconciled: XM reports terminal (was local {old.value}); zombie cleaned up'
  elif new_state == JobState.RUNNING:
    entry.last_reason = reason or f'reconciled: XM confirms RUNNING (was local {old.value})'
  return True


# --- Step3: greedy dispatch with in-memory pre-debit (router half) ----------
# The rewritten worker's DISPATCH decision. Each round, with a fresh XM-truth
# headroom, admit the highest-priority QUEUED jobs that FIT, pre-debiting each
# admitted job's cost IN MEMORY so a big job at the head cannot be "admitted"
# twice and small jobs behind it still get their turn. Over-the-bar jobs are
# deferred (soft), never failed. This is PURE: cost + exemption + headroom are
# all injected, so it is unit-testable without budget_check, RPCs, or a clock.

@dataclasses.dataclass(frozen=True)
class DispatchDecision:
  """One dispatch verdict for one entry this round."""
  job_id: str
  decision: 'JobState'          # BUILD_REQUESTED (fits) or BUDGET_DEFERRED (over)
  cost: float                   # the job's credit cost (0 for exempt)
  headroom_after: float         # headroom remaining after this decision
  reason: str = ''


def plan_dispatch(
    queued: list['QueueEntry'],
    headroom: float,
    cost_of: Callable[['QueueEntry'], float],
    is_exempt: Callable[['QueueEntry'], bool],
) -> list[DispatchDecision]:
  """Greedy-fill this round's headroom with in-memory pre-debit.

  `queued` is the set of dispatch candidates (the caller passes QUEUED entries).
  `headroom` is the CURRENT XM-truth headroom (bar - current usage) for the round.
  `cost_of(e)` returns the job's credit/hr cost; `is_exempt(e)` is True for jobs
  that do not draw on the G9 bar (g3/g5 own balance, BATCH free pool, CPU-only).

  Returns one DispatchDecision per candidate, in priority order:
    * exempt            -> BUILD_REQUESTED, no debit (headroom unchanged)
    * cost <= headroom  -> BUILD_REQUESTED, headroom -= cost   (PRE-DEBIT)
    * else              -> BUDGET_DEFERRED (soft; re-tested next round)
  Priority: highest e.priority first, ties keep list order (stable). A big job
  that does not fit is deferred but does NOT block smaller jobs behind it -- the
  loop continues, so a later cheaper job can still be admitted (fixes H2's
  head-of-line starving). Pure: no mutation of the entries, no I/O.
  """
  ordered = sorted(queued, key=lambda e: -e.priority)   # stable within priority
  out: list[DispatchDecision] = []
  h = headroom
  for e in ordered:
    if is_exempt(e):
      out.append(DispatchDecision(
          e.job_id, JobState.BUILD_REQUESTED, 0.0, h,
          'exempt (g3/g5/BATCH/CPU): dispatched without debit'))
      continue
    c = cost_of(e)
    if c <= h:
      h -= c
      out.append(DispatchDecision(
          e.job_id, JobState.BUILD_REQUESTED, c, h,
          f'fits: cost {c:.1f} <= headroom; pre-debited, {h:.1f} left this round'))
    else:
      out.append(DispatchDecision(
          e.job_id, JobState.BUDGET_DEFERRED, c, h,
          f'over bar: cost {c:.1f} > headroom {h:.1f}; deferred, retry next round'))
  return out
