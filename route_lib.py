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
from typing import Optional


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
}

# Newer-first preference among equally-good architectures (mirrors _ARCH_PREF).
ARCH_PREF: dict[str, int] = {
    'v7': 0, 'v6p': 1, 'v6e': 2, 'v5p': 3, 'v4': 4, 'v5e': 5}

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
  SUBMITTED = 'SUBMITTED'  # handed to XM, watching for RUNNING vs re-route
  RUNNING = 'RUNNING'      # confirmed running; the router is done with it
  DONE = 'DONE'            # finished (terminal)
  FAILED = 'FAILED'        # gave up / user cancelled (terminal)


TERMINAL_STATES = frozenset({JobState.RUNNING, JobState.DONE, JobState.FAILED})


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
  # ---- mutable state ----
  state: JobState = JobState.QUEUED
  xid: Optional[str] = None         # set once submitted
  cell: Optional[str] = None        # cell it was placed into
  arch: Optional[str] = None        # concrete arch chosen
  chips: Optional[int] = None       # concrete chip count chosen
  submitted_at: Optional[float] = None   # epoch when handed to XM
  attempts: int = 0                 # placement attempts so far
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
  `tpu queue` returns an XID."""
  entry.state = JobState.SUBMITTED
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


def mark_reroute(entry: QueueEntry, now: float, cooldown_s: float) -> QueueEntry:
  """Return the entry reset to QUEUED after a failed placement, with the cell
  it was stuck in put on cooldown so the next plan avoids it for a while."""
  if entry.cell:
    entry.cooldown_cells[entry.cell] = now + cooldown_s
  entry.state = JobState.QUEUED
  entry.last_reason = (f"re-routed after pending in {entry.cell} "
                       f">{int((now - (entry.submitted_at or now)))}s")
  entry.xid = None
  entry.cell = None
  entry.arch = None
  entry.chips = None
  entry.submitted_at = None
  entry.attempts += 1
  return entry
