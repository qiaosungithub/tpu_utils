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
import os
import random
import re
from typing import Callable, Optional, Sequence


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


# GPU arch tokens (mirrors the GPU rows of V5P_MULTIPLIER / ARCH_PREF and
# topology._GPU_LEGAL). Kept as a local frozenset so this module stays free of
# any import that drags in RPC stubs; the binary reconciles it with topology.
_GPU_ARCHS: frozenset[str] = frozenset({
    'a100', 'a100_80gib', 'h100', 'h200', 'b200', 'b300', 'gb200', 'gb300'})


def is_gpu(arch: str) -> bool:
  """True if `arch` names an NVIDIA GPU rather than a TPU."""
  return (arch or '').strip().lower() in _GPU_ARCHS


def parse_gpu_shape(power: str) -> Optional[tuple[str, int]]:
  """('h100', 8) if `power` names a concrete GPU slice, else None.

  ★Must be consulted BEFORE parse_power, which collapses any spec to a
  v5p-equivalent float and thereby throws away the one fact that decides the
  whole route: that the caller asked for a GPU BOARD of a fixed width. Once
  'h100-8' has become 17.2 v5p-equiv, nothing downstream can tell it from a TPU
  ask for the same compute, and the compute-equivalence window will rescale a
  b200 down to 4 chips (b200-4 = 19.6 ~ h100-8's 17.2) -- half a board for a
  job that asked for a whole one. See candidate_shapes for how this is used.
  """
  s = str(power).strip().lower()
  for sep in ('-', '='):
    if sep in s:
      arch, cores = s.split(sep, 1)
      if is_gpu(arch):
        try:
          return arch, int(cores)
        except ValueError:
          return None
      return None
  return None


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
# win even when nominally a bit more expensive -- worth up to POOL_BONUS price
# forgiveness for a big pool. Pool size MOVES DAILY, so it is never hardcoded:
# the binary passes the LIVE per-arch pool magnitude (sum of obtainable/free
# chips across the job's allowed metros) computed from the same availability
# data the cell step uses.
#
# ★THIS IS ONE OF TWO SCORES, AND THEY ANSWER DIFFERENT QUESTIONS. This one
# picks the ARCH (v7 vs v6p vs v5p) and reads ONE price for the whole arch.
# `best_cell_for_shape` then picks the CELL inside that arch and reads each
# cell's OWN price. Same shape on purpose -- price on top, a log-scaled bonus
# with a hard ceiling underneath -- so "how much extra am I willing to pay for
# a non-price virtue" is always a number somebody can say out loud.
# --------------------------------------------------------------------------
# Operator 2026-08-31: raised 0.20 -> 0.50. A deep pool may now win against a
# type up to 1.5x its price; below that the log scaling makes the reward taper.
POOL_BONUS = 0.50   # max fraction of price a large pool is forgiven
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
  were 1/1.5 ~= 33% cheaper, floating a large-pool type above a marginally
  cheaper but thin one. Lower is better.

  ARCH-LEVEL ONLY: `raw_price` here is the one global clearing price for the
  arch. The per-CELL prices inside an arch (which differ by up to 3.2x -- v6e
  measured at 15.999 and 51.923 the same minute) are the cell step's business,
  not this one's.
  """
  return raw_price / pool_weight(pool_chips)


# --------------------------------------------------------------------------
# CELL SELECTION: cheapest-effective-first again, one level down.
#
# The arch is already chosen; now pick a cell inside it. Same shape as the arch
# score -- price on top, a log-scaled bonus with a hard ceiling underneath --
# but every input is per-cell.
#
#   score = cell_price / slice_weight(n_slices) * evict_penalty(cell)
#
# WHY IT IS NOT `sort(-n_slices, price, ...)` ANY MORE. That was a lexicographic
# sort: roominess first, price consulted only on an exact tie. It was harmless
# only because every cell of an arch carried the SAME price (the provider read
# `global` and copied it to all of them), so the price key never fired. The
# moment per-cell prices arrive, that order reads as "pay 3.2x to fit one more
# slice" -- the data fix ALONE would have made placement worse, so the two
# changes have to ship together.
# --------------------------------------------------------------------------
# Max fraction of price we forgive for a roomy cell. 0.50 => a cell with 8+
# placeable slices may win against one up to 1.5x cheaper, and no further: on
# v6e's real spread (15.999 vs 51.923, 3.2x) roominess can NEVER buy the dear
# cell, while a 1.3x gap (16.894 vs 22.08) can be outweighed. That is the trade
# being priced -- how much extra for a cell I am less likely to be squeezed out
# of -- and it is now a number rather than an implicit absolute preference.
SLICE_BONUS = 0.50
# Slice count at/above which the FULL bonus applies; below it the reward is
# log-scaled, so 1 slice earns 0, 2 earns ~1/3, 4 ~2/3, 8 all of it. Past 8
# there is nothing left to buy: a cell that fits 8 copies of the job is not
# meaningfully safer than one that fits 30.
SLICE_FULL_BONUS_N = 8.0
# Multiplier added per recorded eviction from a cell, decaying linearly to zero
# over EVICT_DECAY_S. 1.0 => one eviction makes the cell read as DOUBLE price
# immediately after: "just threw me out" is worth about as much as "costs twice
# as much". Two strikes ~= 3x, effectively exclusion -- but SOFT: when every
# candidate has struck, the least-recent offender still wins, whereas the old
# hard cooldown could leave a job with nowhere to go at all.
EVICT_STRIKE_WEIGHT = 1.0
# Evictions age out in half an hour. Preemption describes the market at a
# moment, not a property of the cell, so a strike must not outlive the
# conditions that produced it: yesterday's squeeze is not evidence about now.
EVICT_DECAY_S = 1800.0


def slice_weight(n_slices: int) -> float:
  """Multiplier (1.0 .. 1+SLICE_BONUS) rewarding a roomier cell.

  Log-scaled to SLICE_FULL_BONUS_N, mirroring `pool_weight` one level up.
  n<=1 earns nothing -- a cell that fits the job exactly once is the baseline.
  """
  import math
  if n_slices <= 1:
    return 1.0
  frac = math.log1p(n_slices - 1) / math.log1p(SLICE_FULL_BONUS_N - 1)
  frac = max(0.0, min(1.0, frac))
  return 1.0 + SLICE_BONUS * frac


def evict_penalty(strikes: int, since_s: Optional[float]) -> float:
  """Multiplier (>=1.0) penalising a cell that recently evicted THIS job.

  `strikes` = how many times this job was preempted out of the cell,
  `since_s` = seconds since the most recent one (None = never / unknown).
  Decays linearly to 1.0 at EVICT_DECAY_S.

  Returns 1.0 when there is no eviction record -- which is EVERY cell until
  preemption detection exists to write one. The multiplier is inert by
  construction today, deliberately: the ranking change and its data source ship
  separately so the sort can be verified on its own. Read this as "wired, not
  yet fed", never as "working".
  """
  if strikes <= 0 or since_s is None or since_s < 0:
    return 1.0
  decay = max(0.0, 1.0 - (since_s / EVICT_DECAY_S))
  return 1.0 + EVICT_STRIKE_WEIGHT * strikes * decay


# ★Cooldown is a PENALTY, not a gate (operator 2026-09-01 14:40Z). It used to
# be a hard `continue` in best_cell_for_shape: re-routing a job excluded the
# cell it had just been in for the full cooldown, so the router could not pick
# that cell back even when it had become the best choice a minute later. That
# is the wrong shape for a mechanism whose whole purpose is "re-pick the best
# cell every 10 minutes" -- an exclusion removes the option instead of ranking
# it, and when every candidate is cooling the gate yields NO choice at all
# while a penalty still yields the least-bad one.
# Same construction as evict_penalty: multiplier >= 1.0, decaying linearly to
# 1.0 at the end of the cooldown window.
COOLDOWN_WEIGHT = 1.0


def cooldown_penalty(until: Optional[float], now: float,
                     cooldown_s: float = 1800.0) -> float:
  """Multiplier (>=1.0) for a cell this job was recently re-routed out of.

  `until` is the epoch the cooldown expires (what mark_reroute stores). Returns
  1.0 once expired or absent, so a cell with no history is never penalised.
  The penalty is strongest right after the re-route and fades to nothing as the
  window closes -- so a cell can win again the moment it is genuinely better,
  which the old hard gate made impossible.
  """
  if not until or now >= until:
    return 1.0
  remaining = min(max(until - now, 0.0), cooldown_s)
  return 1.0 + COOLDOWN_WEIGHT * (remaining / cooldown_s)


# ---------------------------------------------------------------------------
# ARCH-LEVEL cooldown -- the SECOND, coarser cooldown (operator 2026-09-10).
#
# WHY IT EXISTS, on top of the per-cell one above. `cooldown_cells` penalises
# the ONE borg cell a job was just re-routed out of. But an arch has many cells
# (v4 in tul: nm, nf, oe, oa...), so cooling `nm` just sends the router to `nf`,
# and the v4 GENERATION is never penalised -- the job keeps fake-landing on v4
# and bouncing, never trying v5p/v7. This penalises the whole arch instead, so a
# job that keeps being knocked off v4 is de-preferred UP the type ladder.
#
# IT IS AN ADDITIVE SURCHARGE ON THE SORT PRICE, KEYED TO THE ARCH'S LIMIT-ORDER
# CAP (operator 2026-09-10). Per strike it adds `pct * cap(arch)` to the price
# the type-ranking sees. Three reasons for this exact shape, each a correction
# of the first (multiplicative) cut:
#   * ADDITIVE, not a multiplier: a job feels a bounded nudge, not a runaway
#     factor -- the operator found the 2x..7x multiplier too harsh.
#   * KEYED TO THE CAP, not to the live price: a multiplier `k*price` collapses
#     to ZERO when the pool cleared free that cycle (price 0), so the free-but-
#     unusable arch that MOST needs pushing off gets no penalty at all. The cap
#     is a fixed non-zero constant per family (cap_policy.CAP_POLICY), so the
#     surcharge is well-defined even at price 0.
#   * SORT-ONLY: the surcharge is added to the ranking key in candidate_shapes,
#     NEVER to `ca.price`. The real-price limit-order gate in best_cell_for_shape
#     keeps seeing the true price, so a penalised arch is DE-PREFERRED, never
#     price-capped out. The cap policy is borrowed only as a stable UNIT of
#     surcharge here -- it does not become a second gate. (operator red line:
#     the limit order caps the REAL price only.)
#
# STACKS across re-routes (strikes) and is FLAT across the window then off at
# `until` -- a decaying window expired before the backlogged serial build-worker
# re-dispatched the job, so it read as un-penalised at the one moment it
# mattered. Capped at ARCH_COOLDOWN_MAX_STRIKES strikes (the operator's penalty
# ceiling).
#
# SIZING at the default 15% (real arch-global PROD prices 2026-09-10, caps in
# parens): v4 4.29 (cap 5), v6p 13.94 (cap 20), v7 35.74 (cap 20). Each v4 strike
# adds 0.15*5 = 0.75, so v4 reads 4.29 -> 5.04 -> 5.79 -> 6.54 -> 7.29 (capped).
# That is a deliberate GENTLE nudge: at 15% a stuck v4 is de-ranked but does not
# overtake v6p, because v4's own cap (5) is small. To make a stuck arch actually
# cross to the next generation, raise the percentage via the env knob below
# (~0.50 makes v4 cross v6p by the 4th strike); it is env-tunable precisely so
# the aggressiveness can be retuned with a worker restart, no rebuild.
ARCH_COOLDOWN_STRIKE_PCT_DEFAULT = 0.15
# Strikes past this add nothing -- the operator's ceiling on the surcharge.
ARCH_COOLDOWN_MAX_STRIKES = 4
# Env override for the per-strike percentage, so the knob is retunable without a
# rebuild (just restart the workers). Unset/blank/garbage/negative -> default.
ARCH_COOLDOWN_PCT_ENV = 'TPU_ARCH_COOLDOWN_PCT'


def _arch_cooldown_pct() -> float:
  """Per-strike surcharge as a fraction of the arch's limit-order cap.

  Reads `TPU_ARCH_COOLDOWN_PCT` so the aggressiveness is retunable by restarting
  the workers, not rebuilding. Never raises: an unset, blank, non-numeric, or
  negative value falls back to ARCH_COOLDOWN_STRIKE_PCT_DEFAULT.
  """
  raw = os.environ.get(ARCH_COOLDOWN_PCT_ENV)
  if raw is None or not str(raw).strip():
    return ARCH_COOLDOWN_STRIKE_PCT_DEFAULT
  try:
    v = float(raw)
  except (TypeError, ValueError):
    return ARCH_COOLDOWN_STRIKE_PCT_DEFAULT
  return v if v >= 0.0 else ARCH_COOLDOWN_STRIKE_PCT_DEFAULT


def arch_cooldown_surcharge(strikes: int, until: Optional[float], now: float,
                            cap: Optional[float]) -> float:
  """ADDITIVE penalty (>=0.0) ON THE SORT PRICE for an ARCH this job keeps being
  re-routed off. Added to the arch's effective price when ranking types; it is
  NOT a multiplier and NOT a price cap.

  = min(strikes, MAX) * pct * cap, where `cap` is the arch's limit-order cap (a
  stable, non-zero constant per family) and `pct` comes from `_arch_cooldown_pct`.
  Returns 0.0 when expired, absent, no strikes, or the arch has no cap policy --
  in every one of those the arch is simply un-penalised.

  FLAT across the window (full value until `until`, then 0.0), because a decaying
  penalty expired before the backlogged serial build-worker re-dispatched the
  job. Keyed to the CAP not the price so it survives a price-0 clearing cycle.
  """
  if not until or now >= until or strikes <= 0 or not cap or cap <= 0:
    return 0.0
  eff = min(strikes, ARCH_COOLDOWN_MAX_STRIKES)
  return eff * _arch_cooldown_pct() * cap


def arch_cooldown_surcharge_for(entry: 'QueueEntry', arch: str,
                                now: float) -> float:
  """Additive arch-cooldown surcharge for `arch` on `entry` at `now`.

  Reads the `{arch: {'until', 'strikes'}}` record `mark_reroute` writes and the
  arch's limit-order cap (`_family_price_cap`, i.e. cap_policy). Returns 0.0 for
  any arch with no live record (the common case) or no cap policy, so it is
  inert for a job that has never been re-routed. Tolerant of an absent/old-schema
  field.
  """
  rec = (getattr(entry, 'cooldown_archs', None) or {}).get(arch.lower())
  if not rec:
    return 0.0
  return arch_cooldown_surcharge(int(rec.get('strikes', 0) or 0),
                                 rec.get('until'), now,
                                 _family_price_cap(arch))


def cell_score(price: Optional[float], n_slices: int,
               strikes: int = 0, since_s: Optional[float] = None,
               cooldown_until: Optional[float] = None,
               now: Optional[float] = None,
               cooldown_s: float = 1800.0) -> float:
  """The cell-selection score. LOWER IS BETTER.

      price / slice_weight(n) * evict_penalty(...) * cooldown_penalty(...)

  An unknown price sorts LAST (inf), never first: a cell we cannot cost is not
  a bargain, and treating a missing number as zero is exactly how the
  cheapest-looking option becomes the one nobody could put a price on.
  """
  if price is None:
    return float('inf')
  cd = 1.0
  if cooldown_until is not None and now is not None:
    cd = cooldown_penalty(cooldown_until, now, cooldown_s)
  return ((price / slice_weight(n_slices))
          * evict_penalty(strikes, since_s) * cd)


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
  # ★THE ENQUEUE-TIME SNAPSHOT: a frozen LOCAL copy of `workdir`, made the
  # instant `tpu enqueue` ran. The build happens minutes-to-hours later, and
  # `tpu queue` packages whatever directory it is pointed at AT BUILD TIME -- so
  # without this, editing the source tree between enqueue and build silently
  # changes the code a queued job ships (a parked row "fires against a moved-on
  # checkout"). Freezing at enqueue closes that window: the builder packages
  # THIS directory, not the live `workdir`. See package_dir() for the read side.
  # Empty = enqueued before this existed (or `tpu enqueue --no_snapshot`): fall
  # back to `workdir` and package the live checkout, exactly as before -- which
  # is what keeps this forward/backward compatible with rows AND with a worker
  # binary that predates the field (its from_dict drops the key it never knew,
  # its builder packages workdir, no regression).
  snapshot_dir: str = ''
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
  # ★SET BY reclaim_stale_building, CLEARED BY route_check.adopt_escaped_builds.
  # Non-None means: this row's BUILDING claim went stale, so the build MIGHT have
  # succeeded and left a live experiment behind under this name. Until the
  # reconcile pass has looked that name up on XManager, the row is NOT claimable
  # -- re-dispatching it would put a second writer on the first one's output
  # path. Holding a row out of the build queue for one reconcile pass is cheap;
  # two jobs writing one checkpoint path is silent and unrecoverable.
  adopt_check_name: Optional[str] = None
  # ★Which alloc group the ROUTER admitted this job under (operator 00:05Z:
  # "我反复要求过优先用 G5 / G3"). The dispatch and build stages are separate
  # processes, so the group chosen while checking budget must be carried ON THE
  # ROW -- otherwise the builder falls back to its global --group (9) and every
  # car lands on the one pool that has a hard 1/10-of-income ceiling, while g5/g3
  # sit idle. None = not yet decided; the builder then uses its own default.
  group: Optional[str] = None
  # ★A CALLER-REQUESTED PIN on the alloc group -- the one thing `group` above is
  # not. `group` is the ROUTER's answer ("which pool did I admit this under");
  # `pin_group` is the caller's QUESTION ("submit this one under g9, whatever
  # your preference order says"). Two fields because they answer to different
  # owners: the router overwrites its own answer every round, so a pin stored
  # there would be erased by the next dispatch pass.
  # WHY IT EXISTS: the preference order ['5','3','9'] is right for the fleet and
  # wrong for a specific job -- g5 is vqfree-xm, whose GHOSTFISHLITE allotment
  # GQM can decline outright, and a job that keeps being preempted there needs
  # the g9 floor even though g9 bills against the hard income/10 bar. Before
  # this, no caller could express that: `tpu enqueue --group=` was consumed by
  # absl and never reached the row, and a launch_kwargs['group'] copy was
  # dropped at build time (correctly -- emitting it appended a SECOND --group=).
  # UNSET (None) = no pin: the router's preference order decides, unchanged.
  pin_group: Optional[str] = None
  # ★EVERY XID THIS ROW HAS EVER HELD, oldest first, excluding the current one.
  # `xid` is overwritten on each placement, so without this a resubmitted row
  # leaves its earlier experiment with NO local record anywhere -- which reads
  # exactly like an orphan to any XM->local audit, and is unfindable when you
  # need to stop it. A row with attempts>0 and an empty prior_xids is itself a
  # signal: the history predates this field.
  prior_xids: list[str] = dataclasses.field(default_factory=list)
  cooldown_cells: dict = dataclasses.field(default_factory=dict)  # cell -> until-epoch
  # ARCH-LEVEL cooldown: {arch: {'until': epoch, 'strikes': int}} -- the coarser
  # sibling of `cooldown_cells`. Where cooldown_cells penalises the ONE cell a
  # re-route left, this penalises the whole ARCH (v4, v6p...), so a job that
  # keeps fake-landing on a cheap-but-unusable generation is pushed UP the type
  # ladder instead of just hopping to the next cell of the same arch. Written by
  # `mark_reroute` (strikes STACK within the window, reset once expired), read
  # by `candidate_shapes` via `arch_cooldown_surcharge_for`. Backward
  # compatible: a row from before this field defaults to {} and the penalty is
  # inert.
  cooldown_archs: dict = dataclasses.field(default_factory=dict)
  # PER-CELL EVICTION HISTORY: {cell: {'strikes': int, 'last': epoch}} -- how
  # often THIS job has been preempted out of THAT cell, and when last. Read by
  # `cell_score` via `evict_penalty`, which decays a strike to nothing over
  # half an hour, so the record is a fading hint and never a permanent verdict.
  #
  # RELATED TO `cooldown_cells`, which the reroute path sets on the ONE cell a
  # job was stuck in. Both are now SOFT multipliers on cell_score (cooldown was
  # a hard gate until 2026-09-01). This one is cumulative and per-cell:
  # a cell that threw the job out twice sorts worse than one that did it once,
  # which a boolean cooldown cannot express -- and when every candidate has
  # struck, a soft penalty still yields a choice where the gate yields none.
  #
  # NOTHING WRITES THIS YET. Preemption detection is a separate change; until
  # it lands every lookup misses and `evict_penalty` returns 1.0, leaving
  # placement decided by price and roominess exactly as this CL ships it. The
  # field exists so the ranking can be reviewed and tested before the detector
  # starts feeding it -- but do not read "the penalty is implemented" as "the
  # penalty is happening".
  evictions: dict = dataclasses.field(default_factory=dict)
  last_reason: str = ''             # why it is where it is (for status view)
  # ★Why the router found NO cell this pass, when the cause was a hard gate
  # rather than an empty fleet. Diagnostic only: nothing reads it to make a
  # decision, and it is deliberately NOT persisted as state -- it describes one
  # pass, and a stale copy would read as a live verdict.
  last_filter_reason: str = ''
  # ★How many times the rule set has already re-dispatched this row. Persisted,
  # because the budget is meaningless if it resets whenever the process does.
  auto_resumes: int = 0

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


def pinned_group(entry: 'QueueEntry') -> Optional[str]:
  """The caller's alloc-group PIN for this row, or None if it did not ask.

  Two spellings, one meaning -- both are the caller saying "use this pool":
    * `entry.pin_group`         -- `tpu enqueue --group=9` (the explicit field)
    * `launch_kwargs['group']`  -- the older spelling, `--launch=group=9`
  The field wins when both are present. Reading the launch_kwargs copy HERE is
  what makes the 90 rows already carrying `group='9'` mean something: it stays
  DROPPED from the emitted argv (`build_tpu_queue_cmd` must never emit a second
  `--group=`), and is instead consumed as an instruction to the router.

  Pure: no I/O, no mutation. Returns a str like '9' / '5', never '' -- an empty
  or whitespace value is treated as "no pin" so a blank flag cannot silently
  pin a job to the empty group.
  """
  raw = getattr(entry, 'pin_group', None)
  if raw is None:
    raw = (entry.launch_kwargs or {}).get('group')
  if raw is None or isinstance(raw, bool):
    return None            # `--launch=group` with no value is not a pin
  s = str(raw).strip()
  return s or None


def package_dir(entry: 'QueueEntry') -> str:
  """The directory a build must package for this row -- the READ side of the
  enqueue-time snapshot.

  Prefer the frozen `snapshot_dir` (a local copy taken the instant `tpu enqueue`
  ran) over the live `workdir`. This is the whole point of the feature: the
  build runs minutes-to-hours after enqueue, and `tpu queue` packages whatever
  it is pointed at AT BUILD TIME, so pointing it at the live checkout ships
  whatever the source happens to be then. Pointing it at the snapshot ships the
  code as it was at enqueue.

  Fall back to `workdir` when there is no snapshot -- a row enqueued before this
  field existed, or one enqueued with `--no_snapshot`. That fallback is exactly
  the pre-feature behaviour, which is what lets an OLD worker binary (no
  snapshot_dir at all) and a NEW row coexist without either regressing.

  Pure: no I/O, no mutation. Returns '' only when BOTH are empty (a flag-only
  run packaged from the router process dir), preserving the existing meaning of
  an empty cwd downstream.
  """
  snap = (getattr(entry, 'snapshot_dir', '') or '').strip()
  if snap:
    return snap
  return (getattr(entry, 'workdir', '') or '').strip()


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
                     now: Optional[float] = None,
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

  ARCH COOLDOWN: whichever mode is in force, an arch this job keeps being
  re-routed off (`cooldown_archs`, written by mark_reroute) adds an ADDITIVE
  surcharge (`arch_cooldown_surcharge_for`) to its sort price, so it sorts BELOW
  an arch it has not been knocked off. In the priced mode the surcharge is added
  to the effective price; in the fallback it is the leading key. Inert (0.0)
  when `now` is None or the job has no live arch-cooldown record, so it never
  perturbs an un-re-routed job. SORT-ONLY: never added to ca.price, so the
  real-price limit-order gate is untouched.

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
  # ★GPU SPECIAL-CASE (operator, 2026-09-11). A GPU power spec names a BOARD OF
  # A FIXED WIDTH, not a compute budget: 'h100-8' means eight H100s, and the
  # only substitution allowed is a DIFFERENT CARD at the SAME chip count
  # (b200-8, not b200-4). The TPU expansion below answers "what else has this
  # much compute", which for a CUDA binary is catastrophic -- power-matched onto
  # a v5e slice it does not run slowly, it does not run -- and even among GPUs
  # it would rescale b200 DOWN to 4 chips (b200-4 = 19.6 ~ h100-8's 17.2
  # v5p-equiv), handing back HALF a board. So a GPU request NEVER consults the
  # power window: the chip count is preserved verbatim and only the job's own
  # allowed GPU archs that offer that exact width are emitted. Biggest-card-
  # first ordering (b200 before h100) comes from ARCH_PREF in the shared sort
  # below. A topology_locked GPU job falls through to the TPU branch and yields
  # nothing (GPUs have no mesh geometry, so no shape can match a lock) -- the
  # correct safe default for a mesh-sharded checkpoint.
  gpu_req = parse_gpu_shape(entry.power)
  if gpu_req is not None and not entry.topology_locked:
    _, gpu_chips = gpu_req
    for arch in allowed:
      if is_gpu(arch) and gpu_chips in legal_sizes.get(arch, []):
        out.append((arch, gpu_chips))
  else:
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

  # Arch-level cooldown SURCHARGE (>=0.0), 0.0 when now is None or no live
  # record. A generation this job keeps being knocked off reads as dearer by an
  # ADDITIVE `strikes * pct * cap(arch)`, de-preferring it DOWN the type ladder
  # toward v5p/v7. Added ONLY to the sort key here -- never to ca.price -- so the
  # real-price limit-order gate in best_cell_for_shape is untouched. See
  # arch_cooldown_surcharge for the shape (additive, cap-keyed, stacking, capped).
  def _acd(arch: str) -> float:
    return arch_cooldown_surcharge_for(entry, arch, now) if now is not None else 0.0

  if gpu_req is not None:
    # ★GPU CARD CHOICE IS BY PREFERENCE, NOT PRICE: the biggest card that clears
    # its cap wins (b200 before h100 via ARCH_PREF), with the arch-cooldown
    # surcharge able to push a repeatedly-knocked-off card DOWN the order. Never
    # rank GPU options by market price -- "biggest that fits" is the operator's
    # rule, and the limit-order cap gate downstream (not a price compare here)
    # is what makes a too-dear or blocked b200 yield to h100.
    out.sort(key=lambda ac: (_acd(ac[0]), ARCH_PREF.get(ac[0], 99), ac[1]))
  elif arch_price:
    def eff(ac):
      arch, chips = ac
      raw = arch_price.get(arch)
      if raw is None:
        return (float('inf'), ARCH_PREF.get(arch, 99), chips)
      pool = (arch_pool or {}).get(arch, 0.0)
      return (effective_price(raw, pool) + _acd(arch),
              ARCH_PREF.get(arch, 99), chips)
    out.sort(key=eff)
  else:
    out.sort(key=lambda ac: (_acd(ac[0]), ARCH_PREF.get(ac[0], 99), ac[1]))
  return out


# ---------------------------------------------------------------------------
# AUTO-RESUME RULE SET
#
# ★A RULE SET, NOT A RETRY LOOP. Each rule maps an OBSERVED failure to one of
# three verdicts, and the default is DO NOTHING. Automatic re-dispatch is only
# safe where the cause is known to be transient and OUTSIDE our control; where
# the cause is on our side, an automatic retry is a machine for re-triggering
# our own bug. That is not hypothetical: one line burned three XIDs in forty
# minutes replaying into a price gate that re-routing can never clear.
#
# THE VERDICTS
#   RESUME_XID   append a new work unit to the SAME experiment. The job then
#                rediscovers its own newest complete checkpoint in-process
#                (main.py::_apply_borg_autoresume, which enumerates
#                $CHECKPOINT_BUCKET/checkpoints and skips any dir lacking
#                extra.json -- a torn write). This is preferred over naming a
#                path because the launcher cannot know which step survived,
#                and CHECKPOINT_BUCKET is derived from the XID, so reusing the
#                XID lands on the same prefix.
#   LOAD_FROM    the old XID is unusable, so start a NEW experiment warm from
#                an explicit checkpoint. ★Must point at a LEAF checkpoint dir,
#                never its parent: orbax restores one directory, and passing
#                `<bucket>/checkpoints` died with FileNotFoundError while ALSO
#                suppressing auto-resume (which skips itself when LOAD_FROM is
#                set). Travels ONLY as the env var, verbatim -- never as a
#                config key, because which key wins differs per line, so
#                writing the key keeps working on most and silently cold-starts
#                the rest.
#   HOLD         do not re-dispatch. Either the cause is ours, or we cannot see
#                well enough to act.
#
# ★NEVER BOTH. --resume_xid and LOAD_FROM are mutually exclusive at the
# launcher: setting LOAD_FROM disables the in-job rediscovery that --resume_xid
# depends on. Emitting both silently picks the worse half.
RESUME_XID = 'RESUME_XID'
LOAD_FROM = 'LOAD_FROM'
HOLD = 'HOLD'

# Ordered (pattern, verdict, why). FIRST MATCH WINS, so the specific
# prohibitions must precede the general permissions.
_RESUME_RULES: list[tuple[str, str, str]] = [
    # -- our own bugs: retrying re-triggers them ---------------------------
    ('triggered_limit_order', HOLD,
     'held by the global per-family price cap, which is pool-wide: re-routing '
     'to another cell cannot clear it, so a retry is unbounded futile replay'),
    ('zero work unit', HOLD,
     'the launcher refused before adding a work unit (usually a metro with no '
     'group storage), so nothing ran and there is no checkpoint to resume; the '
     'fix is upstream, in cell selection'),
    ('no work units were added', HOLD,
     'same shell as the zero-work-unit case, seen in the launcher log'),
    ('refusing to launch', HOLD,
     'a launcher fail-closed gate (locality/bucket): deterministic, so a retry '
     'reproduces it exactly'),
    ('resource_exhausted', HOLD,
     'quota or a poisoned personal bucket -- a retry writes another 0-byte file'),
    # -- outside our control and transient: safe to continue ---------------
    ('preempt', RESUME_XID,
     'preemption is the normal cost of PROD sharing, not a defect; the work '
     'already done is on CNS and the job can continue from it'),
    ('evict', RESUME_XID, 'eviction is preemption by another name'),
    ('task failure limit', RESUME_XID,
     'borg retried the task past its limit; the experiment itself is intact'),
    ('machine drain', RESUME_XID, 'planned host maintenance, nothing to fix'),
    ('cancelled by pruner', LOAD_FROM,
     'the pruner deletes the experiment, so its XID cannot be appended to; a '
     'new one must start warm from the last surviving checkpoint'),
]


def classify_failure(reason: str) -> tuple[str, str]:
  """Map a failure string to (verdict, why). Unmatched -> HOLD.

  ★DEFAULTS TO HOLD, and an empty or unreadable reason holds too. "I do not
  recognise this" must never become "retry it": the whole point of the rule set
  is that re-dispatch is the exception, granted per known-transient cause.
  """
  r = (reason or '').strip().lower()
  if not r:
    return HOLD, 'no failure reason recorded -- cannot classify, so not acting'
  for pat, verdict, why in _RESUME_RULES:
    if pat in r:
      return verdict, why
  return HOLD, f'unrecognised failure ({r[:60]!r}); rule set defaults to HOLD'


def plan_auto_resume(entry: 'QueueEntry',
                     reason: str,
                     checkpoint: Optional[str],
                     max_auto_resumes: int = 3) -> tuple[str, str]:
  """Decide whether this failed row may be re-dispatched, and how.

  Returns (verdict, why). Verdict is one of RESUME_XID / LOAD_FROM / HOLD.

  Three guards, each of which has a real incident behind it:
    * a BUDGET, because an automatic retry with no ceiling is how three XIDs
      went in forty minutes;
    * NO CHECKPOINT, NO RESUME -- resuming a run that never wrote anything is
      a cold start wearing a resume's clothes, and it re-enters whatever killed
      it the first time;
    * RESUME_XID needs a live XID to append to; without one the only honest
      option is a warm start, and only if a checkpoint actually exists.
  """
  verdict, why = classify_failure(reason)
  if verdict == HOLD:
    return HOLD, why
  used = int(getattr(entry, 'auto_resumes', 0) or 0)
  if used >= max_auto_resumes:
    return HOLD, (f'auto-resume budget spent ({used}/{max_auto_resumes}); '
                  f'a human should look before this burns another XID')
  if not checkpoint:
    return HOLD, (f'{verdict} would be a cold start: no complete checkpoint '
                  f'recorded for this job, so there is no progress to continue')
  if verdict == RESUME_XID and not entry.xid:
    if checkpoint:
      return LOAD_FROM, ('no XID to append to, but a checkpoint exists: start a '
                         'new experiment warm from it')
    return HOLD, 'no XID and no checkpoint'
  return verdict, why


def checkpoint_step(path: Optional[str]) -> int:
  """The step number in a checkpoint path, or -1 if not parseable.

  ONE parser for all four fleet shapes, because a pruned run's surviving
  checkpoint may be any of them (see the LOAD_FROM contract table):
    * torch port:     step_<N>.pt    (a FILE, not a directory)
    * EqR-jax:        step_<N>/       (dir; the job appends /state)
    * codi/coconut:   step_<N>/       (flat dir)
    * paligemma:      checkpoint_<N>  (flax file)
  Returns -1 for anything unrecognised, so a malformed name never out-ranks a
  real checkpoint nor reads as step 0 -- which would be indistinguishable from a
  cold start.
  """
  if not path:
    return -1
  name = str(path).rstrip('/').rsplit('/', 1)[-1]
  m = re.match(r'^(?:step|checkpoint)_(\d+)(?:_.*?)?(?:\.pt)?$', name)
  return int(m.group(1)) if m else -1


def elt_checkpoint_leaf_step(name: str) -> int:
  """Step of an ELT/EqR-jax checkpoint leaf `<step>` (a BARE integer), or -1.

  ELT's orbax CheckpointManager writes `<workdir>/checkpoints/<step>/` where the
  leaf is JUST the number -- a shape checkpoint_step() deliberately rejects
  (it demands a `step_`/`checkpoint_` prefix so a random digit-tailed path is
  never mistaken for a checkpoint). Kept SEPARATE from checkpoint_step for that
  reason: the bare-int rule is only safe when the caller already knows it is
  scanning ELT's `checkpoints/` dir. A bare int is COMPLETE by construction --
  orbax writes to `<step>.orbax-checkpoint-tmp-<uuid>` and atomically renames to
  `<step>` on finalize, so an interrupted save leaves a tmp-suffixed name that
  fails `isdigit()` here and is correctly skipped (the same atomic-rename trust
  the torch `.pt`/`.tmp` scan relies on).
  """
  n = str(name).rstrip('/').rsplit('/', 1)[-1]
  return int(n) if n.isdigit() else -1


# The pruned-restart verdict REUSES the LOAD_FROM contract: the pruner deleted
# the old experiment, so there is no XID to append to (RESUME_XID is impossible)
# and the only honest option is a NEW experiment warm from an explicit leaf
# checkpoint. Named separately so call sites read as intent, not mechanism.
RESUME_WARM = LOAD_FROM


def plan_pruned_restart(
    entry: 'QueueEntry',
    *,
    xm_terminal: bool,
    code_bug: Optional[str],
    checkpoint: Optional[str],
    other_live_writer: bool,
    max_auto_resumes: int = 3,
) -> tuple[str, str]:
  """Decide whether a TERMINATED row was PRUNED -- killed from outside an
  otherwise-healthy run -- and may be re-queued warm from its last checkpoint.
  Returns (verdict, why); verdict is RESUME_WARM or HOLD.

  ★THIS IS THE CHECKPOINT-AS-EVIDENCE PATH, distinct from classify_failure's
  reason-STRING path, and it exists because the two kinds of death look nothing
  alike in the record. A WIM/duty-cycle prune -- and a plain preemption with no
  restart budget -- leaves NO failure reason: the log stops mid-step, the work
  unit goes terminal, and there is no traceback (measured on the dw line, killed
  at step 1759 with a clean cut). classify_failure sees an empty reason and
  correctly defaults to HOLD, because there is no string to match. So the signal
  that an EXTERNAL kill hit a HEALTHY run, rather than a crash, is inverted: a
  complete checkpoint survived AND the log carries no code-bug signature. That
  is what this function keys on -- the presence of evidence, not the content of
  a message.

  Every guard defaults to HOLD, on purpose. An over-eager auto-resume is the
  expensive direction: it burns PROD budget replaying a bug, or it puts a
  SECOND writer on one checkpoint path (measured 2026-09-10: two jobs into one
  out_dir, the 2.93 GB checkpoint at risk of truncation). HOLD costs one dead
  row a human will see; a wrong RESUME costs a corrupted checkpoint or a bug on
  a loop.
  """
  if not xm_terminal:
    return HOLD, 'not terminal on XM -- nothing to restart'
  if code_bug:
    return HOLD, (
        f'the log shows a code bug ({code_bug}); a warm restart would replay it '
        f'and burn another XID. A human must fix the code first.')
  if not checkpoint:
    return HOLD, (
        'terminal with no complete checkpoint: a warm restart would be a cold '
        'start, which re-enters whatever killed it. Nothing to continue from.')
  if other_live_writer:
    return HOLD, (
        'another live job already writes this out_dir; re-queuing would put a '
        'SECOND writer on one checkpoint path (truncation risk). Left alone.')
  used = int(getattr(entry, 'auto_resumes', 0) or 0)
  if used >= max_auto_resumes:
    return HOLD, (
        f'auto-resume budget spent ({used}/{max_auto_resumes}); a human should '
        f'look before this burns another XID')
  step = checkpoint_step(checkpoint)
  return RESUME_WARM, (
      f'pruned/preempted death of a healthy run (no code bug; checkpoint at '
      f'step {step} survived) -- re-queue warm from the last checkpoint')


# --------------------------------------------------------------------------
# PRUNED-RESTART SUPPORT: the pure helpers the reconcile pass needs to turn a
# dead row into a warm restart. The I/O that FEEDS them (reading the log text,
# listing the checkpoint dir) lives in route_check; these stay pure so the
# taxonomy is unit-tested without CNS -- and, being in this already-symlinked
# source module, a daemon restart picks them up with NO rebuild.
# --------------------------------------------------------------------------

# Code-bug signatures, a trimmed copy of
# infra_check._APPLICATION_ERROR_SIGNATURES. ★KEEP IN SYNC with that file -- it
# is the richer twin and owns the full list; this copy exists only so route_lib
# keeps its no-heavy-imports promise. Matched against the log TAIL only (see
# looks_like_code_bug).
_CODE_BUG_SIGNATURES = (
    ('SEGMENTATION FAULT', 'segfault (SIGSEGV)'),
    ('SIGSEGV', 'segfault (SIGSEGV)'),
    ('SIGNAL 11', 'segfault (SIGSEGV)'),
    ('SIGABRT', 'abort (SIGABRT)'),
    ('SIGNAL 6', 'abort (SIGABRT)'),
    ('OUT OF MEMORY', 'out of memory'),
    ('OUTOFMEMORY', 'out of memory'),
    ('RESOURCE_EXHAUSTED: OOM', 'out of memory (HBM)'),
    ('OOM_KILLED', 'OOM-killed'),
    ('OOMKILLED', 'OOM-killed'),
    ('TRACEBACK (MOST RECENT CALL LAST)', 'unhandled Python exception'),
    ('ASSERTIONERROR', 'AssertionError'),
    ('RUNTIMEERROR', 'RuntimeError'),
    ('VALUEERROR', 'ValueError'),
    ('TYPEERROR', 'TypeError'),
    ('KEYERROR', 'KeyError'),
    ('INDEXERROR', 'IndexError'),
    ('FILENOTFOUNDERROR', 'FileNotFoundError'),
    ('XLARUNTIMEERROR', 'XLA runtime error'),
    ('CUDA ERROR', 'CUDA error'),
    ('NON-ZERO EXIT', 'non-zero exit'),
    ('APPLICATION LEVEL ERROR', 'application-level failure'),
)


def looks_like_code_bug(log_tail: str) -> Optional[str]:
  """Return a short code-bug verdict if the log TAIL shows the run killed
  ITSELF (crash / OOM / unhandled exception), else None.

  ★PASS THE TAIL, NOT THE WHOLE LOG. A pruned run's crash, if any, is at the
  end; the head is a boot banner that legitimately prints a benign
  'ModuleNotFoundError: No module named base' readback note (measured on the dw
  line), and scanning it would misread every healthy pruned run as a bug and
  HOLD it -- the exact false positive this whole path exists to avoid.
  """
  if not log_tail:
    return None
  up = log_tail.upper()
  for needle, verdict in _CODE_BUG_SIGNATURES:
    if needle in up:
      return verdict
  return None


_OUT_DIR_PATTERNS = (
    re.compile(r"out_dir \(post-locality\) = '([^']+)'"),
    re.compile(r'checkpoint saved -> (\S+?)/steps/'),
    # ELT/EqR-jax (elt_dit_pkg/main_eqr.py) names no launcher out_dir line; its
    # durable workdir is announced by the boot banner instead:
    #   [main_eqr] TRAINING run: redirecting workdir -> $CHECKPOINT_BUCKET <path>
    # That <path> IS the out_dir (the parent of checkpoints/), so parse it too --
    # without this the evidence layer finds no out_dir for an ELT row and every
    # ELT auto-resume falls to HOLD for lack of a checkpoint it never looked for.
    re.compile(r'redirecting workdir -> \$CHECKPOINT_BUCKET (/cns/\S+?)(?:\s|$)'),
    re.compile(r"out_dir[:=]\s*'?(/cns/[^\s']+)'?"),
)


def out_dir_from_log(log_text: str) -> Optional[str]:
  """The run's post-locality out_dir (checkpoint parent), parsed from its log.

  Prefers the launcher's explicit 'out_dir (post-locality) = ...' line -- the
  ground-truth path the job actually wrote, already remapped to the metro it
  landed in -- so the checkpoint scan needs no metro guessing. Falls back to a
  'checkpoint saved -> <dir>/steps/...' line, then a bare out_dir key. None if
  the log names no path.
  """
  if not log_text:
    return None
  for pat in _OUT_DIR_PATTERNS:
    m = pat.search(log_text)
    if m:
      return m.group(1).rstrip('/')
  return None


_LIVE_STATES = frozenset({
    JobState.QUEUED, JobState.BUILD_REQUESTED, JobState.BUILDING,
    JobState.SUBMITTED, JobState.RUNNING, JobState.HELD,
    JobState.BUDGET_DEFERRED,
})


def has_live_config_sibling(entry: 'QueueEntry',
                            entries: Sequence['QueueEntry']) -> bool:
  """True if some OTHER still-live entry targets the same run as `entry`.

  ★OUT_DIR IS A FUNCTION OF THE CONFIG for this fleet: each config .yml pins a
  fixed out_dir, so two live entries sharing launch_kwargs['config'] write the
  SAME checkpoint directory. That is the double-writer hazard measured
  2026-09-10 (two dw jobs into one out_dir, a 2.93 GB checkpoint at truncation
  risk), so a warm restart is refused while any same-config sibling is live.
  A restart's own new entry is a live sibling too, which is what stops a second
  restart of the same run on the next reconcile pass.
  """
  cfg = (getattr(entry, 'launch_kwargs', None) or {}).get('config')
  if not cfg:
    return False
  for e in entries:
    if e.job_id == entry.job_id:
      continue
    if e.state not in _LIVE_STATES:
      continue
    if (getattr(e, 'launch_kwargs', None) or {}).get('config') == cfg:
      return True
  return False


def _resume_exp_name(base: str, attempt: int) -> str:
  """`base` with a '-r<attempt>' suffix, replacing any existing one so suffixes
  do not stack (`...-r1-r2`). out_dir is unaffected (it lives in the config), so
  this is cosmetic for XM readability, not a path change.
  """
  base = re.sub(r'-r\d+$', '', base or '')
  return f'{base}-r{attempt}'


def _elt_restart_from_checkpoint(
    checkpoint: Optional[str]) -> Optional[tuple[str, int]]:
  """(workdir, step) if `checkpoint` is an ELT CheckpointManager leaf, else None.

  ELT/EqR-jax lays a checkpoint out as `<workdir>/checkpoints/<step>` with a
  BARE-INTEGER leaf -- a shape distinct from all four that checkpoint_step()
  knows (`step_<N>[.pt]`, `checkpoint_<N>`), which is exactly why
  checkpoint_step() returns -1 for it. This is the INVERSE of
  load_config._apply_restart_from_env: restart_from is the WORKDIR (the parent
  of `checkpoints/`, and must NOT end in a digit) and restart_step is the named
  step. Returns None for any other shape, so the caller falls back to the
  load_from contract that every non-ELT family uses.
  """
  if not checkpoint:
    return None
  parts = str(checkpoint).rstrip('/').rsplit('/', 2)
  if len(parts) < 3 or parts[-2] != 'checkpoints' or not parts[-1].isdigit():
    return None
  return parts[0], int(parts[-1])


def build_warm_restart_entry(dead: 'QueueEntry', checkpoint: str,
                             new_job_id: str) -> 'QueueEntry':
  """A fresh QUEUED entry that resumes `dead` warm from `checkpoint`.

  Clones the launch spec (power / archs / tier / metros / workdir / config)
  verbatim so the restart is the SAME run, wires the surviving checkpoint into
  the RIGHT resume mechanism for its layout (see below), increments auto_resumes
  (the budget the restart decision reads), and records the dead xid in
  prior_xids. NOT called unless plan_pruned_restart returned RESUME_WARM, so
  every guard has already passed.

  ★RESUME MECHANISM IS LAYOUT-SPECIFIC, and picking the wrong one is silently
  destructive. An ELT/EqR-jax CheckpointManager writes `<workdir>/checkpoints/
  <step>` (a bare-integer leaf), and ELT TRAINING must resume via
  restart_from(=workdir)+restart_step(=N): handed $LOAD_FROM instead, main_eqr
  does `workdir = LOAD_FROM` and read+writes the rescued dir, so orbax
  (max_to_keep) deletes the very checkpoints it resumed from (measured
  2026-09-09: XID 288109183/288109952 died at step 0 this way). Every OTHER
  fleet layout (torch `step_<N>.pt`, paligemma `checkpoint_<N>`, ...) honours
  $LOAD_FROM. So we emit restart_from+restart_step for an ELT leaf and load_from
  for everything else -- and CLEAR the other mechanism's keys, because the two
  are mutually exclusive at the launcher (both set trips main_eqr's guard). See
  elt_dit_pkg/configs/load_config.py::_apply_restart_from_env (the contract this
  inverts) and elt_dit_pkg/load_from_guard.py (the runtime backstop).
  """
  lk = dict(getattr(dead, 'launch_kwargs', None) or {})
  elt = _elt_restart_from_checkpoint(checkpoint)
  if elt is not None:
    workdir_ckpt, step = elt
    lk.pop('load_from', None)          # mutually exclusive with restart_from
    lk['restart_from'] = workdir_ckpt  # the WORKDIR (parent of checkpoints/)
    lk['restart_step'] = str(step)     # named explicitly (load_config fails closed
                                       # on a missing/ambiguous step)
  else:
    lk.pop('restart_from', None)       # do not carry a stale ELT resume forward
    lk.pop('restart_step', None)
    lk['load_from'] = checkpoint
  base = lk.get('exp_name') or dead.job_id
  new_attempt = int(getattr(dead, 'auto_resumes', 0) or 0) + 1
  lk['exp_name'] = _resume_exp_name(base, new_attempt)
  prior = list(getattr(dead, 'prior_xids', None) or [])
  if dead.xid:
    prior.append(str(dead.xid))
  return QueueEntry(
      job_id=new_job_id,
      power=dead.power,
      allowed_archs=list(dead.allowed_archs),
      tier=dead.tier,
      allowed_metros=(list(dead.allowed_metros)
                      if dead.allowed_metros else dead.allowed_metros),
      priority=dead.priority,
      power_tolerance=dead.power_tolerance,
      max_price=dead.max_price,
      launch_kwargs=lk,
      workdir=dead.workdir,
      # ★NOT snapshot_dir: a warm restart runs potentially DAYS after the
      # original enqueue, by when the enqueue snapshot has been reclaimed (it is
      # GC'd once the row reaches a terminal state, since the build's own
      # durable CitC stagedir then exists). Inheriting that path would point the
      # resume at a deleted directory and park it HELD. Leaving it empty makes
      # package_dir fall back to `workdir` -- the pre-feature behavior, always
      # safe. The enqueue snapshot protects the HUMAN's edit window between their
      # enqueue and the first build; an infra-driven auto-resume has no such
      # window to protect.
      state=JobState.QUEUED,
      auto_resumes=new_attempt,
      prior_xids=prior,
      last_reason=(f'auto warm-restart from {checkpoint} after '
                   f'pruned/preempted death of {dead.job_id} '
                   f'(xid {dead.xid})'),
  )


def _metro_has_group_storage(metro: str) -> bool:
  """Does this metro have a group CNS registration the launcher will accept?

  ★FAILS OPEN, on purpose, in BOTH unknown directions: an empty metro, or one
  the locality snapshot has never heard of, returns True. A cell we cannot
  classify must stay a candidate, because the alternative -- deleting it -- is
  how a lookup miss becomes an indistinguishable "no capacity anywhere". The
  launcher still refuses it at submit time, so the cost of guessing wrong here
  is one wasted XID, whereas the cost of guessing wrong the other way is a
  fleet that cannot place anything.

  ★Reads cell_locality, the ONE measured table (storage.md: never hand-maintain
  a second cell/metro/bucket map). Import is local so this module keeps working
  -- fail-open -- wherever cell_locality is not on the path, e.g. a unit test
  that exercises routing with synthetic cells.

  PERSONAL-ONLY metros (phx/ske) count as NO storage. They are the worst case,
  not a lesser one: they resolve and launch, then every write lands on a
  personal 500 GiB quota whose handle is poisoned, failing with
  resource_exhausted while LEAVING A 0-BYTE FILE -- so the job looks like it
  produced output. A zero-work-unit shell is at least visibly broken.
  """
  m = (metro or '').strip().lower()
  if not m:
    return True
  try:
    from google3.experimental.users.qiaos.tpu_utils import cell_locality
  except ImportError:
    try:
      import cell_locality  # type: ignore
    except ImportError:
      return True
  # ★"Not in the storage table" has TWO meanings and only one of them is a
  # refusal. `tpe`/`uos`/`nrt` are metros the snapshot KNOWS and that have no
  # group registration -- a real, measured prohibition. A metro the snapshot has
  # never seen (a new turn-up, or a synthetic cell in a test) is simply UNKNOWN,
  # and treating unknown as "no storage" deletes candidates for a reason nobody
  # measured. Caught by a negative control that fed a made-up metro and watched
  # the router return "nothing placeable".
  try:
    known = {x.lower() for x in cell_locality.metros_with_storage()}
  except Exception:
    return True
  if not known:
    return True
  if m in known:
    return True
  # Known-and-unregistered only counts when the snapshot can place the metro at
  # all: personal-only metros are named explicitly, and anything the snapshot
  # can name via a cell lookup is known. Otherwise fail open.
  personal = {x.lower() for x in getattr(cell_locality, '_PERSONAL_ONLY_METROS', {})}
  if m in personal:
    return False   # resolves, launches, then writes into a poisoned personal quota
  # A metro is KNOWN if the measured snapshot can name at least one cell in it.
  try:
    is_known_metro = bool(cell_locality.cells_in_metro(m))
  except Exception:
    is_known_metro = False
  if is_known_metro:
    return False   # snapshot knows this metro and it has no group storage
  return True      # never heard of it -> fail open


def _family_price_cap(arch: str) -> Optional[float]:
  """The GLOBAL limit-order cap for an accelerator family, or None if no policy.

  ★READ-ONLY VIEW of an existing policy. This never sets, raises or relaxes a
  cap; it only lets the router see the gate that already exists downstream, so
  it can stop sending jobs to be refused by it (operator red line: never raise
  the cap).

  None means "no policy for this family", which the caller must treat as
  UNCAPPED, never as a zero cap -- a 0 would exclude every cell at any nonzero
  price and empty the fleet.
  """
  a = (arch or '').strip().lower()
  if not a:
    return None
  try:
    from google3.experimental.users.qiaos.tpu_utils import cap_policy
  except ImportError:
    try:
      import cap_policy  # type: ignore
    except ImportError:
      return None
  try:
    return cap_policy.cap_for_family(a)
  except Exception:
    return None


def best_cell_for_shape(
    arch: str,
    chips: int,
    entry: QueueEntry,
    avail_by_cell: dict[str, CellAvail],
    now: float,
) -> Optional[tuple[CellAvail, int]]:
  """Best placeable cell for one (arch, chips), or None if none can place it.

  HARD GATES first, in order: metro filter, price cap, oversold drop, cooldown
  drop, and >=1 placeable slice. Those are prohibitions -- a cell that fails one
  is not merely unattractive, it is unusable, so no score can rescue it.

  Survivors are then RANKED BY `cell_score` (lower is better):

      cell_price / slice_weight(n_slices) * evict_penalty(strikes, age)

  i.e. the cheapest cell wins unless a roomier one is within SLICE_BONUS of its
  price, and a cell that recently evicted this job is read as proportionally
  dearer. Ties break on roominess. Returns (cell_avail, n_slices).

  Previously this sorted `(-n_slices, price, -free_chips)` -- roominess first,
  price only on an exact tie. Harmless while every cell of an arch reported the
  same price; wrong the instant per-cell prices became visible."""
  allowed_metros = [m.lower() for m in (entry.allowed_metros or [])]
  # ★THE ROUTER MUST NOT PROPOSE A CELL A DOWNSTREAM GATE WILL REFUSE.
  # Two gates sit between this choice and a running job, and until now neither
  # was visible here, so the router kept picking cells that could not possibly
  # work and spent a real XID finding out.
  #
  # 1. STORAGE. `xm_launcher._local_bucket()` SystemExits when the landing
  #    cell's metro has no group CNS registration. That exit happens AFTER
  #    `create_experiment` and BEFORE `experiment.add(job)`, so it leaves an
  #    XID that resolves with ZERO work units and no logs -- indistinguishable
  #    from "XM flaked". MEASURED 2026-09-02: 7 cars died this way in one day
  #    (cells td/gd/rx = metros tpe/uos/nrt), while every car that landed in a
  #    storage metro lived. Re-routing cannot fix it: the next cell in the same
  #    metro fails identically.
  # 2. PRICE. The limit order is a GLOBAL per-family constant (cap_policy.py,
  #    mirrored in tpu_wrapper.sh `_tpu_limit_price_for_arch`), not the
  #    per-entry `max_price` filtered below -- and every queue row carries
  #    max_price=None, so that filter has never once excluded anything. A job
  #    over the family cap is held by TRIGGERED_LIMIT_ORDER *before* the
  #    capacity check, times out, and re-routes to another cell, where the same
  #    pool-wide price gate stops it again: unbounded futile replay.
  #    ★This only makes the EXISTING cap visible to the router. It does not
  #    raise, lower, or override any cap (operator red line).
  #
  # Both are prohibitions, so they belong with the hard gates, ahead of scoring.
  # Both fail OPEN on an unknown answer: a metro we cannot resolve, or a family
  # with no cap policy, must not silently delete every candidate -- that would
  # turn a lookup miss into "no capacity anywhere", the exact failure that made
  # `--metro` read as an empty fleet.
  ranked: list[tuple[float, int, CellAvail]] = []
  rejected_no_storage: list[str] = []
  rejected_over_cap: list[str] = []
  cap = _family_price_cap(arch)
  for ca in avail_by_cell.values():
    if ca.arch.lower() != arch.lower():
      continue
    if allowed_metros and ca.metro.lower() not in allowed_metros:
      continue
    if not _metro_has_group_storage(ca.metro):
      rejected_no_storage.append(ca.cell)
      continue
    if cap is not None and ca.price is not None and ca.price > cap:
      rejected_over_cap.append(ca.cell)
      continue
    if entry.max_price is not None and ca.price is not None and ca.price > entry.max_price:
      continue
    if ca.oversold:
      continue
    # ★Cooldown is NOT a gate here any more (operator 2026-09-01 14:40Z); it
    # feeds cell_score as a decaying multiplier instead. Excluding the cell
    # fought the very purpose of re-routing every 10 minutes -- it removed the
    # option rather than ranking it, and with every candidate cooling it left
    # the router with nothing to pick.
    n = slices_for(ca.free_chips, chips)
    if n < 1:
      continue
    ev = entry.evictions.get(ca.cell) if entry.evictions else None
    strikes = int((ev or {}).get('strikes', 0) or 0)
    last = (ev or {}).get('last')
    since = (now - float(last)) if last is not None else None
    score = cell_score(ca.price, n, strikes, since,
                       cooldown_until=entry.cooldown_cells.get(ca.cell),
                       now=now)
    ranked.append((score, -n, ca))
  if not ranked:
    # ★SAY WHY THE SHAPE WAS DROPPED. A silent filter is how the previous
    # metro bug stayed invisible for a day: "no capacity" and "every candidate
    # was refused by a gate" look identical from the caller, and only the
    # second one is actionable. Recorded on the entry so the reroute log can
    # print it without changing this function's return type.
    if rejected_no_storage or rejected_over_cap:
      bits = []
      if rejected_no_storage:
        bits.append(f'{len(rejected_no_storage)} cell(s) in metros with no group '
                    f'storage (would launch a ZERO-work-unit shell): '
                    f'{",".join(sorted(rejected_no_storage)[:6])}')
      if rejected_over_cap:
        bits.append(f'{len(rejected_over_cap)} cell(s) above the {arch} limit-order '
                    f'cap of {cap} cr/chip-hr (would be held by '
                    f'TRIGGERED_LIMIT_ORDER, and re-routing cannot clear a '
                    f'pool-wide price gate): {",".join(sorted(rejected_over_cap)[:6])}')
      entry.last_filter_reason = f'{arch}-{chips}: ' + '; '.join(bits)
    return None
  # score asc (cheapest effective), then roomier first as the tie-break
  ranked.sort(key=lambda r: (r[0], r[1]))
  best = ranked[0]
  return best[2], -best[1]


def _placement_for(entry: QueueEntry, arch: str, chips: int,
                   ca: CellAvail, n: int) -> Placement:
  """Build the Placement for one resolved (arch, chips, cell). Shared by the
  two exits of plan_one so the reason string stays identical."""
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

  ★CROSS-ARCH COOLDOWN FALLTHROUGH. `best_cell_for_shape` treats a cell this
  job was just re-routed OFF (`cooldown_cells`) as a SOFT `cell_score` penalty,
  not a gate -- correct when the arch has several cells, so the penalty can
  reorder them. But when the top-preferred arch has exactly ONE usable cell
  (b200 -> only sj in the allowed metros), the penalty has nothing to reorder,
  and the old "first arch with any placeable cell wins" rule sent a re-routed
  job straight back to the same stuck cell every pass -- never trying the next
  arch (h100), which had free capacity elsewhere. So a shape whose chosen cell
  is STILL cooling is held only as a fallback: we keep scanning later archs,
  take the first shape whose cell is NOT cooling, and use the cooled fallback
  only if every arch resolves to a cooling cell (then going back is no worse
  than today). No cooldown in play => identical behaviour to before.

  Returns a Placement, or None if nothing can place it right now (it stays
  QUEUED and is retried next tick -- we NEVER submit into an oversold/full
  cell, which is exactly the bug this whole system exists to avoid)."""
  fallback: Optional[tuple[str, int, CellAvail, int]] = None
  for arch, chips in candidate_shapes(entry, legal_sizes, arch_price, arch_pool,
                                      now=now):
    hit = best_cell_for_shape(arch, chips, entry, avail_by_cell, now)
    if hit is None:
      continue
    ca, n = hit
    cooling = entry.cooldown_cells.get(ca.cell)
    if cooling is not None and cooling > now:
      # This shape only resolves to a cell the job is cooling off. Remember the
      # best (first, i.e. cheapest/preferred) such option and keep looking for
      # an arch/cell that is not cooling.
      if fallback is None:
        fallback = (arch, chips, ca, n)
      continue
    return _placement_for(entry, arch, chips, ca, n)
  if fallback is not None:
    arch, chips, ca, n = fallback
    return _placement_for(entry, arch, chips, ca, n)
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


# ★Re-route backoff. `reroutes` was written by mark_reroute and read by NOTHING
# (measured 2026-09-01), so attempt 1 and attempt 7 got exactly the same 600s of
# patience. That is what let a car churn: cancelled at 600s, re-queued to the
# back, placed again, cancelled at 600s again -- 7 number plates in 2.5 hours
# for one job, and the system had no way to notice it was the same car.
REROUTE_BACKOFF_MAX_DOUBLINGS = 4   # cap the exponent: 600s -> 9600s, not forever
# NO give-up bound (removed 2026-09-11 by operator request): a job is re-routed
# as many times as it takes, never auto-parked as HELD for churning. Churn stays
# bounded by the two OTHER brakes -- per-row backoff (reroute_deadline_s doubles
# the patience each move, capped at ~2.7h) and the global rate cap
# (REROUTE_GLOBAL_MAX_PER_HOUR). entry.reroutes is surfaced on the board instead,
# so a human can see a high count and intervene -- that is the new safety valve.


# ★The GLOBAL brake, and why per-row backoff is not enough on its own: the
# backoff above keys off entry.reroutes, which lives ON THE ROW, so dequeuing a
# churning job and re-enqueuing it resets the counter to 0. Measured 2026-09-01:
# one experiment burned 7 number plates across TWO rows (5 on a row that was
# then dequeued, 2 on its replacement), and a per-row counter sees only the 2.
# exp_name cannot key it either (it changed every re-send: armA, armA2, ...
# 22 rows, 22 names), nor can load_from (9 unrelated rows shared one ckpt).
# So the brake is keyed on NOTHING: it counts what the re-router itself did in
# the last hour. A fleet-wide churn rate survives any renaming of the cars.
REROUTE_GLOBAL_MAX_PER_HOUR = 8


def global_reroute_brake(recent_reroute_times: Sequence[float], now: float,
                         max_per_hour: int = REROUTE_GLOBAL_MAX_PER_HOUR,
                         window_s: float = 3600.0) -> bool:
  """True if the re-router should STOP re-routing this pass. Pure.

  `recent_reroute_times` is every re-route this process performed, as epoch
  seconds; only those inside `window_s` count. Tripping means the problem is on
  the re-router's side, not the car's, so the caller alerts and does nothing --
  cancelling more cars cannot fix a re-router that is cancelling too much.

  Threshold 8/hour is set one notch above the measured peak of 6/hour
  (2026-09-01 10Z, the worst hour of a day whose other hours ran 2-3)."""
  cutoff = now - window_s
  return sum(1 for t in recent_reroute_times if t >= cutoff) >= max_per_hour


def reroute_deadline_s(entry: QueueEntry, base_s: float) -> float:
  """Patience for THIS attempt: base * 2^min(reroutes, MAX_DOUBLINGS). Pure.

  A job that has already been moved N times is, empirically, not a job whose
  next placement lands in 600s -- so each move buys the next attempt more time
  instead of re-running the same failed experiment at the same speed."""
  n = min(max(entry.reroutes, 0), REROUTE_BACKOFF_MAX_DOUBLINGS)
  return base_s * (2 ** n)


def needs_reroute(entry: QueueEntry, now: float, reroute_after_s: float) -> bool:
  """True if a SUBMITTED job has been pending past the re-route deadline.

  The core does NOT know the job's live XM status -- the binary passes that in
  by only calling this for jobs it has confirmed are still PENDING. Here we just
  own the CLOCK part of the rule: submitted, and older than the deadline.

  The deadline GROWS with entry.reroutes (see reroute_deadline_s): a job moved
  many times waits longer before the next move, but it is never given up on --
  there is no bound past which re-routing stops. Churn stays bounded by that
  backoff and by the global rate cap, and entry.reroutes is shown on the board
  so a human can spot a stuck car and act."""
  if entry.state != JobState.SUBMITTED:
    return False
  if entry.submitted_at is None:
    return False
  return (now - entry.submitted_at) >= reroute_deadline_s(entry, reroute_after_s)


def needs_liveness_recheck(entry: QueueEntry, now: float,
                          grace_s: float) -> bool:
  """True if a RUNNING row is old enough that its liveness must be re-verified
  against something OTHER than XManager.

  ★Deliberately separate from needs_reroute. That function owns one question --
  "is a SUBMITTED job stuck in the auction past its patience" -- and its clock
  carries the re-route backoff. This one owns a different question: "is a row we
  already promoted to RUNNING actually on hardware?" XManager reports RUNNING
  for a job whose Borg VM groups never left PENDING, and because needs_reroute
  selects SUBMITTED only, promoting such a row removed it from every mechanism
  that could move it (measured: xid 286573746, 12 h in one cell, every VM group
  PENDING, zero bytes written, six unused fallback metros).

  Selecting a row here is NOT a verdict -- the caller must confirm with an
  independent probe before acting.
  """
  if entry.state != JobState.RUNNING:
    return False
  if entry.submitted_at is None:
    return False
  return (now - entry.submitted_at) >= grace_s


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
  """Return the entry reset to QUEUED after a failed placement.

  Two cooldowns are stamped, both for `cooldown_s` seconds:
    * the CELL it was stuck in (`cooldown_cells`) -- a decaying soft penalty on
      re-picking that exact cell;
    * the ARCH it was on (`cooldown_archs`) -- a flat, STACKING penalty on the
      whole generation, so a job that keeps being knocked off a cheap-but-
      unusable arch (v4's many cells defeat the per-cell cooldown) is pushed up
      the type ladder. Strikes stack while the arch is still cooling and reset
      once a full window has passed with no re-route (the arch recovered)."""
  if entry.cell:
    entry.cooldown_cells[entry.cell] = now + cooldown_s
  # ARCH-LEVEL cooldown (operator 2026-09-10). Cool the whole ARCH too, not just
  # its cell -- cooling one v4 cell just sends the router to the next v4 cell.
  # `entry.arch` is still set here (cleared below). Strikes STACK while the arch
  # is still cooling (knocked off the same generation again before its window
  # closes => bites harder), and RESET to 1 once the previous window has fully
  # elapsed (a full cooldown with no re-route is evidence the arch recovered).
  if entry.arch:
    a = entry.arch.lower()
    prev = (entry.cooldown_archs or {}).get(a) or {}
    prev_until = prev.get('until')
    if prev_until and now < prev_until:
      strikes = int(prev.get('strikes', 0) or 0) + 1
    else:
      strikes = 1
    if not entry.cooldown_archs:
      entry.cooldown_archs = {}
    entry.cooldown_archs[a] = {'until': now + cooldown_s, 'strikes': strikes}
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
  Returns the list of entries reclaimed. Frees the single-build slot.

  ★A stale claim does NOT prove the build failed. The other reading is that
  the build SUCCEEDED, the experiment is running on XManager, and only the
  write-back of the xid was lost -- observed 2026-09-02 on
  elt-dit-50k-fid-v3b, whose row was reclaimed to QUEUED while xid 285706173
  was RUNNING. Re-dispatching such a row puts a SECOND writer on the first
  one's output path, which is silent and destroys both.

  This function cannot resolve that ambiguity itself: it runs inside the queue
  flock, and an XManager RPC there would block every reader for the length of
  a network call. So it records the suspicion instead -- `adopt_check_name`
  carries the experiment name the reconcile pass must look up BEFORE the row
  is allowed to build again. `route_check.adopt_escaped_builds` (outside the
  lock) does the lookup and either adopts the live xid or clears the flag.
  """
  reclaimed = []
  for e in entries:
    if building_is_stale(e, now, stale_after_s):
      e.state = JobState.QUEUED
      e.build_started_at = None
      e.worker_id = None
      e.attempts += 1
      # The name to look up. launch_kwargs is where `tpu queue` gets --exp_name,
      # so it is the same string the experiment was created under.
      e.adopt_check_name = (e.launch_kwargs or {}).get('exp_name') or None
      if e.adopt_check_name:
        e.last_reason = (
            f'reclaimed: BUILDING claim went stale (>{int(stale_after_s)}s); '
            f'HOLDING for adopt-check on {e.adopt_check_name} -- the build may '
            f'have escaped to XManager and re-dispatching would double-write')
      else:
        e.last_reason = (
            f'reclaimed: BUILDING claim went stale (>{int(stale_after_s)}s); '
            f'no exp_name to adopt-check with, so an escaped build cannot be '
            f'ruled out')
      reclaimed.append(e)
  return reclaimed


def next_queued(entries: list['QueueEntry']) -> Optional['QueueEntry']:
  """The next QUEUED entry to build, highest priority first then FIFO-ish by
  list order. Returns None if nothing is queued. Does NOT consider whether a
  build is already in flight -- the caller enforces the single-build invariant."""
  # A row awaiting its adopt-check is deliberately invisible here: see
  # QueueEntry.adopt_check_name. This is the gate that actually prevents the
  # double-write; the flag alone would only be a comment.
  queued = [e for e in entries
            if e.state == JobState.QUEUED and not e.adopt_check_name]
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
  reqd = [e for e in entries
          if e.state == JobState.BUILD_REQUESTED and not e.adopt_check_name]
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
                     completed_const: str = 'COMPLETED',
                     gone_const: str = 'GONE',
                     age_s: Optional[float] = None,
                     gone_min_age_s: float = 1800.0,
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
    XM GONE      -> FAILED, but ONLY once the row is older than gone_min_age_s
                   (the id resolves and reports zero work units -- see below)
  A local entry not in RECONCILABLE_STATES is never passed here (caller filters).

  GONE vs UNKNOWN is the distinction that unblocked 13 rows stuck 5-8 days:
  UNKNOWN means the probe could not read XM, GONE means it read XM fine and XM
  has no work units for this id. Only the second is a verdict about the world.
  The age gate exists because a JUST-submitted experiment is briefly 0-WU too,
  and that window is indistinguishable from a long-dead one by status alone --
  so `age_s` (seconds since submitted_at) must be supplied and exceed
  gone_min_age_s before GONE is actioned. An absent age_s is treated as
  'cannot tell how old this is' and, like UNKNOWN, does nothing.

  ★THE GATE MUST CLEAR THE TAIL OF THE STARTUP DISTRIBUTION, NOT ITS MEDIAN.
  Measured over 9 cars that XM later confirmed RUNNING, the delay from XM
  CreateTime to the local RUNNING promotion ran 449-876 s (median 650 s), so a
  900 s gate sat at 97% of the slowest observed startup and killed live cars on
  the tail. Re-measure with the same two timestamps before moving this number;
  a gate below the observed maximum reads as 'the experiment is gone' when it
  only means 'it has not come up yet', and the cost is asymmetric: a row failed
  early keeps its xid RUNNING and billing while the queue stops tracking it,
  whereas a row failed late is merely cleaned up one round later.

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
  if xm_status == gone_const:
    # Fail CLOSED on a missing/young age: a fresh submission legitimately shows
    # zero work units for a short while, and killing that row would cancel a
    # car that is about to come up.
    if age_s is not None and age_s >= gone_min_age_s:
      return JobState.FAILED
    return None
  # RUNNING+RUNNING, any PENDING, any UNKNOWN, or an unrecognised status: no-op.
  return None


def reconcile_entry(entry: QueueEntry, xm_status: str, reason: str = '',
                    age_s: Optional[float] = None) -> bool:
  """Apply decide_reconcile to one entry in place. Returns True if the entry's
  state changed (a zombie was cleaned up or a placement promoted), False if left
  unchanged. Only touches entries currently in RECONCILABLE_STATES.

  `age_s` is only consulted for the GONE verdict (zero work units), which is
  ignored on a young row; see decide_reconcile."""
  if entry.state not in RECONCILABLE_STATES:
    return False
  new_state = decide_reconcile(entry.state, xm_status, age_s=age_s)
  if new_state is None or new_state == entry.state:
    return False
  old = entry.state
  entry.state = new_state
  if new_state == JobState.DONE:
    entry.last_reason = reason or f'reconciled: XM reports COMPLETED (was local {old.value}); finished normally'
  elif new_state == JobState.FAILED:
    if xm_status == 'GONE':
      entry.last_reason = reason or (
          f'reconciled: XM resolves the id but reports ZERO work units '
          f'(was local {old.value}); experiment is gone, row cleaned up')
    else:
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
