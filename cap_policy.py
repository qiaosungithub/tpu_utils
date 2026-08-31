"""Fixed per-family limit-order (price-cap) policy: the ONE source of truth.

A limit order caps the price per chip-hour a job will pay; above it the job is
pulled from the auction (pending) or paused (running) -- reversible, never
fatal. See ../../infra/quota_market.md (wiki) for the market model.

POLICY, not market tracker (operator directive 2026-08-25). The cap is a
HARDCODED absolute price per chip-hour, one number per accelerator family. It is
a blast-radius bound -- generous enough to ride ordinary intraday swings, tight
enough to stop a runaway bill -- NOT an attempt to clear at the median. It is
deliberately NOT externally overridable: there is exactly one number per family
and this is where it lives.

TWO HALVES, ONE POLICY. The shell half is tpu_wrapper.sh
`_tpu_limit_price_for_arch` (which actually SETS the per-XID cap at launch via
set_limit_order); this Python half is read by money_check.py so the DISPLAYED
cap and the ENFORCED cap cannot drift. The numbers here and there must match.
LINT.IfChange(cap_policy)

Per chip-hour, tiered by compute class (per-chip `vle` vs v5p from
borg/util/reports/gxu/gxus_by_platform_ga.textproto):
  * TPU: v4/v5p 5, v6e 10, v6p/v7 20.
  * GPU: A100 ~0.68x v5p -> 5 (v5p tier); H100/H200 ~2.15x ~= v6e -> 10;
    B200/B300/GB200/GB300 ~4.9x ~= v6p/v7 -> 20.
A GPU device counts as one chip and set_limit_order is per chip-hour, so the GPU
caps are per GPU-hour. All sit far above current market (H100 PROD ~0.1, B200
~0.6, A100 ~0.9, GB200 free), by design.
"""

# Family name (matching the shell case labels) -> cap in credits per chip-hour.
# A family absent here has NO policy: the caller leaves the job uncapped rather
# than guess.
CAP_POLICY: dict[str, float] = {
    # TPUs
    "v7": 20.0,
    "v6p": 20.0,
    "v6e": 10.0,
    "v5p": 5.0,
    "v4": 5.0,
    # NVIDIA GPUs
    "a100": 5.0,
    "a100_80gib": 5.0,
    "h100": 10.0,
    "h200": 10.0,
    "b200": 20.0,
    "b300": 20.0,
    "gb200": 20.0,
    "gb300": 20.0,
}
# LINT.ThenChange(//depot/google3/experimental/users/qiaos/tpu_wrapper.sh:cap_policy)

# Numeric ResourceType id -> family name, so a caller holding only the Spanner
# card code (money_check reads ids, not names) can resolve the same policy.
# Ids from //depot/google3/third_party/py/xmanager/xm/resources.py.
_TYPE_ID_TO_FAMILY: dict[int, str] = {
    # TPUs
    34: "v4",       # PUFFERFISH
    59: "v5p",      # VIPERFISH
    60: "v6e",      # (viperlite -> historical; v6e price rows show under 63/76)
    63: "v6e",      # GHOSTLITE_POD
    76: "v6e",      # GHOSTLITE_POD (alt code seen in the market feed)
    92: "v6p",      # GHOSTFISH
    101: "v7",      # GHOSTFISHLITE
    # NVIDIA GPUs
    46: "a100",         # GPU_TESLA_A100_40GIB
    66: "a100_80gib",   # GPU_TESLA_A100_80GIB
    70: "h100",         # GPU_NVIDIA_H100
    86: "h200",         # GPU_NVIDIA_H200
    87: "b200",         # GPU_NVIDIA_B200
    112: "b300",        # GPU_NVIDIA_B300
    89: "gb200",        # GPU_NVIDIA_GB200
    100: "gb300",       # GPU_NVIDIA_GB300
}


def cap_for_family(family: str):
  """Cap in credits/chip-hour for a family name, or None if no policy."""
  return CAP_POLICY.get((family or "").lower())


def cap_for_type_id(type_id: int):
  """Cap in credits/chip-hour for a numeric ResourceType id, or None.

  None means 'no policy for this card' -- the caller must treat it as uncapped,
  never as a zero cap (a 0 cap would pause the job at any nonzero price).
  """
  fam = _TYPE_ID_TO_FAMILY.get(int(type_id))
  if fam is None:
    return None
  return CAP_POLICY.get(fam)
