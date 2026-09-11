"""Single source of truth for our per-accelerator price caps (limit orders).

Operator directive (2026-08-25): every job this workstation launches is capped
at a FIXED price per chip-hour by accelerator family -- no external override.
The same table is what `tpu money` displays and colours against, so what we
enforce and what we show can never drift apart.

Cap is an ABSOLUTE price in credits per CHIP-hour, PROD tier. BATCH clears at
~0 and its admission never reads a price, so a cap there is inert -- we do not
apply caps to BATCH jobs (money still displays the policy for reference).

Keep in sync with the launcher's hardcoded table:
LINT.IfChange
"""

# family -> credits/chip-hour. The one place these numbers live.
#
# ★GPU rows are TIERED BY COMPUTE AGAINST v5p, so a card and the TPU it matches
# carry the SAME cap. Per-chip capability (v5p=1, from the `vle` field in
# gxus_by_platform_ga.textproto -- see tpu_reference.md §Per-Chip Capability):
# a100 0.68 -> v5p tier; h100/h200 2.15 ~= v6e -> v6e tier; b200/b300/gb200/
# gb300 4.90 ~= v6p/v7 -> v6p tier. A GPU "chip" is one device, so the cap is
# per GPU exactly as it is per TPU chip.
#
# These GPU values are NOT new policy: they already existed in
# tpu_wrapper.sh::_tpu_limit_price_for_arch and in wiki_agents/tools/
# budget_check.py, and are copied here byte for byte. What was missing is that
# the file calling itself the single source of truth did not carry them, so
# cap_for_family('h100') returned None and every GPU caller silently ran
# uncapped. Adding a row that already governs the launcher is not raising a cap.
CAP_POLICY = {
    "v7": 20,
    "v6p": 20,
    "v6e": 10,
    "v5p": 5,
    "v4": 5,
    # NVIDIA GPUs, same tiers by compute (see note above).
    "a100": 5,
    "a100_80gib": 5,
    "h100": 10,
    "h200": 10,
    "b200": 20,
    "b300": 20,
    "gb200": 20,
    "gb300": 20,
}
# LINT.ThenChange(//depot/google3/experimental/users/qiaos/tpu_cmd/tpu_wrapper.sh)

# GQM keys accelerators by numeric product id; map id -> family so a caller
# holding a market.json / TARGET_CARDS id can resolve the policy. Ids per the
# authoritative table in wiki_agents/tpu_reference.md: VIPERFISH(59)=v5p,
# VIPERLITE(60)=v5e (NOT the swapped pair some older tools carry), GHOSTFISH(92)
# =v6p, GHOSTFISHLITE(101)=v7. v6e also appears as 63 in one pool cache.
_FAMILY_BY_ID = {34: "v4", 59: "v5p", 60: "v5e", 63: "v6e", 76: "v6e",
                 92: "v6p", 101: "v7",
                 # NVIDIA card codes, per tpu_reference.md §NVIDIA GPUs and
                 # money_check.py's _CARDS table (verified against
                 # borg/common/scalar_resource.proto enum names).
                 46: "a100", 66: "a100_80gib", 70: "h100", 86: "h200",
                 87: "b200", 112: "b300", 89: "gb200", 100: "gb300"}


def cap_for_family(family):
  """Cap for a family name like 'v7', or None if we have no policy for it."""
  return CAP_POLICY.get(family)


def cap_for_type_id(type_id):
  """Cap for a numeric GQM product id, or None if unknown / no policy."""
  fam = _FAMILY_BY_ID.get(int(type_id)) if type_id is not None else None
  return CAP_POLICY.get(fam) if fam else None
