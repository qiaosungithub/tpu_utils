"""L1 check: local topology whitelist + per-alloc PROD min-slice rules.

Source of truth for topologies: `borg/common/locus_info.cc`. We keep a small
mirror table here because:
  (a) Borg's table is C++ and reading it at import time is expensive;
  (b) our wrapper only supports a handful of accelerators anyway;
  (c) mismatches will be caught later by the L2 check, not silently accepted.

If Borg adds a new accelerator, extend `_LOCUS_TABLE` below and refresh the
compat notes in xmanager.md.
"""

import dataclasses
from typing import Optional


# accelerator arch (lowercase) -> {chip_count: locus_shape_string}
# Locus shape is the string Borg expects, e.g. "4_4" (v6e-16), "2x2x2" (v4-8).
# For 3-D torus accelerators we store the shape without the wrap suffix; those
# are added at request-build time in xm_launcher.py.
_LOCUS_TABLE: dict[str, dict[int, str]] = {
    # v4 = pufferfish (3-D torus, 4 chips/machine)
    'v4': {8: '2x2x2', 16: '2x2x4', 32: '2x4x4', 64: '4x4x4',
           128: '4x4x8', 256: '4x8x8', 512: '4x8x16', 1024: '8x8x16', 2048: '8x16x16'},
    # v5p = viperfish (3-D torus, 4 chips/machine)
    'v5p': {8: '2x2x2', 16: '2x2x4', 32: '2x4x4', 64: '4x4x4',
            128: '4x4x8', 256: '4x8x8', 512: '4x8x16', 1024: '8x8x16'},
    # v6p = ghostfish (3-D torus)
    'v6p': {8: '2x2x2', 16: '2x2x4', 32: '2x4x4', 64: '4x4x4',
            128: '4x4x8', 256: '4x8x8', 512: '4x8x16'},
    # v7 = ghostfishlite (3-D torus, 4 chips/host). Same slice geometry as v6p:
    # platforms/accelerator_metadata/platforms/ghostfishlite.gcl declares the
    # identical static sub-cube list (2x2x1, 2x2x2, 2x2x4, 2x4x4) with the same
    # chips_per_host=4 and the same 4x4x4 entry commented out, and its dynamic
    # slice expansion is the same multiple-of-4 rule capped at 4 cubes. The two
    # files differ only in the locus name (DEPLOYMENT_TYPE_GHOSTFISH_LITE).
    # 64+ needs dynamic slice creation (OCS manager), so stop at 32 until a
    # larger shape is actually exercised.
    'v7': {4: '2x2x1', 8: '2x2x2', 16: '2x2x4', 32: '2x4x4'},
    # v6e = ghostlite_pod (2-D torus, 8 chips/machine, pod = 16x16)
    'v6e': {8: '2_4', 16: '4_4', 32: '4_8', 64: '8_8',
            128: '8_16_wrap_y', 256: '16_16_wrap_xy'},
    # v5e = viperlite_pod (2-D)
    'v5e': {8: '2_4', 16: '4_4', 32: '4_8', 64: '8_8'},
    # v4lite = puffylite (dragonfish)
    'v4lite': {},  # Explicitly no support; historically rejects slice=8.
}

# NVIDIA GPUs are NOT a torus. There is no locus SHAPE string to validate the
# way a TPU 2x4x4 must be checked -- Borg takes a scalar device count plus an
# NVLink-domain grouping the allocator resolves. So the "legal shape" here is
# simply the chip count, and the value we carry is the scalar as a string
# (used only for display / the request kwarg, never remapped to a torus).
#
# The cap in each dict is the card's NVLINK DOMAIN (device_group in the
# platform GCL, platforms/accelerator_metadata/platforms/*.gcl): the largest
# single-node fully-NVLink-connected slice. Larger asks ARE legal (chips talk
# over network RDMA past the domain) but are the caller's responsibility, so we
# whitelist the common in-domain sizes and let anything bigger fall to the L2
# capacity check rather than hard-blocking here.
_GPU_LEGAL: dict[str, dict[int, str]] = {
    'a100':       {1: '1', 2: '2', 4: '4', 8: '8', 16: '16'},        # domain 16
    'a100_80gib': {1: '1', 2: '2', 4: '4', 8: '8'},                  # domain 8
    'h100':       {1: '1', 2: '2', 4: '4', 8: '8'},                  # domain 8
    'h200':       {1: '1', 2: '2', 4: '4', 8: '8'},                  # domain 8
    'b200':       {1: '1', 2: '2', 4: '4', 8: '8'},                  # domain 8
    'b300':       {1: '1', 2: '2', 4: '4', 8: '8'},                  # domain 8
    'gb200':      {1: '1', 2: '2', 4: '4', 8: '8', 16: '16', 32: '32', 64: '64', 72: '72'},  # NVL72
    'gb300':      {1: '1', 2: '2', 4: '4', 8: '8', 16: '16', 32: '32', 64: '64', 72: '72'},  # NVL72
}
_GPU_ARCHS = frozenset(_GPU_LEGAL)
_LOCUS_TABLE.update(_GPU_LEGAL)

# Borg ScalarResource.Key enum name for each arch. Used when calling
# GoodputService.GetCellAvailability.
BORG_PLATFORM_KEY: dict[str, str] = {
    'v4': 'PUFFERFISH',
    'v5p': 'VIPERFISH',
    'v6p': 'GHOSTFISH',
    'v6e': 'GHOSTLITE_POD',
    'v5e': 'VIPERLITE_POD',
    # GHOSTFISHLITE (101) is v7, NOT v5e/v6e -- see quota_check.py's note.
    'v7': 'GHOSTFISHLITE',
    # NVIDIA GPUs: ScalarResource.Key enum names, for GetCellAvailability.
    'a100': 'GPU_TESLA_A100_40GIB',
    'a100_80gib': 'GPU_TESLA_A100_80GIB',
    'h100': 'GPU_NVIDIA_H100',
    'h200': 'GPU_NVIDIA_H200',
    'b200': 'GPU_NVIDIA_B200',
    'b300': 'GPU_NVIDIA_B300',
    'gb200': 'GPU_NVIDIA_GB200',
    'gb300': 'GPU_NVIDIA_GB300',
}

# XManager-side codenames used by xm.JobRequirements() (matches money_check.py
# `map_tpu_types`). Preserved for `resource_service` cross-references.
XM_ACCELERATOR_KEY: dict[str, str] = {
    'v4': 'tpu_pufferfish',
    'v5p': 'tpu_viperfish',
    'v6p': 'tpu_ghostfish',
    'v6e': 'tpu_ghostlite_pod',
    'v5e': 'tpu_viperlite_pod',
    'v7': 'tpu_ghostfishlite',
    # NVIDIA GPUs: proto field names in the ResourceSet (resource_model.proto),
    # matching quota_check.GPU_DISPLAY_NAMES keys.
    'a100': 'gpu_a100',
    'a100_80gib': 'gpu_a100_80gib',
    'h100': 'gpu_h100',
    'h200': 'gpu_h200',
    'b200': 'gpu_b200',
    'b300': 'gpu_b300',
    'gb200': 'gpu_gb200',
    'gb300': 'gpu_gb300',
}

# Per-allocator hard minimum slice size overrides. These are POOL POLICIES,
# not physical Borg rules. Requesting below the minimum causes an immediate
# allocator reject with no useful error message.
#
# Rules gathered from wiki_agents/xmanager.md and hardened over time:
#   - `deepmind-dynamic` root alloc: PROD requires min 16 chips for v4/v5p,
#     min 16 for v6e (the 4x4 shape).
#   - `vqfree-xm`: same as deepmind-dynamic (it's a child pool).
#
# Non-PROD tiers usually allow smaller shapes; we conservatively enforce
# arch-native minimums per §3 of the peppy-goose research (v4 min=8, v6e min=8).
#
# The key is a substring matched against the alloc string. The value is a
# per-tier per-arch minimum slice.
_ALLOC_MIN_SLICE_RULES: dict[str, dict[tuple[str, str], int]] = {
    'deepmind-dynamic': {
        ('PROD', 'v4'): 16,
        ('PROD', 'v5p'): 16,
        ('PROD', 'v6e'): 16,
        ('PROD', 'v6p'): 16,
        ('BATCH', 'v4'): 8,
        ('BATCH', 'v6e'): 8,
    },
    # Fallback minimums applied when no more-specific rule matches.
    '': {
        ('PROD', 'v4'): 8,
        ('PROD', 'v5p'): 8,
        ('PROD', 'v6e'): 4,
    },
}


@dataclasses.dataclass(frozen=True)
class TopologyResult:
  """Outcome of the L1 topology check."""
  ok: bool
  locus_shape: Optional[str]        # e.g. "4_4"; None if invalid
  borg_platform_key: Optional[str]  # e.g. "GHOSTLITE_POD"
  xm_accelerator_key: Optional[str] # e.g. "tpu_ghostlite_pod"
  arch: str                         # normalized: 'v6e', ...
  chips: int
  hard_error: Optional[str] = None  # non-empty → RED (do not submit)
  warnings: tuple[str, ...] = ()    # non-empty → YELLOW


def parse_tpu_type(tpu_type: str) -> tuple[str, int]:
  """Parses 'v6e-16' into ('v6e', 16). Raises ValueError on bad input."""
  s = tpu_type.strip().lower()
  # Accept both 'v6e-16' and 'v6e=16' (legacy). Not accepting 'ghostlite_pod-16'
  # because our whole wrapper canonicalises on the marketing name.
  for sep in ('-', '='):
    if sep in s:
      arch, cores = s.split(sep, 1)
      break
  else:
    raise ValueError(f"tpu_type '{tpu_type}' missing '-'; expected e.g. 'v6e-16'")
  try:
    return arch, int(cores)
  except ValueError as e:
    raise ValueError(f"tpu_type '{tpu_type}' cores must be an integer") from e


def _lookup_min_slice(alloc: str, tier: str, arch: str) -> Optional[int]:
  """Returns the enforced min slice for (alloc, tier, arch), or None."""
  tier_up = (tier or '').upper()
  arch_lc = (arch or '').lower()
  # Try most-specific alloc match first.
  for pool_substr, rules in _ALLOC_MIN_SLICE_RULES.items():
    if pool_substr and pool_substr in alloc:
      if (tier_up, arch_lc) in rules:
        return rules[(tier_up, arch_lc)]
  # Fallback bucket.
  return _ALLOC_MIN_SLICE_RULES.get('', {}).get((tier_up, arch_lc))


def check_topology(tpu_type: str, alloc: str, tier: str) -> TopologyResult:
  """L1 check: validate topology + per-alloc min-slice rules.

  Args:
    tpu_type: e.g. 'v6e-16'.
    alloc:    e.g. 'group:deepmind-dynamic/vqfree-xm'.
    tier:     'PROD' | 'BATCH' | '' (default treated as unknown; we still
              validate topology but skip PROD-min-slice check).

  Returns a TopologyResult. `hard_error` non-empty means we should not submit.
  """
  try:
    arch, chips = parse_tpu_type(tpu_type)
  except ValueError as e:
    return TopologyResult(
        ok=False, locus_shape=None, borg_platform_key=None,
        xm_accelerator_key=None, arch=str(tpu_type), chips=0,
        hard_error=f"Invalid tpu_type: {e}")

  arch_lc = arch.lower()
  legal = _LOCUS_TABLE.get(arch_lc)
  if legal is None:
    return TopologyResult(
        ok=False, locus_shape=None, borg_platform_key=None,
        xm_accelerator_key=None, arch=arch_lc, chips=chips,
        hard_error=(f"Unknown accelerator arch '{arch_lc}'. Known: "
                    f"{sorted(_LOCUS_TABLE.keys())}"))

  if chips not in legal:
    legal_sizes = sorted(legal.keys())
    return TopologyResult(
        ok=False, locus_shape=None,
        borg_platform_key=BORG_PLATFORM_KEY.get(arch_lc),
        xm_accelerator_key=XM_ACCELERATOR_KEY.get(arch_lc),
        arch=arch_lc, chips=chips,
        hard_error=(f"{arch_lc}-{chips} is not a supported slice size "
                    f"(Borg has no legal locus for it). "
                    f"Supported sizes for {arch_lc}: {legal_sizes}"))

  # Chips + arch OK. Now check per-alloc/tier min-slice policy.
  min_slice = _lookup_min_slice(alloc, tier, arch_lc)
  if min_slice is not None and chips < min_slice:
    return TopologyResult(
        ok=False, locus_shape=legal[chips],
        borg_platform_key=BORG_PLATFORM_KEY.get(arch_lc),
        xm_accelerator_key=XM_ACCELERATOR_KEY.get(arch_lc),
        arch=arch_lc, chips=chips,
        hard_error=(f"Allocator '{alloc}' at tier {tier} enforces a min "
                    f"slice of {arch_lc}-{min_slice}, but you requested "
                    f"{arch_lc}-{chips}. Requesting below the minimum is "
                    f"immediately rejected by the allocator. "
                    f"Try {arch_lc}-{min_slice} or a different alloc."))

  return TopologyResult(
      ok=True, locus_shape=legal[chips],
      borg_platform_key=BORG_PLATFORM_KEY.get(arch_lc),
      xm_accelerator_key=XM_ACCELERATOR_KEY.get(arch_lc),
      arch=arch_lc, chips=chips)


def legal_sizes_for(arch: str) -> list[int]:
  """Public helper: legal chip counts for an arch (empty if unknown arch)."""
  return sorted(_LOCUS_TABLE.get(arch.lower(), {}).keys())
