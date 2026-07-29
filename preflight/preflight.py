"""Preflight orchestrator.

Public entry point: `run_preflight(tpu_type, alloc, tier) -> Verdict`.
"""

import dataclasses
import enum
from typing import Optional

from google3.experimental.users.qiaos.tpu_utils.preflight import capacity as _capacity_mod
from google3.experimental.users.qiaos.tpu_utils.preflight import topology as _topology_mod


class Status(enum.Enum):
  GREEN = 'GREEN'    # go ahead, no known blockers
  YELLOW = 'YELLOW'  # submit is possible but risky (warn + pause)
  RED = 'RED'        # do not submit; will be rejected


@dataclasses.dataclass(frozen=True)
class Verdict:
  """Merged verdict from L1 + L2 checks."""
  status: Status
  reasons: tuple[str, ...]        # human-readable, one line each
  topology: Optional[_topology_mod.TopologyResult] = None
  capacity: Optional[_capacity_mod.CapacityResult] = None

  def as_console_lines(self) -> list[str]:
    """Format for CLI display (color codes handled by caller)."""
    icon = {'GREEN': '✅', 'YELLOW': '⚠️ ', 'RED': '❌'}[self.status.value]
    lines = [f"{icon} preflight: {self.status.value}"]
    for r in self.reasons:
      lines.append(f"    · {r}")
    if self.capacity:
      cap = self.capacity
      if cap.alloc_scoped_quota > 0:
        remaining = max(0, cap.alloc_scoped_quota - cap.alloc_scoped_used)
        lines.append(
            f"    · alloc quota: {cap.alloc_scoped_quota} chips "
            f"(used={cap.alloc_scoped_used}, remaining={remaining})")
      if cap.cells_ok:
        top = ', '.join(f"{c.cell}({c.obtainable})"
                        for c in cap.cells_ok[:5])
        lines.append(f"    · candidate cells (chips obtainable): {top}")
    return lines


def run_preflight(tpu_type: str, alloc: str, tier: str,
                  skip_capacity: bool = False) -> Verdict:
  """Runs L1 (topology) + L2 (capacity) checks in sequence.

  Args:
    tpu_type: e.g. 'v6e-16'.
    alloc:    e.g. 'group:deepmind-dynamic/vqfree-xm'.
    tier:     'PROD' | 'BATCH' | ''.
    skip_capacity: for unit tests / offline mode, skip the RPC.

  Returns a Verdict. RED means the caller should NOT submit (unless the user
  passes --force). YELLOW should be surfaced but the submit is allowed.
  """
  # L1: topology + min-slice.
  topo = _topology_mod.check_topology(tpu_type, alloc, tier)
  if not topo.ok:
    return Verdict(status=Status.RED,
                   reasons=(topo.hard_error or 'topology check failed',),
                   topology=topo)

  reasons: list[str] = []
  if topo.warnings:
    reasons.extend(topo.warnings)

  if skip_capacity:
    reasons.append('capacity check skipped (offline)')
    status = Status.YELLOW if reasons else Status.GREEN
    return Verdict(status=status, reasons=tuple(reasons), topology=topo)

  # L2: capacity. After L1 pass, xm/borg keys are guaranteed non-None.
  assert topo.xm_accelerator_key is not None
  assert topo.borg_platform_key is not None
  cap = _capacity_mod.check_capacity(
      alloc=alloc, tier=tier,
      xm_accelerator_key=topo.xm_accelerator_key,
      borg_platform_key=topo.borg_platform_key,
      chips_required=topo.chips)

  if not cap.ok:
    reasons.append(cap.hard_error or 'capacity check failed')
    # A capacity RPC failure is not always RED — it could be a transient auth
    # blip. But if the response was parsed and we still have 0 fit cells, that
    # IS RED. Distinguish by whether we have any cell rows at all.
    has_any_signal = (cap.total_pool_capacity > 0 or cap.cells_insufficient)
    if has_any_signal:
      status = Status.RED
    else:
      # RPC probably failed or returned empty; downgrade to YELLOW so the user
      # can still --force if they think it's a false alarm.
      status = Status.YELLOW
      reasons.append('capacity signal unavailable; treating as YELLOW warning')
    return Verdict(status=status, reasons=tuple(reasons),
                   topology=topo, capacity=cap)

  if cap.warnings:
    reasons.extend(cap.warnings)

  status = Status.YELLOW if reasons else Status.GREEN
  return Verdict(status=status, reasons=tuple(reasons),
                 topology=topo, capacity=cap)
