"""Renders the TPU quota tables consumed by `tpu quota`.

Data-source contract (this is the part that is easy to get wrong):

* An alloc's OWN guaranteed quota lives in
  ``ResourceAllocationDetails.floor_v2`` -- a ResourcesByPriority keyed by
  "HighlyAvailable" / "NonProd" / "BestEffort".
* ``get_pool_capacity(pool)`` returns the capacity of the whole shared
  resource pool (e.g. ~425 cells of ``deepmind-dynamic-pool``). It is NOT
  this group's quota, and six of our nine groups share that single pool, so
  using it made every group display near-identical numbers.
* ``get_forecast_info(alloc)`` is alloc-scoped and estimates what is
  actually obtainable right now. That is a forecast, not a guarantee, so it
  is reported in its own column rather than as "quota".
* All ``tpu_*`` fields in resource_model.proto are integer CHIP COUNTS.
  There is no milli-unit scaling; do not divide by 1000.

For ``deepmind-dynamic/*`` allocs the floor is recomputed continuously to
track live usage, so it drifts between refreshes and "Used" can briefly
exceed "Quota". That is expected for a dynamic alloc, not a rendering bug.
"""

import concurrent.futures
import os

from absl import app
from rich.console import Console
from rich.table import Table

from google3.experimental.users.qiaos.tpu_utils import group_utils
from google3.learning.deepmind.xmanager2.client import resource_service

TIERS = ("HighlyAvailable", "NonProd", "BestEffort")

DISPLAY_NAMES = {
    "HighlyAvailable": "PROD",
    "NonProd": "BATCH",
    "BestEffort": "SPOT",
}

TPU_DISPLAY_NAMES = {
    "tpu_jellyfish": "TPU v2/v3",
    "tpu_jellydonut": "TPU v2/v3 Donut",
    "tpu_dragonfish": "TPU v3",
    "tpu_dragondonut": "TPU v3 Donut",
    "tpu_pufferfish": "TPU v4",
    "tpu_puffylite": "TPU v4 Lite (v4i)",
    "tpu_viperlite": "TPU v5e",
    "tpu_viperlite_pod": "TPU v5e Pod",
    "tpu_viperfish": "TPU v5p",
    "tpu_ghostlite_pod": "TPU v6e",
    "tpu_ghostfish": "TPU v6p",
    # GHOSTFISHLITE (101) is v7, NOT another name for v6e. It used to be merged
    # into the v6e row, which hid v7 capacity entirely.
    "tpu_ghostfishlite": "TPU v7",
    "tpu_sunfish_sampling": "TPU Sunfish (sampling)",
    "tpu_zebrafish": "TPU Zebrafish",
}

# Legacy accelerators we never schedule on; hidden to keep the table short.
EXCLUDED_TPU_KEYS = {
    "tpu_jellyfish",
    "tpu_jellydonut",
    "tpu_dragonfish",
    "tpu_dragondonut",
    "tpu_puffylite",
}

CACHE_DIR = os.path.expanduser("~/.tpu_quota_cache_dir")

# Tiers that are only worth printing when they carry a real signal.
#
# PROD is always shown: its floor is the number that actually gates admission.
# BATCH and SPOT are different -- for these allocs BATCH floors are ~0 across
# the board and SPOT has no floor at all by construction, so both sections
# rendered a dozen rows of "0 / 0 / 0" (or "Spot / - / Spot") and pushed the
# one informative tier off the top of the screen. They are hidden unless some
# row has a nonzero quota or nonzero live usage; a footnote records the fact
# so a suddenly-granted BATCH floor is not silently invisible.
OPTIONAL_TIERS = ("NonProd", "BestEffort")


def _tier_has_signal(type_dict):
  """True if any row in this tier has a real floor or live usage."""
  return any(
      stats.get("quota", 0.0) > 0 or stats.get("used", 0.0) > 0
      for stats in (type_dict or {}).values()
  )


def map_tpu_types(tpu_type):
  return TPU_DISPLAY_NAMES.get(tpu_type, tpu_type)


def get_tpu_map(res_set):
  """Returns {proto_field_name: chip_count} for a ResourceSet.

  Values are raw chip counts straight from the proto -- see the module
  docstring on why there is no /1000 here.
  """
  tpus = {}
  if not res_set:
    return tpus
  for field in res_set.DESCRIPTOR.fields:
    if not field.name.startswith("tpu_"):
      continue
    val = getattr(res_set, field.name)
    if val:
      tpus[field.name] = float(val)
  return tpus


def _blank_agg():
  return {tier: {} for tier in TIERS}


def _bucket(agg, tier, tpu_type):
  """Returns the accumulator dict for one (tier, tpu type), creating it."""
  friendly = map_tpu_types(tpu_type)
  return agg[tier].setdefault(
      friendly, {"quota": 0.0, "used": 0.0, "obtainable": 0.0}
  )


def generate_table(title, agg_dict, obtainable_is_max=False):
  """Builds the rich table for one alloc (or the global roll-up).

  Returns (table, hidden_tiers) where hidden_tiers lists the display names of
  the optional tiers that were suppressed for carrying no signal.
  """
  table = Table(title=title, show_header=True, header_style="bold magenta")
  table.add_column("Tier")
  table.add_column("TPU Type")
  table.add_column("Quota", justify="right")
  table.add_column("Used", justify="right")
  table.add_column("Available", justify="right")
  table.add_column("Obtainable*" if obtainable_is_max else "Obtainable",
                   justify="right")

  hidden_tiers = []
  for tier in TIERS:
    type_dict = agg_dict.get(tier) or {}
    if not type_dict:
      continue
    if tier in OPTIONAL_TIERS and not _tier_has_signal(type_dict):
      hidden_tiers.append(DISPLAY_NAMES[tier])
      continue
    tier_printed = False

    for friendly_tpu, stats in sorted(type_dict.items()):
      quota = stats["quota"]
      used = stats["used"]
      obtainable = stats["obtainable"]
      available = max(0.0, quota - used)

      quota_str = f"{quota:,.0f}"
      # Dynamic allocs re-derive their floor from live usage, so a sample can
      # show used slightly above quota. Flag it instead of clamping silently.
      if tier != "BestEffort" and used > quota:
        quota_str += " [yellow]~[/yellow]"
      origin = stats.get("quota_origin", "")
      if origin:
        quota_str += f" {origin}"
      used_str = f"{used:,.0f}"
      avail_color = "green" if available > 0 else "red"
      avail_str = f"[{avail_color}]{available:,.0f}[/{avail_color}]"

      if tier == "BestEffort":
        # SPOT is preemptible free-pool capacity: there is no guaranteed
        # floor, so quota/available would be misleading.
        quota_str = "Spot"
        avail_str = "Spot"
        used_str = "-" if not used else f"{used:,.0f}"

      obt_color = "green" if obtainable > 0 else "red"
      obt_str = f"[{obt_color}]{obtainable:,.0f}[/{obt_color}]"

      disp_tier = (
          f"[bold]{DISPLAY_NAMES[tier]}[/bold]" if not tier_printed else ""
      )
      tier_printed = True

      table.add_row(disp_tier, friendly_tpu, quota_str, used_str, avail_str,
                    obt_str)
  return table, hidden_tiers


def _hidden_note(hidden_tiers):
  """Renders the one-line footnote for suppressed tiers (or '')."""
  if not hidden_tiers:
    return ""
  return (
      f"[dim]{' + '.join(hidden_tiers)} hidden: no quota floor and no live "
      "usage. They reappear automatically once either becomes nonzero.[/dim]"
  )


def write_to_cache(filename, content):
  with open(os.path.join(CACHE_DIR, filename), "w", encoding="utf-8") as f:
    f.write(content)


def _process_one(idx_and_alloc):
  """Fetches quota (floor_v2), usage and forecast for a single alloc."""
  idx, alloc_name = idx_and_alloc
  agg = _blank_agg()
  has_tpu = False

  # 1. Quota == this alloc's own guaranteed floor.
  try:
    details = resource_service.get_resource_alloc(
        resource_alloc_name=alloc_name
    )
    for tier in TIERS:
      res_set = details.floor_v2.priorities.get(tier)
      for tpu_type, chips in get_tpu_map(res_set).items():
        if tpu_type in EXCLUDED_TPU_KEYS:
          continue
        _bucket(agg, tier, tpu_type)["quota"] += chips
        has_tpu = True
  except Exception:  # pylint: disable=broad-except
    pass

  # 2. Live usage, per tier.
  for tier in TIERS:
    try:
      usage_set = resource_service.get_resource_usage(alloc_name, [tier])
      for tpu_type, chips in get_tpu_map(usage_set).items():
        if tpu_type in EXCLUDED_TPU_KEYS:
          continue
        _bucket(agg, tier, tpu_type)["used"] += chips
        has_tpu = True
    except Exception:  # pylint: disable=broad-except
      pass

  # 3. Alloc-scoped forecast of what is obtainable right now.
  try:
    forecast = resource_service.get_forecast_info(
        resource_alloc_name=alloc_name,
        with_global_batch_accelerators_availability=True,
        with_full_availability=True,
    )
    for _cell, by_priority in forecast.forecast.items():
      for tier, res_set in by_priority.priorities.items():
        if tier not in agg:
          continue
        for tpu_type, chips in get_tpu_map(res_set).items():
          if tpu_type in EXCLUDED_TPU_KEYS:
            continue
          _bucket(agg, tier, tpu_type)["obtainable"] += chips
          has_tpu = True
  except Exception:  # pylint: disable=broad-except
    pass

  return idx, (agg if has_tpu else None)


def _build_global_agg(group_aggs):
  """Rolls the per-group aggregates up into one table.

  Quota and usage are summed: each alloc has its own distinct floor. The
  obtainable forecast is NOT summed -- most of our groups sit in the same
  shared pool, so adding their forecasts would count the same free chips
  several times. The max across groups is the honest roll-up.
  """
  global_agg = _blank_agg()
  for tier in TIERS:
    tpu_names = set()
    for agg in group_aggs.values():
      tpu_names.update(agg[tier].keys())

    for name in tpu_names:
      per_group = {
          idx: agg[tier].get(name) for idx, agg in group_aggs.items()
      }
      per_group = {i: s for i, s in per_group.items() if s}

      total_quota = sum(s["quota"] for s in per_group.values())
      total_used = sum(s["used"] for s in per_group.values())
      max_obtainable = max(
          (s["obtainable"] for s in per_group.values()), default=0.0
      )

      origin = ""
      if tier != "BestEffort" and total_quota > 0:
        top_idx = max(per_group, key=lambda i: per_group[i]["quota"])
        origin = f"\\[G{top_idx}]"

      global_agg[tier][name] = {
          "quota": total_quota,
          "used": total_used,
          "obtainable": max_obtainable,
          "quota_origin": origin,
      }
  return global_agg


def main(argv):
  del argv
  os.makedirs(CACHE_DIR, exist_ok=True)

  group_mapping = group_utils.get_group_mapping()[1]

  # Each alloc needs ~5 independent RPCs; fan the allocs out in parallel so
  # the whole refresh stays close to the latency of a single alloc.
  group_aggs = {}
  with concurrent.futures.ThreadPoolExecutor(max_workers=9) as pool:
    for idx, agg in pool.map(_process_one, list(group_mapping.items())):
      if agg is not None:
        group_aggs[idx] = agg

  if not group_aggs:
    write_to_cache(
        "default.txt",
        "No resource allocations returned for any known group. This is "
        "usually a transient RPC issue -- rerun in a minute. If it "
        "persists, run 'gcert' to refresh credentials.\n",
    )
    return

  # height is required alongside width: rich's Console.size ignores an
  # explicit width unless height is also set, so under TERM=dumb (a non-tty
  # daemon or agent shell) it would silently fall back to 80x25 and wrap the
  # quota tables.
  mem_console = Console(width=150, height=200, force_terminal=True,
                        color_system="standard")

  def render(renderable):
    with mem_console.capture() as cap:
      mem_console.print(renderable)
    return cap.get()

  legend = (
      "[dim]Quota = your alloc's guaranteed floor (chips). "
      "Obtainable = alloc-scoped forecast of what is schedulable now.[/dim]"
  )

  # 1. Default view: group index plus the global roll-up.
  out_default = render("[bold cyan]=== Available Groups ===[/bold cyan]")
  for idx, name in group_mapping.items():
    if idx in group_aggs:
      out_default += render(f"[bold]\\[G{idx}][/bold] {name}")
  out_default += "\n"
  global_table, global_hidden = generate_table(
      "TPU Quota [Sum of All Groups]",
      _build_global_agg(group_aggs),
      obtainable_is_max=True,
  )
  out_default += render(global_table)
  out_default += render(legend)
  if global_hidden:
    out_default += render(_hidden_note(global_hidden))
  out_default += render(
      "[dim]* Obtainable is the max across groups, not a sum: most groups "
      "share one pool, so summing would double-count the same chips.[/dim]"
  )
  out_default += render(
      "[dim]Use 'tpu quota -l' to list all groups, or 'tpu quota -g 1' for a "
      "specific group.[/dim]"
  )
  write_to_cache("default.txt", out_default)

  # 2. 'tpu quota -l' -- every group, one table each.
  out_l = ""
  for idx in sorted(group_aggs):
    table, hidden = generate_table(
        f"\\[G{idx}] {group_mapping[idx]}", group_aggs[idx])
    out_l += render(table)
    if hidden:
      out_l += render(_hidden_note(hidden))
    out_l += "\n"
  out_l += render(legend)
  write_to_cache("list.txt", out_l)

  # 3. 'tpu quota -g X' -- one file per group.
  for idx in sorted(group_aggs):
    table, hidden = generate_table(
        f"\\[G{idx}] {group_mapping[idx]}", group_aggs[idx])
    content = render(table)
    content += render(legend)
    if hidden:
      content += render(_hidden_note(hidden))
    write_to_cache(f"g{idx}.txt", content)


if __name__ == "__main__":
  app.run(main)
