"""Single Source of Truth for XManager Allocator Group Mappings.

Group ids (g1..g9) are STATIC, matching the hardcoded mapping in
``~/work/tpu_cmd/tpu_wrapper.sh::get_alloc_by_group_id``. Both the shell
wrapper and the Python CLIs (``tpu preflight``, ``tpu route``, ``tpu quota``,
``tpu money``) MUST use this same numbering, otherwise ``--group=5`` in one
tool would resolve to a different allocator than ``--group=5`` in another.

If the shell mapping changes, this table must change to match (and vice
versa). Consider generating one from the other in the future.
"""

# LINT.IfChange(group_map)
# Keep this in sync with ~/work/tpu_cmd/tpu_wrapper.sh::get_alloc_by_group_id.
GROUP_MAP: dict[int, str] = {
    1: "group:deepmind-dynamic/gdm-resources-prod-shared-users-dynamic",
    2: "group:deepmind-dynamic/gdm-viscam-goflow-dynamic",
    3: "group:deepmind-dynamic/gdm-viscam-interns-dynamic",
    4: "group:deepmind-dynamic/viscam-interns",
    5: "group:deepmind-dynamic/vqfree-xm",
    6: "group:dm/deepmind-large-scale-workshop",
    7: "group:dm/dm-resources-prod-shared",
    8: "group:gdm-aux/brain-vasp-shared-user-xm",
    9: "group:deepmind-dynamic/fr-dna-grand-challenge-team-resource",
}
# LINT.ThenChange(//depot/google3/../../../work/tpu_cmd/tpu_wrapper.sh)

_REVERSE_GROUP_MAP: dict[str, str] = {v: f"g{k}" for k, v in GROUP_MAP.items()}


def get_alloc_by_id(group_id):
  """Resolve 'g3' / '3' / 3 to the full 'group:...' allocation string.

  Returns "" if the id cannot be resolved.
  """
  try:
    gid = int(str(group_id).lower().replace("g", ""))
  except (ValueError, TypeError):
    return ""
  return GROUP_MAP.get(gid, "")


def get_group_id_by_alloc(alloc):
  """Reverse lookup: 'group:.../vqfree-xm' -> 'g5'.

  Falls back to the trailing name after 'group:' if the alloc isn't in the
  canonical map.
  """
  alloc_str = str(alloc or "").strip()
  if not alloc_str:
    return "-"
  if alloc_str in _REVERSE_GROUP_MAP:
    return _REVERSE_GROUP_MAP[alloc_str]
  for alloc_key, gid in _REVERSE_GROUP_MAP.items():
    if alloc_key in alloc_str or alloc_str in alloc_key:
      return gid
  if alloc_str.startswith("group:"):
    return alloc_str.replace("group:", "").split("/")[-1]
  return alloc_str or "-"


# Backwards-compat shim for callers that still expect (resources, mapping).
# quota_check / money_check use their own list_resources fetch; return an
# empty resources dict and the static map so they don't blow up.
def get_group_mapping():
  return {}, dict(GROUP_MAP)
