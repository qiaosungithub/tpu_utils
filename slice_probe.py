"""How many v7-32 slices can actually be PLACED, per cell, right now.

WHY THIS EXISTS, when `tpu preflight --json` already prints a per-cell chip
count: obtainable chips are not a placeable slice. Preflight listed `yutulpz`
with 1616 obtainable v7 chips while the production run sat in that same cell
being descheduled with `GQM_OVERSOLD_MARKET ... deficit of GHOSTFISHLITE`,
because the free chips were fragmented and no contiguous 2x4x4 existed. Chips
answer "is there quota"; this answers "will the allocator find a slice".

Reads only. Prints one line per cell, most slices first.

    blaze run :slice_probe -- --topology=2_4_4 --group=9
"""

import datetime

from absl import app
from absl import flags

from google3.borg.common import scalar_resource_pb2
from google3.borg.xborg.frontend.goodput_optimizer.proto import goodput_optimizer_service_pb2
from google3.learning.deepmind.xmanager2.client import resource_service
from google3.net.rpc.python.contrib import rpc_factory_factory
from google3.net.rpc2.contrib.smartservice.python import smartservice_util

# The same map the tpu wrapper uses, so --group=9 means here what it means there.
_GROUP_MAP = {
    '1': 'group:deepmind-dynamic/gdm-resources-prod-shared-users-dynamic',
    '2': 'group:deepmind-dynamic/gdm-viscam-goflow-dynamic',
    '3': 'group:deepmind-dynamic/gdm-viscam-interns-dynamic',
    '4': 'group:deepmind-dynamic/viscam-interns',
    '5': 'group:deepmind-dynamic/vqfree-xm',
    '6': 'group:dm/deepmind-large-scale-workshop',
    '7': 'group:dm/dm-resources-prod-shared',
    '8': 'group:gdm-aux/brain-vasp-shared-user-xm',
    '9': 'group:deepmind-dynamic/fr-dna-grand-challenge-team-resource',
}

# GHOSTFISHLITE is v7. It is NOT another name for v6e (GHOSTLITE_POD) -- the two
# were merged once in the money checker and the row read 4x the real capacity.
_PLATFORM = {
    'v7': 'GHOSTFISHLITE',
    'v6e': 'GHOSTLITE_POD',
    'v6p': 'GHOSTFISH',
}

_GROUP = flags.DEFINE_string('group', '9', 'Group number or a full alloc string.')
_ACCEL = flags.DEFINE_string('accel', 'v7', f'One of {sorted(_PLATFORM)}.')
_TOPOLOGY = flags.DEFINE_string(
    'topology', '2_4_4',
    'Slice shape, underscore-separated. v7-32 == 2_4_4 (2*4*4 = 32 chips).')
_METRO = flags.DEFINE_string(
    'metro', '', 'Optional substring filter on the cell name, e.g. "cbf".')


def _alloc() -> str:
  spec = (_GROUP.value or '').strip()
  return _GROUP_MAP.get(spec, spec)


def main(argv) -> None:
  del argv
  alloc = _alloc()
  platform_name = _PLATFORM[_ACCEL.value]
  # The proto enum arrives as an INT, not its name -- comparing
  # str(pc.platform) to 'GHOSTFISHLITE' silently matched nothing and
  # printed an empty table that read as 'no capacity anywhere'.
  platform_key = getattr(scalar_resource_pb2.ScalarResource.Key, platform_name)
  platform = int(platform_key)
  details = resource_service.get_resource_alloc(alloc)
  locus = f'locus:DEPLOYMENT_TYPE_{platform_name}:{_TOPOLOGY.value}'
  print(f'alloc={alloc}\npool={details.resource_pool_name} '
        f'allotment={details.xborg_allotment_name}\nlocus={locus} (platform enum {platform})\n')

  stub = smartservice_util.new_stub(
      goodput_optimizer_service_pb2.GoodputService,
      smartservice_util.parse('blade:xborg-prod-routing-layer'),
      rpc_factory=rpc_factory_factory.new_factory(
          deadline=datetime.timedelta(seconds=60)))

  req = goodput_optimizer_service_pb2.GetCellAvailabilityRequest()
  req.resource_pool = details.resource_pool_name
  req.allowed_allotments.append(details.xborg_allotment_name)
  req.platforms.append(platform_key)
  resp = stub.GetCellAvailability(req)

  chips_per_slice = 1
  for part in _TOPOLOGY.value.split('_'):
    chips_per_slice *= int(part)

  # Response shape (GetCellAvailabilityResponse, DYNAMIC pool):
  #   tiered_dynamic_pool_availabilities[]      one per tier
  #     .availability
  #       .allotment_availability[].obtainable_capacity[]  per-cell chips our
  #                                                        allotment may obtain
  #       .max_available_chips[]     per-cell chips ACTUALLY free right now
  #       .oversold_statuses[]       cells whose demand already exceeds supply
  #
  # THE TWO NUMBERS MEAN DIFFERENT THINGS AND THE SECOND IS THE ONE THAT
  # DECIDES. `obtainable_capacity` is a quota-shaped promise; `max_available_chips`
  # is what is free. yutulpz reported 1616 obtainable while holding 3 free
  # chips and an oversold flag -- and that is the cell where the production run
  # spent 4 h being descheduled.
  tiers = []
  for tiered in getattr(resp, 'tiered_dynamic_pool_availabilities', []):
    tier = str(getattr(tiered, 'tier', '?'))
    av = getattr(tiered, 'availability', None)
    if av is None:
      continue
    free, obtainable, oversold = {}, {}, set()
    for entry in getattr(av, 'max_available_chips', []):
      for pc in getattr(entry, 'platform_to_chip_counts', []):
        if int(getattr(pc, 'platform', -1)) == platform:
          free[entry.cell] = int(pc.num_chips)
    for aa in getattr(av, 'allotment_availability', []):
      for entry in getattr(aa, 'obtainable_capacity', []):
        for pc in getattr(entry, 'platform_to_chip_counts', []):
          if int(getattr(pc, 'platform', -1)) == platform:
            obtainable[entry.cell] = int(pc.num_chips)
    for entry in getattr(av, 'oversold_statuses', []):
      if platform in [int(p) for p in getattr(entry, 'platforms', [])]:
        oversold.add(entry.cell)
    tiers.append((tier, free, obtainable, oversold))

  if not tiers:
    print('COULD NOT PARSE the availability response. Raw follows -- read it by '
          'hand rather than trusting a zero.\n')
    print(str(resp)[:6000])
    return

  for tier, free, obtainable, oversold in tiers:
    print(f'=== tier {tier} ===')
    cells = sorted(set(free) | set(obtainable))
    if _METRO.value:
      cells = [c for c in cells if _METRO.value in c]
    rows = []
    for cell in cells:
      f = free.get(cell, 0)
      rows.append((f // chips_per_slice, f, obtainable.get(cell, 0),
                   cell in oversold, cell))
    rows.sort(reverse=True)
    print(f'{"slices":>7} {"free":>8} {"obtainable":>11}  cell')
    for slices, f, o, over, cell in rows:
      flag = '  <- OVERSOLD' if over else ''
      print(f'{slices:>7} {f:>8} {o:>11}  {cell}{flag}')
    print()

  print(f'"slices" = free chips // {chips_per_slice} ({_ACCEL.value}-'
        f'{chips_per_slice}). It is an UPPER BOUND: free chips can be '
        'fragmented across the cell with no contiguous slice among them, which '
        'is exactly how a cell with thousands of chips rejects one job.')


if __name__ == '__main__':
  app.run(main)
