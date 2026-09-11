"""Offline unit tests for router ranking and limit-order blocking.

Builds Candidate objects directly so the ranking rules can be exercised with no
RPC, no daemon and no market cache. These are the rules that decide what the
user is told to run, so they must be pinned:

  * a limit-order-blocked combo must never outrank a runnable one;
  * PROD ranks on quota headroom, BATCH must NOT (its floor is never read);
  * cheaper cells win, but only after correctness-of-fit;
  * every group in GROUP_MAP is scanned (the g9 bug).

Run: blaze run experimental/users/qiaos/tpu_utils/preflight:router_test
"""

import sys
from typing import Any, Optional

from google3.experimental.users.qiaos.tpu_utils import group_utils
from google3.experimental.users.qiaos.tpu_utils.preflight import capacity
from google3.experimental.users.qiaos.tpu_utils.preflight import market
from google3.experimental.users.qiaos.tpu_utils.preflight import preflight
from google3.experimental.users.qiaos.tpu_utils.preflight import router

fails: list[str] = []
n_checks = 0


def check(name: str, got: Any, want: Any) -> None:
  global n_checks
  n_checks += 1
  if got != want:
    fails.append(f'{name}: got {got!r} want {want!r}')
  else:
    print(f'  ok  {name}')


def _cand(gid: int, arch: str, chips: int, *, quota: int = 0, used: int = 0,
          cell: str = 'aa', price: Optional[float] = 1.0,
          obtainable: int = 1000, blocked: bool = False,
          green: bool = True) -> router.Candidate:
  """A Candidate with just enough structure for rank() to read."""
  cap = capacity.CapacityResult(
      ok=True, cells_ok=(), cells_insufficient=(), total_pool_capacity=0,
      total_obtainable=obtainable, alloc_scoped_quota=quota,
      alloc_scoped_used=used, pool='deepmind-dynamic-pool')
  verdict = preflight.Verdict(
      status=preflight.Status.GREEN if green else preflight.Status.YELLOW,
      reasons=(), capacity=cap)
  return router.Candidate(
      group_id=gid, alloc=f'group:x/g{gid}', arch=arch, chips=chips,
      tpu_type=f'{arch}-{chips}', power_score=router.to_power(arch, chips),
      verdict=verdict, cell=cell, price=price,
      cost_per_hour=None if price is None else chips * price,
      obtainable=obtainable, blocked=blocked)


print('--- blocked never outranks runnable ---')
# The blocked one is better on every other axis: GREEN, huge headroom, free.
best_but_blocked = _cand(9, 'v6e', 16, quota=9999, price=0.0, blocked=True)
worse_but_runnable = _cand(1, 'v4', 32, quota=32, price=99.0, green=False)
order = router.rank([best_but_blocked, worse_but_runnable], tier='PROD')
check('runnable first despite worse on every other axis',
      [c.group_id for c in order], [1, 9])
check('blocked one is still returned, not dropped', len(order), 2)

print('--- PROD ranks on quota headroom ---')
lots = _cand(9, 'v5p', 32, quota=3200, used=0, price=5.0)
none_ = _cand(1, 'v5p', 32, quota=32, used=0, price=5.0)
check('more headroom wins',
      [c.group_id for c in router.rank([none_, lots], tier='PROD')], [9, 1])

print('--- PROD: verified floor beats unverifiable floor_v2==0 ---')
# Both have remaining==0, so plain headroom ties. quota==0 means "we could not
# read a floor at all", which is strictly less trustworthy than a spent floor.
spent = _cand(9, 'v5p', 32, quota=304, used=304)
unknown = _cand(2, 'v5p', 32, quota=0, used=0)
check('unverifiable sinks below spent-but-real',
      [c.group_id for c in router.rank([unknown, spent], tier='PROD')], [9, 2])

print('--- cheap cell wins when fit is equal ---')
dear = _cand(1, 'v5p', 32, quota=320, cell='nk', price=48.93)
cheap = _cand(2, 'v5p', 32, quota=320, cell='ea', price=22.23)
ranked = router.rank([dear, cheap], tier='PROD')
check('cheaper cell first', ranked[0].cell, 'ea')
check('cost_per_hour = chips * price', ranked[0].cost_per_hour, 32 * 22.23)

print('--- headroom outranks price (fit before thrift) ---')
cheap_no_room = _cand(1, 'v5p', 32, quota=0, price=0.0)
dear_with_room = _cand(9, 'v5p', 32, quota=3200, price=48.93)
check('a free cell does not buy a group that cannot fit the job',
      [c.group_id for c in router.rank([cheap_no_room, dear_with_room],
                                       tier='PROD')], [9, 1])

print('--- unpriced sorts after priced, never crashes ---')
priced = _cand(1, 'v5p', 32, quota=320, price=10.0)
unpriced = _cand(2, 'v5p', 32, quota=320, price=None)
r = router.rank([unpriced, priced], tier='PROD')
check('priced first', [c.group_id for c in r], [1, 2])
check('unpriced keeps cost None', r[1].cost_per_hour, None)

print('--- BATCH must NOT rank on quota ---')
# Same pool => same obtainable. Quota differs wildly but is meaningless at
# BATCH: that pass never consults floor_v2.
big_quota_few_chips = _cand(1, 'v5p', 32, quota=99999, obtainable=100)
no_quota_many_chips = _cand(9, 'v5p', 32, quota=0, obtainable=9000)
rb = router.rank([big_quota_few_chips, no_quota_many_chips], tier='BATCH')
check('BATCH picks more obtainable, ignoring the bigger floor',
      [c.group_id for c in rb], [9, 1])
rp = router.rank([big_quota_few_chips, no_quota_many_chips], tier='PROD')
check('same inputs at PROD pick the bigger floor (proves the split is real)',
      [c.group_id for c in rp], [1, 9])

print('--- BATCH ties are stable, not random ---')
ties = [_cand(g, 'v5p', 32, quota=0, obtainable=5699) for g in (3, 1, 2)]
check('same-pool BATCH ties resolve by group id',
      [c.group_id for c in router.rank(ties, tier='BATCH')], [1, 2, 3])

print('--- GREEN outranks YELLOW before anything else ---')
yellow_rich = _cand(1, 'v5p', 32, quota=99999, price=0.0, green=False)
green_poor = _cand(9, 'v5p', 32, quota=32, price=50.0, green=True)
check('GREEN first', [c.group_id for c in router.rank([yellow_rich, green_poor],
                                                      tier='PROD')], [9, 1])

print('--- PROD: g3/g5 spent before g9 (exempt from the G9 1/10 cap) ---')
# g9 is better on every economic axis (huge floor, free cell); g3/g5 must STILL
# win at PROD because their credit is not rationed by the income/10 gate.
g9_rich = _cand(9, 'v5p', 32, quota=999999, price=0.0)
g3_poor = _cand(3, 'v5p', 32, quota=0, price=99.0)
check('g3 outranks a richer, cheaper g9 at PROD',
      [c.group_id for c in router.rank([g9_rich, g3_poor], tier='PROD')], [3, 9])
g5_poor = _cand(5, 'v5p', 32, quota=0, price=99.0)
check('g5 too outranks g9 at PROD',
      [c.group_id for c in router.rank([g9_rich, g5_poor], tier='PROD')], [5, 9])
check('g3 and g5 tie on preference, then fall through to economics/group id',
      [c.group_id for c in router.rank(
          [_cand(5, 'v5p', 32, quota=0, price=10.0),
           _cand(3, 'v5p', 32, quota=0, price=10.0)], tier='PROD')], [3, 5])

print('--- group preference NEVER overrides placement correctness ---')
# A blocked or YELLOW g3 must not be promoted over a runnable/GREEN g9 just to
# save budget: correctness of placement sits above whose budget pays.
g3_blocked = _cand(3, 'v5p', 32, quota=0, price=0.0, blocked=True)
g9_runnable = _cand(9, 'v5p', 32, quota=32, price=50.0)
check('a blocked g3 still sinks below a runnable g9',
      [c.group_id for c in router.rank([g3_blocked, g9_runnable],
                                       tier='PROD')], [9, 3])
g3_yellow = _cand(3, 'v5p', 32, quota=0, price=0.0, green=False)
g9_green = _cand(9, 'v5p', 32, quota=32, price=50.0, green=True)
check('a YELLOW g3 still sinks below a GREEN g9 (confidence beats budget)',
      [c.group_id for c in router.rank([g3_yellow, g9_green],
                                       tier='PROD')], [9, 3])

print('--- group preference is neutral at BATCH (one free pool) ---')
# At BATCH nothing is spent and no floor is read, so g3 must NOT jump the queue;
# ordering falls to obtainable then group id, exactly as before this feature.
g9_more_obt = _cand(9, 'v5p', 32, quota=0, obtainable=9000)
g3_less_obt = _cand(3, 'v5p', 32, quota=0, obtainable=100)
check('BATCH ignores group preference, ranks on obtainable',
      [c.group_id for c in router.rank([g3_less_obt, g9_more_obt],
                                       tier='BATCH')], [9, 3])

print('--- group coverage: the g9 bug ---')
check('GROUP_MAP has 9 groups', len(group_utils.GROUP_MAP), 9)
check('default scan covers every group id, derived not hardcoded',
      sorted(group_utils.GROUP_MAP), list(range(1, 10)))
check('g9 resolves to the real dynamic alloc',
      group_utils.get_alloc_by_id(9),
      'group:deepmind-dynamic/fr-dna-grand-challenge-team-resource')
check('old buggy range(1,9) would have missed g9', 9 in list(range(1, 9)), False)

print('--- power parsing / equivalence ---')
check('v6e-16 == v5p-32 in power', router.parse_power_input('v6e-16'),
      router.parse_power_input('v5p-32'))
check('bare int', router.parse_power_input('32'), 32.0)
# Ratios come from Borg's per-chip `vle` unit, cross-checked against ART MXU
# bf16 FLOPs -- see router._V5P_MULTIPLIER. The old table said v4 == v5p and
# v6p == 2x v5p; both were wrong, so these values are the regression guard.
check('v4-32 == 19.2 v5p-equivalents (v4 is 0.60x, not 1.0x)',
      router.to_power('v4', 32), 19.2)
check('v6p is 4.34x v5p, not 2x', router.to_power('v6p', 1), 4.34)
check('v7 matches v6p chip-for-chip (same GFC chip)',
      router.to_power('v7', 8), router.to_power('v6p', 8))
check('v6p-8 ~ v5p-32 within routing tolerance',
      abs(router.to_power('v6p', 8) - router.to_power('v5p', 32)) < 5.0, True)
check('v7 is enumerated as a routing candidate at its legal sizes',
      ('v7', 16) in router._candidate_options(router.to_power('v7', 16)),  # pylint: disable=protected-access
      True)

print('--- _pick_offer: unobtainable cells are skipped ---')
offers = [
    router.CellOffer(cell='dead', obtainable=9999, price=None,
                     price_known=True, unobtainable=True, cost_per_hour=None),
    router.CellOffer(cell='ea', obtainable=100, price=22.23, price_known=True,
                     unobtainable=False, cost_per_hour=711.4),
]
picked = router._pick_offer(offers)  # pylint: disable=protected-access
assert picked is not None
check('skips the unobtainable cell even though it lists more chips',
      picked.cell, 'ea')
check('empty offer list -> None',
      router._pick_offer([]), None)  # pylint: disable=protected-access

print('--- metro_of resolves cells consistently (shared leaf) ---')
check('yu-prefixed cell -> its metro', router.metro_util.metro_of('yutulpz'),
      'tul')
check('override table wins for legacy short names',
      router.metro_util.metro_of('je'), 'cbf')
check('unknown cell is its own metro, never crashes',
      router.metro_util.metro_of('zz'), 'zz')

print('--- _evaluate_cells: --metro is a HARD in-metro filter ---')
# Two placeable cells in different metros; a metro allow-list must keep only the
# in-metro one, so the recommendation can never leave the data's metro.
_cap_two = capacity.CapacityResult(
    ok=True,
    cells_ok=(
        capacity.CellCapacity(cell='yucbfiv', tier='PROD', within_floor=64,
                              max_available=64, obtainable=64),
        capacity.CellCapacity(cell='yutulpz', tier='PROD', within_floor=64,
                              max_available=64, obtainable=64),
    ),
    cells_insufficient=(), total_pool_capacity=128, total_obtainable=128,
    alloc_scoped_quota=64, alloc_scoped_used=0, pool='deepmind-dynamic-pool')
_vd_two = preflight.Verdict(status=preflight.Status.GREEN, reasons=(),
                            capacity=_cap_two)

# An empty real snapshot (no price rows): the correct type, and with no quotes
# cell choice falls back to obtainable/order -- fine, since these cases assert
# WHICH cells survive the metro filter, not their pricing.
_snap = market.MarketSnapshot()
_all = router._evaluate_cells('v6p', 32, 'PROD', _vd_two, _snap)  # pylint: disable=protected-access
check('no filter -> both cells offered',
      sorted(o.cell for o in _all), ['yucbfiv', 'yutulpz'])
_cbf = router._evaluate_cells('v6p', 32, 'PROD', _vd_two, _snap,  # pylint: disable=protected-access
                             metros=['cbf'])
check('metro=cbf keeps only the in-metro cell',
      [o.cell for o in _cbf], ['yucbfiv'])
_multi = router._evaluate_cells('v6p', 32, 'PROD', _vd_two, _snap,  # pylint: disable=protected-access
                               metros=['cbf', 'tul'])
check('a metro LIST admits cells from every listed metro',
      sorted(o.cell for o in _multi), ['yucbfiv', 'yutulpz'])
_none = router._evaluate_cells('v6p', 32, 'PROD', _vd_two, _snap,  # pylint: disable=protected-access
                              metros=['dfw'])
check('an all-out-of-metro allow-list yields NO offer (fail-closed input)',
      _none, [])
_ws = router._evaluate_cells('v6p', 32, 'PROD', _vd_two, _snap,  # pylint: disable=protected-access
                            metros=['  CBF '])
check('metro tokens are whitespace- and case-insensitive',
      [o.cell for o in _ws], ['yucbfiv'])

print()
print('--- an explicit --groups turns OFF the g3/g5 budget preference ---')
# g9 is better on the economics (real floor, cheaper); g3 wins only because
# _GROUP_PREF sits above headroom in the sort key. That is right when the
# router is choosing whose budget to spend, and wrong when the caller already
# said which group they are pinned to.
_g9_better = _cand(9, 'v6e', 32, quota=256, price=1.0)
_g3_worse = _cand(3, 'v6e', 32, quota=0, price=9.0)
check('default (no --groups): g3/g5 preference still applies',
      [c.group_id for c in router.rank([_g9_better, _g3_worse],
                                       tier='PROD')],
      [3, 9])
check('explicit --groups: ranked on merit, g9 first',
      [c.group_id for c in router.rank([_g9_better, _g3_worse], tier='PROD',
                                       groups_were_explicit=True)],
      [9, 3])
check('explicit --groups is neutral at BATCH too (already neutral)',
      [c.group_id for c in router.rank([_g9_better, _g3_worse], tier='BATCH',
                                       groups_were_explicit=True)],
      [c.group_id for c in router.rank([_g9_better, _g3_worse],
                                       tier='BATCH')])

print()
print('--- an unreadable floor is not a zero floor ---')
_unreadable = capacity.CapacityResult(
    ok=True, cells_ok=(), cells_insufficient=(), total_pool_capacity=0,
    total_obtainable=0, alloc_scoped_quota=0, alloc_scoped_used=0,
    quota_readable=False, pool='deepmind-dynamic-pool')
_no_floor = capacity.CapacityResult(
    ok=True, cells_ok=(), cells_insufficient=(), total_pool_capacity=0,
    total_obtainable=0, alloc_scoped_quota=0, alloc_scoped_used=0,
    quota_readable=True, pool='deepmind-dynamic-pool')
check('a read FAILURE is distinguishable from a real zero',
      (_unreadable.quota_readable, _no_floor.quota_readable), (False, True))
check('both still carry quota == 0, so ranking is unchanged',
      (_unreadable.alloc_scoped_quota, _no_floor.alloc_scoped_quota), (0, 0))
check('quota_readable defaults to True, so existing callers are unaffected',
      capacity.CapacityResult(
          ok=True, cells_ok=(), cells_insufficient=(), total_pool_capacity=0,
          total_obtainable=0).quota_readable,
      True)

print()
if fails:
  print(f'FAILED {len(fails)}:')
  for f in fails:
    print('  ', f)
  sys.exit(1)
print(f'ALL {n_checks} ROUTER TESTS PASSED')
