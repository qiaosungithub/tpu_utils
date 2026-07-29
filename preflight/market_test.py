"""Offline unit tests for preflight.market: sentinels, blocking, degradation.

Runs without a daemon, without Spanner and without any RPC -- everything here
is pure data handling, which is exactly the part that must not regress: a
mis-decoded sentinel or a silently-swallowed missing cache would make the
router confidently recommend a blocked combination.

Run: blaze run experimental/users/qiaos/tpu_utils/preflight:market_test
"""

import json
import os
import sys
import tempfile
import time
from typing import Any, Optional

from google3.experimental.users.qiaos.tpu_utils.preflight import market

fails: list[str] = []


def check(name: str, got: Any, want: Any) -> None:
  global n_checks
  n_checks += 1
  if got != want:
    fails.append(f'{name}: got {got!r} want {want!r}')
  else:
    print(f'  ok  {name}')


n_checks = 0

print('--- decode_price sentinels ---')
check('INT64_MAX -> None (unobtainable)', market.decode_price(9223372036854775807), None)
check('INT64_MIN -> 0.0 (no bid)', market.decode_price(-9223372036854775808), 0.0)
check('negative -> 0.0', market.decode_price(-5), 0.0)
check('0 -> 0.0 (real free price)', market.decode_price(0), 0.0)
check('14235 -> 14.235', market.decode_price(14235), 14.235)

print('--- is_blocked ---')
check('no cap -> never blocked', market.is_blocked(999.0, None), False)
check('unobtainable is not a cap problem', market.is_blocked(None, 1.0), False)
check('above cap -> blocked', market.is_blocked(20.0, 14.23), True)
check('exactly at cap -> NOT blocked (strict >)', market.is_blocked(14.23, 14.23), False)
check('below cap -> not blocked', market.is_blocked(1.0, 14.23), False)

print('--- arch -> resource type (no hardcoded ints) ---')
check('v6e -> 76 GHOSTLITE_POD (not 63)', market.arch_resource_type('v6e'), 76)
check('v5p -> 59', market.arch_resource_type('v5p'), 59)
check('v4  -> 34', market.arch_resource_type('v4'), 34)
check('v6p -> 92', market.arch_resource_type('v6p'), 92)
check('v5e -> 62', market.arch_resource_type('v5e'), 62)
check('bogus -> None', market.arch_resource_type('v99z'), None)

print('--- graceful degradation ---')
snap = market.load_snapshot('/nonexistent/market.json')
check('missing file -> not available', snap.available, False)
check('missing file -> warns', 'tpu-daemon' in snap.warning, True)
check('missing file -> age None', snap.age_seconds, None)
check('missing file -> is_stale', snap.is_stale(), True)
check('missing file -> empty prices', snap.get_prices(76, 'PROD'), {})
check('missing file -> no limit order', snap.get_limit_order('mdb', 76, 'PROD'), None)
check('missing file -> age_str', snap.age_str(), 'n/a')

with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
    f.write('{not json')
    bad = f.name
check('corrupt file -> not available', market.load_snapshot(bad).available, False)
check('corrupt file -> warns', 'unreadable' in market.load_snapshot(bad).warning, True)

with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
    json.dump({'schema_version': 999, 'prices': {}}, f)
    oldver = f.name
check('wrong schema -> not available', market.load_snapshot(oldver).available, False)
check('wrong schema -> warns', 'schema_version' in market.load_snapshot(oldver).warning, True)

print('--- round-trip build/write/load preserves 3-way price state ---')
prices_by_key: dict[tuple[int, str],
                    list[tuple[str, Optional[float], str]]] = {
    (76, 'PROD'): [('ea', 22.23, 'p1'), ('nk', 48.93, 'p1'),
                   ('dead', None, 'p1'), ('global', 22.23, 'p1'),
                   ('ea', 99.0, 'p2')],
    (34, 'BATCH'): [('gg', 0.0, 'p1')],
    (34, 'SPOT'): [('gg', 1.0, 'p1')],      # unknown tier: must be dropped
}
los = {('mymdb', 76, 'PROD'): (14235, 'someone'),
       ('othermdb', 76, 'PROD'): (1, 'stranger'),
       ('mymdb', 34, 'BATCH'): (500, 'someone')}
lo_pools = {('mymdb', 76, 'PROD'): 'p1', ('mymdb', 34, 'BATCH'): 'p1',
            ('othermdb', 76, 'PROD'): 'p1'}
payload = market.build_payload(prices_by_key, los, pools={'p1', 'p2'},
                               mdbs={'mymdb'}, limit_order_pools=lo_pools,
                               generated_unix=time.time())
path = os.path.join(tempfile.mkdtemp(), 'market.json')
market.write_snapshot(payload, path)
s = market.load_snapshot(path)
check('available', s.available, True)
check('SPOT tier dropped', ('p1', 34, 'SPOT') in s.prices, False)
check('other mdb filtered out', s.get_limit_order('othermdb', 76, 'PROD'), None)
p1 = s.get_prices(76, 'PROD', pool='p1')
check('global excluded from per-cell map', 'global' in p1, False)
check('unobtainable preserved as None', p1['dead'], None)
check('ea in p1 = 22.23 (not p2 99.0)', p1['ea'], 22.23)
check('pool p2 isolated', s.get_prices(76, 'PROD', pool='p2')['ea'], 99.0)
check('pool price = global row', s.get_pool_price(76, 'PROD', pool='p1'), 22.23)
lo = s.get_limit_order('group:pool/mymdb', 76, 'PROD', pool='p1')
assert lo is not None, 'limit order for mymdb/76/PROD went missing'
check('alloc string resolves to mdb', lo.cap, 14.235)
check('limit order user', lo.user, 'someone')
check('blocks pool price', market.is_blocked(22.23, lo.cap), True)
batch_lo = s.get_limit_order('mymdb', 34, 'BATCH')
assert batch_lo is not None, 'BATCH limit order was dropped'
check('BATCH limit order kept', batch_lo.cap, 0.5)
check('fresh snapshot not stale', s.is_stale(), False)

print('--- pool price falls back to MAX per-cell when no global row ---')
noglobal = market.build_payload({(59, 'PROD'): [('a', 5.0, 'p1'), ('b', 30.0, 'p1')]},
                                {}, generated_unix=time.time())
p2 = os.path.join(tempfile.mkdtemp(), 'm.json')
market.write_snapshot(noglobal, p2)
check('no global row -> max, not min', market.load_snapshot(p2).get_pool_price(59, 'PROD', pool='p1'), 30.0)

print()
if fails:
  print(f'FAILED {len(fails)}:')
  for f in fails:
    print('  ', f)
  sys.exit(1)
print(f'ALL {n_checks} MARKET TESTS PASSED')
