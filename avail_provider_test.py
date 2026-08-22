"""Unit tests for the availability provider. Fakes stand in for the RPC + proto,
so the pure helpers and the fetch glue are covered without google3 at runtime."""

import json
import os
import tempfile
import unittest

from google3.experimental.users.qiaos.tpu_utils import avail_provider as AP
from google3.experimental.users.qiaos.tpu_utils import route_lib as R


# --- duck-typed proto fakes ------------------------------------------------
class _PC:
  def __init__(self, platform, num_chips):
    self.platform = platform
    self.num_chips = num_chips


class _Entry:
  def __init__(self, cell, pcs):
    self.cell = cell
    self.platform_to_chip_counts = pcs


class _Oversold:
  def __init__(self, cell, platforms):
    self.cell = cell
    self.platforms = platforms


class _AA:
  def __init__(self, obtainable):
    self.obtainable_capacity = list(obtainable)


class _Avail:
  def __init__(self, max_available_chips=(), oversold_statuses=(),
               allotment_availability=()):
    self.max_available_chips = list(max_available_chips)
    self.oversold_statuses = list(oversold_statuses)
    self.allotment_availability = list(allotment_availability)


class _Tiered:
  def __init__(self, availability):
    self.availability = availability


class _Resp:
  def __init__(self, tiers):
    self.tiered_dynamic_pool_availabilities = list(tiers)


def _resp_one_tier(free_map, oversold_cells=(), platform=101):
  """free_map: {cell -> free_chips} for a single platform, single tier."""
  entries = [_Entry(c, [_PC(platform, n)]) for c, n in free_map.items()]
  over = [_Oversold(c, [platform]) for c in oversold_cells]
  return _Resp([_Tiered(_Avail(entries, over))])


class MetroTest(unittest.TestCase):

  def test_yu_prefix(self):
    self.assertEqual(AP.metro_of('yutulpz'), 'tul')
    self.assertEqual(AP.metro_of('yulpptr'), 'lpp')
    self.assertEqual(AP.metro_of('yucbfiv'), 'cbf')
    self.assertEqual(AP.metro_of('yudfwra'), 'dfw')
    self.assertEqual(AP.metro_of('yuskedq'), 'ske')

  def test_overrides(self):
    self.assertEqual(AP.metro_of('dl'), 'las')
    self.assertEqual(AP.metro_of('sk'), 'sin')
    self.assertEqual(AP.metro_of('je'), 'cbf')

  def test_unknown_falls_back_to_self(self):
    self.assertEqual(AP.metro_of('zz'), 'zz')


class LoadPricesTest(unittest.TestCase):

  def _write(self, prices):
    fd, path = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    with open(path, 'w') as f:
      json.dump({'prices': prices}, f)
    self.addCleanup(os.remove, path)
    return path

  def test_reads_global_prod_per_arch(self):
    path = self._write({
        'deepmind-dynamic-pool|101|PROD': {'global': 20.0, 'yutulpz': 20.0},
        'deepmind-dynamic-pool|92|PROD': {'global': 9.15},
        'deepmind-dynamic-pool|34|PROD': {'global': 0.41},
    })
    p = AP.load_prices(path)
    self.assertEqual(p['v7'], 20.0)
    self.assertEqual(p['v6p'], 9.15)
    self.assertEqual(p['v4'], 0.41)
    self.assertNotIn('v5p', p)   # no key present

  def test_v6e_falls_back_to_second_card(self):
    path = self._write({'deepmind-dynamic-pool|63|PROD': {'global': 15.5}})
    p = AP.load_prices(path)
    self.assertEqual(p['v6e'], 15.5)   # 76 missing, 63 used

  def test_missing_file_is_empty(self):
    self.assertEqual(AP.load_prices('/no/such/market.json'), {})

  def test_no_v5e_ever(self):
    path = self._write({'deepmind-dynamic-pool|62|PROD': {'global': 1.26}})
    p = AP.load_prices(path)
    self.assertEqual(p, {})   # 62 is v5e, not in ARCH_CARDS


class ParseCellAvailabilityTest(unittest.TestCase):

  def test_free_and_oversold(self):
    resp = _resp_one_tier({'yutulpz': 3911, 'yulpptr': 545},
                          oversold_cells=['yulpptr'], platform=101)
    got = AP.parse_cell_availability(resp, 101)
    self.assertEqual(got['yutulpz'], (3911, False))
    self.assertEqual(got['yulpptr'], (545, True))

  def test_obtainable_is_ignored(self):
    # A response carrying only obtainable_capacity (no max_available_chips)
    # must yield NOTHING -- obtainable is the number that lies.
    av = _Avail(
        max_available_chips=[], oversold_statuses=[],
        allotment_availability=[_AA([_Entry('yutulpz', [_PC(101, 1616)])])])
    resp = _Resp([_Tiered(av)])
    got = AP.parse_cell_availability(resp, 101)
    self.assertEqual(got, {})

  def test_wrong_platform_filtered(self):
    resp = _resp_one_tier({'yutulpz': 100}, platform=92)   # v6p
    self.assertEqual(AP.parse_cell_availability(resp, 101), {})   # asking v7

  def test_sums_across_tiers(self):
    t1 = _Tiered(_Avail([_Entry('c', [_PC(101, 10)])]))
    t2 = _Tiered(_Avail([_Entry('c', [_PC(101, 5)])]))
    resp = _Resp([t1, t2])
    self.assertEqual(AP.parse_cell_availability(resp, 101)['c'], (15, False))


class BuildAvailabilityTest(unittest.TestCase):

  def test_assembles_cellavail_price_pool(self):
    per_arch = {
        'v7': {'yutulpz': (3911, False), 'yulpptr': (545, True)},
        'v6p': {'yucbfiv': (100, False)},
    }
    price = {'v7': 20.0, 'v6p': 9.15}
    avail, ap_price, pool = AP.build_availability(per_arch, price)
    self.assertEqual(avail['yutulpz|v7'].arch, 'v7')
    self.assertEqual(avail['yutulpz|v7'].free_chips, 3911)
    self.assertEqual(avail['yutulpz|v7'].price, 20.0)
    self.assertEqual(avail['yutulpz|v7'].metro, 'tul')
    self.assertTrue(avail['yulpptr|v7'].oversold)
    self.assertEqual(pool['v7'], 3911 + 545)   # sum of free chips
    self.assertEqual(pool['v6p'], 100)
    self.assertIs(ap_price, price)

  def test_dual_arch_cell_keeps_both(self):
    # A cell that physically hosts two generations (je: v6e + v7) must keep
    # BOTH -- keying by bare cell name would drop one and hide it from routing.
    per_arch = {
        'v6e': {'je': (10, False)},
        'v7': {'je': (99, False)},
    }
    avail, _, pool = AP.build_availability(per_arch, {})
    self.assertEqual(avail['je|v6e'].free_chips, 10)
    self.assertEqual(avail['je|v7'].free_chips, 99)
    self.assertEqual(pool['v6e'], 10)
    self.assertEqual(pool['v7'], 99)


class FetchTest(unittest.TestCase):
  """End-to-end fetch with injected fakes: no google3 stub, no real alloc."""

  def _provider(self, resp_by_arch, price_path):
    class _Details:
      resource_pool_name = 'deepmind-dynamic-pool'
      xborg_allotment_name = 'allot'

    calls = []

    class _Stub:
      def GetCellAvailability(self, req):  # noqa: N802 (proto-style name)
        # platform is the single appended element
        plat = int(req.platforms[0])
        calls.append(plat)
        return resp_by_arch[plat]

    # Patch the request proto used inside fetch with a tiny local fake.
    class _Req:
      def __init__(self):
        self.platforms = []
        self.allowed_allotments = []
        self.resource_pool = ''

    prov = AP.AvailabilityProvider(
        archs=['v7', 'v6p'],
        market_json_path=price_path,
        stub_factory=lambda: _Stub(),
        alloc_resolver=lambda g: _Details(),
        platform_enum=lambda arch: {'v7': 101, 'v6p': 92}[arch],
        request_factory=_Req,
    )
    return prov, calls

  def test_fetch_end_to_end(self):
    price_path = tempfile.mkstemp(suffix='.json')[1]
    with open(price_path, 'w') as f:
      json.dump({'prices': {
          'deepmind-dynamic-pool|101|PROD': {'global': 20.0},
          'deepmind-dynamic-pool|92|PROD': {'global': 9.15},
      }}, f)
    self.addCleanup(os.remove, price_path)

    resp_by_arch = {
        101: _resp_one_tier({'yutulpz': 3911, 'yulpptr': 545},
                            oversold_cells=['yulpptr'], platform=101),
        92: _resp_one_tier({'yucbfiv': 200}, platform=92),
    }
    prov, calls = self._provider(resp_by_arch, price_path)
    avail, price, pool = prov.fetch()
    self.assertEqual(avail['yutulpz|v7'].arch, 'v7')
    self.assertEqual(avail['yutulpz|v7'].free_chips, 3911)
    self.assertTrue(avail['yulpptr|v7'].oversold)
    self.assertEqual(avail['yucbfiv|v6p'].arch, 'v6p')
    self.assertEqual(price['v7'], 20.0)
    self.assertEqual(pool['v7'], 3911 + 545)
    self.assertEqual(pool['v6p'], 200)
    self.assertEqual(sorted(calls), [92, 101])


if __name__ == '__main__':
  unittest.main()
