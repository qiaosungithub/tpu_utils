"""Unit tests for the availability provider. Fakes stand in for the RPC + proto,
so the pure helpers and the fetch glue are covered without google3 at runtime."""

import json
import os
import sys
import tempfile
import importlib
import types
import typing
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

  def test_unknown_cell_fails_CLOSED_and_never_guesses(self):
    # Was `assertEqual(metro_of('zz'), 'zz')` -- the old guess-your-own-name
    # fallback, which made `--metro` silently drop valid cells (an unknown cell
    # read as "no capacity"). metro_of now yields the UNKNOWN sentinel, and
    # callers tell the two cases apart with `is UNKNOWN`.
    self.assertIs(AP.metro_of('zz'), AP.metro_util.UNKNOWN)
    self.assertNotEqual(AP.metro_of('zz'), 'zz')


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


# --- half-initialised module recovery --------------------------------------
# The bug: one transient gRPC failure inside a lazy import leaves a populated-
# but-empty module in sys.modules, and Python never re-imports it, so a
# long-lived worker is pinned forever while LOOKING like it is queueing.
# Negative controls matter more than the positive case here: evicting on the
# WRONG exception would mask a genuine missing dependency.

def _install_real_module(testcase, name, body='VALUE = 1\n'):
  """Write a REAL module to a temp dir on sys.path and import it.

  importlib.reload() re-runs find_spec, so a module that exists only as an
  in-memory object cannot be reloaded -- the fix must be exercised against a
  module that is genuinely importable, as the RPC stack is.
  """
  d = tempfile.mkdtemp()
  with open(os.path.join(d, name + '.py'), 'w') as f:
    f.write(body)
  sys.path.insert(0, d)
  testcase.addCleanup(lambda: sys.path.remove(d) if d in sys.path else None)
  testcase.addCleanup(sys.modules.pop, name, None)
  return importlib.import_module(name)


_POISON_MSG = (
    "module 'google3.net.rpc.python.contrib.base_stubby_api' has no "
    "attribute 'BaseStubbyApi'")


class HalfInitDiscriminatorTest(unittest.TestCase):
  """`_looks_half_initialised`: AttributeError yes, everything else no."""

  def test_attribute_error_on_module_is_half_initialised(self):
    self.assertTrue(AP._looks_half_initialised(AttributeError(_POISON_MSG)))

  def test_import_error_is_NOT_half_initialised(self):
    # ImportError means genuinely absent. Evicting + retrying cannot help and
    # would hide the real error behind a second RPC deadline.
    self.assertFalse(
        AP._looks_half_initialised(ImportError('gRPC is not installed')))
    self.assertFalse(
        AP._looks_half_initialised(ModuleNotFoundError('No module named x')))

  def test_unrelated_attribute_error_is_NOT_half_initialised(self):
    # A plain attribute typo on an object must not trigger eviction.
    self.assertFalse(
        AP._looks_half_initialised(
            AttributeError("'NoneType' object has no attribute 'foo'")))

  def test_rpc_error_is_NOT_half_initialised(self):
    # An ordinary RPC failure (deadline, unavailable) is the COMMON case and
    # must fall through to the caller untouched.
    self.assertFalse(AP._looks_half_initialised(RuntimeError('deadline exceeded')))

  def test_poisoned_module_name_is_parsed(self):
    self.assertEqual(
        AP._poisoned_module_name(AttributeError(_POISON_MSG)),
        'google3.net.rpc.python.contrib.base_stubby_api')
    self.assertIsNone(AP._poisoned_module_name(AttributeError('nope')))


class ReloadScopeTest(unittest.TestCase):
  """What `_reload_candidates` takes, and -- more important -- what it spares."""

  def setUp(self):
    super().setUp()
    self._saved = dict(sys.modules)
    self.addCleanup(self._restore)

  def _restore(self):
    sys.modules.clear()
    sys.modules.update(self._saved)

  def test_seed_module_and_submodules_are_reloaded(self):
    seed = 'zzz_fake_pkg.stub_layer'
    sys.modules[seed] = types.ModuleType(seed)
    sys.modules[seed + '.inner'] = types.ModuleType(seed + '.inner')
    got = AP._reload_candidates(seed)
    self.assertIn(seed, got)
    self.assertIn(seed + '.inner', got)

  def test_generated_protos_are_NEVER_reloaded(self):
    # Re-executing a _pb2 duplicates descriptor-pool entries and RAISES --
    # that would turn a recoverable stall into a hard crash.
    name = 'zzz_fake_grpc_service_pb2'
    sys.modules[name] = types.ModuleType(name)
    self.assertNotIn(name, AP._reload_candidates(None))

  def test_pb_stubby_siblings_ARE_reloaded(self):
    # The poison actually sits in the generated *_pb_stubby module, whose body
    # dies at `_client_stub_base_class = ...`. Sparing it would defeat the fix.
    name = 'zzz_fake_service_pb_stubby'
    sys.modules[name] = types.ModuleType(name)
    self.assertIn(name, AP._reload_candidates(None))

  def test_unrelated_modules_are_spared(self):
    # Scoped eviction, not sys.modules.clear(): the worker keeps running.
    name = 'zzz_fake_numpy_lookalike'
    sys.modules[name] = types.ModuleType(name)
    self.assertNotIn(name, AP._reload_candidates(None))

  def test_none_tombstones_are_evicted(self):
    # A failed import really does leave `None` in sys.modules at runtime; the
    # cast is only to satisfy the type checker, which types the dict as
    # dict[str, ModuleType].
    typing.cast(dict, sys.modules)['zzz_fake_tombstone'] = None
    self.assertIn('zzz_fake_tombstone', AP._tombstone_names())
    AP._heal_half_initialised_modules(None)
    self.assertNotIn('zzz_fake_tombstone', sys.modules)

  def test_heal_reloads_IN_PLACE_keeping_the_same_module_object(self):
    # ★The whole point: pop+reimport builds a NEW dict, but the cached
    # generated service class reads the OLD one via __globals__, so it would
    # keep raising. Reload re-executes into the SAME object.
    seed = 'zzz_stubby_inplace_probe'
    mod = _install_real_module(self, seed)
    AP._heal_half_initialised_modules(seed)
    self.assertIs(sys.modules.get(seed), mod,
                  'module object must survive healing (reload, not evict)')


class FetchRetryTest(unittest.TestCase):
  """fetch() retries ONCE on the poison shape, and not at all otherwise."""

  def _provider_raising(self, errors):
    """Provider whose stub_factory raises `errors` in order, then succeeds."""
    calls = {'n': 0}

    class _Details:
      resource_pool_name = 'deepmind-dynamic-pool'
      xborg_allotment_name = 'allot'

    class _Req:
      def __init__(self):
        self.platforms = []
        self.allowed_allotments = []
        self.resource_pool = ''

    def factory():
      i = calls['n']
      calls['n'] += 1
      if i < len(errors):
        raise errors[i]
      class _Stub:
        def GetCellAvailability(self, req):
          del req
          return _Resp([])  # no tiers -> empty availability
      return _Stub()

    prov = AP.AvailabilityProvider(
        archs=['v7'],
        market_json_path='/nonexistent-market.json',
        stub_factory=factory,
        alloc_resolver=lambda g: _Details(),
        platform_enum=lambda arch: 101,
        request_factory=_Req,
    )
    return prov, calls

  def test_poisoned_fetch_retries_once_and_succeeds(self):
    # THE FIX: first call poisoned, healing happens, second call succeeds.
    _install_real_module(self, 'zzz_poison_stubby_probe')
    prov, calls = self._provider_raising([AttributeError(_POISON_MSG)])
    avail, _, _ = prov.fetch()
    self.assertEqual(calls['n'], 2, 'expected exactly one retry')
    self.assertEqual(avail, {})

  def test_retry_happens_AT_MOST_once(self):
    # If the second attempt fails too, the error propagates -- no retry storm,
    # and the caller still sees a real exception rather than a silent stall.
    _install_real_module(self, 'zzz_poison_stubby_probe2')
    prov, calls = self._provider_raising(
        [AttributeError(_POISON_MSG), AttributeError(_POISON_MSG)])
    with self.assertRaises(AttributeError):
      prov.fetch()
    self.assertEqual(calls['n'], 2, 'must not retry more than once')

  def test_nameerror_from_dead_module_body_IS_retried(self):
    # The shape actually observed live: the stubby module body died before
    # binding `_client_stub_base_class`, so every later call is a NameError.
    _install_real_module(self, 'zzz_poison_stubby_probe3')
    prov, calls = self._provider_raising(
        [NameError("name '_client_stub_base_class' is not defined")])
    prov.fetch()
    self.assertEqual(calls['n'], 2, 'NameError poison must trigger one retry')

  def test_unrelated_nameerror_is_NOT_retried(self):
    prov, calls = self._provider_raising([NameError('something else entirely')])
    with self.assertRaises(NameError):
      prov.fetch()
    self.assertEqual(calls['n'], 1)

  def test_import_error_is_NOT_retried(self):
    # Negative control: a real missing dependency must surface immediately.
    prov, calls = self._provider_raising([ImportError('gRPC is not installed')])
    with self.assertRaises(ImportError):
      prov.fetch()
    self.assertEqual(calls['n'], 1, 'ImportError must not trigger a retry')

  def test_ordinary_rpc_error_is_NOT_retried(self):
    prov, calls = self._provider_raising([RuntimeError('deadline exceeded')])
    with self.assertRaises(RuntimeError):
      prov.fetch()
    self.assertEqual(calls['n'], 1)

  def test_healthy_fetch_does_not_touch_sys_modules(self):
    # The guard must be inert on the happy path.
    before = set(sys.modules)
    prov, calls = self._provider_raising([])
    prov.fetch()
    self.assertEqual(calls['n'], 1)
    self.assertEqual(before - set(sys.modules), set())


if __name__ == '__main__':
  unittest.main()
