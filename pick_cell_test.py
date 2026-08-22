"""Unit tests for the cell picker. Fake provider: no RPC."""

import unittest

from google3.experimental.users.qiaos.tpu_utils import pick_cell as PC
from google3.experimental.users.qiaos.tpu_utils import route_lib as R


def _avail(cell, arch, free, oversold=False, price=20.0, metro=''):
  return R.CellAvail(cell=cell, arch=arch, free_chips=free, oversold=oversold,
                     price=price, metro=metro or cell)


class _FakeProvider:
  def __init__(self, avail_by_cell, raise_exc=False):
    self._a = avail_by_cell
    self._raise = raise_exc

  def fetch(self):
    if self._raise:
      raise RuntimeError('RPC down')
    return self._a, {}, {}


class ParseTest(unittest.TestCase):

  def test_basic(self):
    self.assertEqual(PC.parse_tpu_type('v7-32'), ('v7', 32))
    self.assertEqual(PC.parse_tpu_type('V6P-16'), ('v6p', 16))

  def test_multi_type_rejected(self):
    self.assertIsNone(PC.parse_tpu_type('v4-64,v5p-32'))

  def test_garbage(self):
    self.assertIsNone(PC.parse_tpu_type(''))
    self.assertIsNone(PC.parse_tpu_type('v7'))
    self.assertIsNone(PC.parse_tpu_type('v7-abc'))
    self.assertIsNone(PC.parse_tpu_type('v7-0'))
    self.assertIsNone(PC.parse_tpu_type('v7--8'))


class PickTest(unittest.TestCase):

  def test_picks_most_free_nonoversold(self):
    avail = {
        'yulpptr|v7': _avail('yulpptr', 'v7', 320, oversold=True),   # oversold -> skip
        'yukulwh|v7': _avail('yukulwh', 'v7', 3200),                 # most free -> win
        'yuskedq|v7': _avail('yuskedq', 'v7', 320),
    }
    self.assertEqual(PC.pick('v7-32', _FakeProvider(avail)), 'yukulwh')

  def test_none_when_all_oversold(self):
    avail = {'c|v7': _avail('c', 'v7', 320, oversold=True)}
    self.assertIsNone(PC.pick('v7-32', _FakeProvider(avail)))

  def test_none_when_no_slice_fits(self):
    avail = {'c|v7': _avail('c', 'v7', 16)}   # 16 < 32 chips
    self.assertIsNone(PC.pick('v7-32', _FakeProvider(avail)))

  def test_rpc_failure_returns_none(self):
    self.assertIsNone(PC.pick('v7-32', _FakeProvider({}, raise_exc=True)))

  def test_unparseable_type_returns_none(self):
    self.assertIsNone(PC.pick('v4-64,v5p-32', _FakeProvider({})))

  def test_metro_filter(self):
    avail = {
        'yutulpz|v7': _avail('yutulpz', 'v7', 3200, metro='tul'),
        'yucbfiv|v7': _avail('yucbfiv', 'v7', 320, metro='cbf'),
    }
    # restrict to cbf -> the smaller cbf cell wins despite tul being bigger
    self.assertEqual(PC.pick('v7-32', _FakeProvider(avail), metros=['cbf']),
                     'yucbfiv')

  def test_max_price_filter(self):
    avail = {
        'cheap|v7': _avail('cheap', 'v7', 320, price=10.0),
        'dear|v7': _avail('dear', 'v7', 3200, price=99.0),
    }
    # dear has more free but exceeds the cap -> cheap wins
    self.assertEqual(PC.pick('v7-32', _FakeProvider(avail), max_price=50.0),
                     'cheap')


if __name__ == '__main__':
  unittest.main()
