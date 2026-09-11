# Copyright 2026 Google LLC. All Rights Reserved.
"""Re-route cooldown must RANK a cell down, never remove it (operator order).

Until 2026-09-01 `best_cell_for_shape` did `continue` on any cell inside its
cooldown window. That fought the purpose of re-routing every 10 minutes: the
point is to re-pick the best cell each round, and an exclusion takes the option
away instead of ranking it. Worse, when every candidate was cooling the gate
returned NO cell at all, while a penalty still yields the least-bad one.

The NEGATIVE CONTROLS matter more than the positive here, because softening a
gate can only make the router pick MORE cells: the risk is that a just-vacated
cell instantly wins again and the job ping-pongs. So we pin that a cooling cell
still LOSES to an equal-priced rival, and that the real gates (oversold,
max_price, metro) are untouched.
"""

from google3.experimental.users.qiaos.tpu_utils import route_lib as R
from google3.testing.pybase import googletest

_NOW = 1_000_000.0
_CD = 1800.0


def _ca(cell, price, free=8, metro='sin', arch='h100', oversold=False):
  return R.CellAvail(cell=cell, arch=arch, metro=metro, free_chips=free,
                     price=price, oversold=oversold)


def _entry(cooldowns=None, metros=None, max_price=None):
  e = R.QueueEntry(job_id='j', power='h100-8', allowed_archs=['h100'])
  e.cooldown_cells = dict(cooldowns or {})
  e.allowed_metros = metros
  e.max_price = max_price
  return e


class CooldownPenaltyTest(googletest.TestCase):

  def test_no_cooldown_is_neutral(self):
    self.assertEqual(R.cooldown_penalty(None, _NOW), 1.0)
    self.assertEqual(R.cooldown_penalty(0, _NOW), 1.0)

  def test_expired_cooldown_is_neutral(self):
    self.assertEqual(R.cooldown_penalty(_NOW - 1, _NOW), 1.0)

  def test_penalty_is_strongest_right_after_and_decays(self):
    fresh = R.cooldown_penalty(_NOW + _CD, _NOW, _CD)
    half = R.cooldown_penalty(_NOW + _CD / 2, _NOW, _CD)
    nearly = R.cooldown_penalty(_NOW + 1, _NOW, _CD)
    self.assertAlmostEqual(fresh, 2.0, places=3)
    self.assertAlmostEqual(half, 1.5, places=3)
    self.assertLess(nearly, half)
    self.assertGreater(nearly, 1.0)


class BestCellCooldownTest(googletest.TestCase):

  def test_cooling_cell_is_STILL_ELIGIBLE_when_it_is_the_only_one(self):
    # The old hard gate returned None here -- the job could not be placed at all.
    e = _entry({'sh': _NOW + _CD})
    got = R.best_cell_for_shape('h100', 8, e, {'sh': _ca('sh', 10.0)}, now=_NOW)
    self.assertIsNotNone(got)
    self.assertEqual(got[0].cell, 'sh')

  def test_all_cells_cooling_still_yields_a_choice(self):
    e = _entry({'sh': _NOW + _CD, 'sm': _NOW + _CD})
    av = {'sh': _ca('sh', 20.0), 'sm': _ca('sm', 10.0)}
    got = R.best_cell_for_shape('h100', 8, e, av, now=_NOW)
    self.assertIsNotNone(got)
    self.assertEqual(got[0].cell, 'sm')   # cheaper of two equally-cooled

  # --- NEGATIVE CONTROL: a cooling cell must still LOSE to an equal rival ---
  def test_NEGCTL_cooling_cell_loses_to_equal_priced_clean_cell(self):
    e = _entry({'sh': _NOW + _CD})
    av = {'sh': _ca('sh', 10.0), 'sm': _ca('sm', 10.0)}
    got = R.best_cell_for_shape('h100', 8, e, av, now=_NOW)
    assert got is not None
    self.assertEqual(got[0].cell, 'sm')

  # --- NEGATIVE CONTROL: but a MUCH cheaper cooling cell may still win ---
  def test_deeply_cheaper_cooling_cell_can_still_win(self):
    # 2x penalty at worst, so a 3x cheaper cell must come back.
    e = _entry({'sh': _NOW + _CD})
    av = {'sh': _ca('sh', 10.0), 'sm': _ca('sm', 30.0)}
    got = R.best_cell_for_shape('h100', 8, e, av, now=_NOW)
    assert got is not None
    self.assertEqual(got[0].cell, 'sh')

  # --- NEGATIVE CONTROL: the REAL gates must be untouched ---
  def test_NEGCTL_oversold_is_still_a_hard_exclusion(self):
    e = _entry()
    av = {'sh': _ca('sh', 1.0, oversold=True)}
    self.assertIsNone(R.best_cell_for_shape('h100', 8, e, av, now=_NOW))

  def test_NEGCTL_max_price_is_still_a_hard_exclusion(self):
    e = _entry(max_price=5.0)
    av = {'sh': _ca('sh', 10.0)}
    self.assertIsNone(R.best_cell_for_shape('h100', 8, e, av, now=_NOW))

  def test_NEGCTL_metro_is_still_a_hard_exclusion(self):
    e = _entry(metros=['tul'])
    av = {'sh': _ca('sh', 1.0, metro='sin')}
    self.assertIsNone(R.best_cell_for_shape('h100', 8, e, av, now=_NOW))

  # --- NEGATIVE CONTROL: no cooldown anywhere = pre-existing behaviour ---
  def test_NEGCTL_without_cooldowns_cheapest_still_wins(self):
    e = _entry()
    av = {'sh': _ca('sh', 30.0), 'sm': _ca('sm', 10.0)}
    got = R.best_cell_for_shape('h100', 8, e, av, now=_NOW)
    assert got is not None
    self.assertEqual(got[0].cell, 'sm')


if __name__ == '__main__':
  googletest.main()
