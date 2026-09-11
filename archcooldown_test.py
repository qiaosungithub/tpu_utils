# Copyright 2026 Google LLC. All Rights Reserved.
"""Arch-level re-route cooldown: penalise the whole GENERATION, not just a cell.

The per-cell cooldown (`cooldown_cells`, tested in cooldownweight_test.py) could
not move a job off a cheap-but-unusable arch: v4 has many cells (nm, nf, oe...),
so cooling one just sent the router to the next v4 cell, and the v4 generation
was never penalised. Operator 2026-09-10 asked for an arch-level cooldown that,
after a revision, is an ADDITIVE surcharge on the SORT price:

  1. keyed by ARCH, so the job is de-preferred up the type ladder, not hopped
     to the next cell of the same arch;
  2. ADDITIVE (`strikes * pct * cap`), a bounded nudge -- the first cut was a
     2x..7x multiplier the operator found too harsh;
  3. keyed to the arch's LIMIT-ORDER CAP, a stable non-zero constant, so it does
     NOT collapse to zero when the pool cleared free (price 0) that cycle -- the
     free-but-stuck arch that most needs pushing off still gets penalised;
  4. STACKS across re-routes and is FLAT across the window then off at `until`
     (a decaying window expired before the backlogged build-worker re-dispatched);
  5. CAPPED at ARCH_COOLDOWN_MAX_STRIKES strikes;
  6. SORT-ONLY: added to the ranking key, NEVER to the real price, so the
     limit-order gate (which caps the REAL price only) is untouched.

The NEGATIVE CONTROLS carry the weight. Two are new for the revision and pin the
operator's explicit constraints: the price=0 case (a multiplier would have made
the penalty vanish) and the sort-only case (the surcharge must not act as a
price gate). The rest pin: never-re-routed is untouched, off at the window edge,
strikes reset after a quiet window, and the cap holds.
"""

import os

from google3.experimental.users.qiaos.tpu_utils import route_lib as R
from google3.testing.pybase import googletest

_NOW = 1_000_000.0
_CD = 7200.0  # the 2h window (operator 2026-09-10)
_PCT = R.ARCH_COOLDOWN_STRIKE_PCT_DEFAULT   # 0.15
# Limit-order caps the surcharge is keyed to (cap_policy.CAP_POLICY).
_CAP = {'v4': 5.0, 'v5p': 5.0, 'v6p': 20.0, 'v7': 20.0}


def _entry(archs=('v4', 'v5p', 'v6p', 'v7'), power='v4-128', cooldown_archs=None):
  e = R.QueueEntry(job_id='j', power=power, allowed_archs=list(archs))
  if cooldown_archs is not None:
    e.cooldown_archs = dict(cooldown_archs)
  return e


class ArchCooldownSurchargeTest(googletest.TestCase):
  """The pure surcharge: additive, cap-keyed, stacking, capped, off-at-expiry."""

  def test_no_record_is_zero(self):
    self.assertEqual(R.arch_cooldown_surcharge(0, _NOW + _CD, _NOW, 5.0), 0.0)
    self.assertEqual(R.arch_cooldown_surcharge(3, None, _NOW, 5.0), 0.0)
    self.assertEqual(R.arch_cooldown_surcharge(3, 0, _NOW, 5.0), 0.0)

  def test_no_cap_policy_is_zero(self):
    # An arch with no cap policy has no surcharge UNIT, so it is un-penalised
    # rather than crashing or penalised by a guessed amount.
    self.assertEqual(R.arch_cooldown_surcharge(3, _NOW + _CD, _NOW, None), 0.0)
    self.assertEqual(R.arch_cooldown_surcharge(3, _NOW + _CD, _NOW, 0.0), 0.0)

  def test_surcharge_stacks_additively_with_strikes(self):
    # strikes * pct * cap, cap=5 -> 0.75 per strike at the 15% default.
    self.assertAlmostEqual(
        R.arch_cooldown_surcharge(1, _NOW + _CD, _NOW, 5.0), 1 * _PCT * 5.0)
    self.assertAlmostEqual(
        R.arch_cooldown_surcharge(2, _NOW + _CD, _NOW, 5.0), 2 * _PCT * 5.0)
    self.assertAlmostEqual(
        R.arch_cooldown_surcharge(3, _NOW + _CD, _NOW, 5.0), 3 * _PCT * 5.0)

  def test_surcharge_is_capped_at_max_strikes(self):
    at_cap = R.arch_cooldown_surcharge(
        R.ARCH_COOLDOWN_MAX_STRIKES, _NOW + _CD, _NOW, 5.0)
    self.assertAlmostEqual(at_cap, R.ARCH_COOLDOWN_MAX_STRIKES * _PCT * 5.0)
    # More strikes add nothing past the cap.
    self.assertEqual(R.arch_cooldown_surcharge(99, _NOW + _CD, _NOW, 5.0), at_cap)

  def test_surcharge_is_FLAT_not_decaying(self):
    fresh = R.arch_cooldown_surcharge(2, _NOW + _CD, _NOW, 5.0)
    almost_expired = R.arch_cooldown_surcharge(2, _NOW + 1, _NOW, 5.0)
    self.assertEqual(fresh, almost_expired)

  def test_NEGCTL_surcharge_is_OFF_at_and_after_expiry(self):
    self.assertEqual(R.arch_cooldown_surcharge(4, _NOW, _NOW, 5.0), 0.0)
    self.assertEqual(R.arch_cooldown_surcharge(4, _NOW - 1, _NOW, 5.0), 0.0)

  def test_bigger_cap_means_bigger_surcharge(self):
    # The surcharge scales with the arch's own cap: a v6p strike (cap 20) is
    # worth 4x a v4 strike (cap 5) at the same percentage.
    v4 = R.arch_cooldown_surcharge(1, _NOW + _CD, _NOW, 5.0)
    v6p = R.arch_cooldown_surcharge(1, _NOW + _CD, _NOW, 20.0)
    self.assertAlmostEqual(v6p, 4.0 * v4)


class ArchCooldownPctEnvTest(googletest.TestCase):
  """The percentage is env-tunable so it can be retuned by a worker restart."""

  def setUp(self):
    super().setUp()
    self._saved = os.environ.get(R.ARCH_COOLDOWN_PCT_ENV)
    os.environ.pop(R.ARCH_COOLDOWN_PCT_ENV, None)

  def tearDown(self):
    if self._saved is None:
      os.environ.pop(R.ARCH_COOLDOWN_PCT_ENV, None)
    else:
      os.environ[R.ARCH_COOLDOWN_PCT_ENV] = self._saved
    super().tearDown()

  def test_default_when_unset(self):
    self.assertEqual(R._arch_cooldown_pct(), R.ARCH_COOLDOWN_STRIKE_PCT_DEFAULT)

  def test_env_overrides(self):
    os.environ[R.ARCH_COOLDOWN_PCT_ENV] = '0.5'
    self.assertEqual(R._arch_cooldown_pct(), 0.5)
    # And it flows into the surcharge: 1 strike * 0.5 * cap 20 = 10.0.
    self.assertAlmostEqual(
        R.arch_cooldown_surcharge(1, _NOW + _CD, _NOW, 20.0), 10.0)

  def test_NEGCTL_garbage_and_negative_fall_back_to_default(self):
    for bad in ('', '   ', 'abc', '-0.3'):
      os.environ[R.ARCH_COOLDOWN_PCT_ENV] = bad
      self.assertEqual(R._arch_cooldown_pct(),
                       R.ARCH_COOLDOWN_STRIKE_PCT_DEFAULT,
                       msg=f'value {bad!r} should fall back to default')


class ArchCooldownLookupTest(googletest.TestCase):
  """arch_cooldown_surcharge_for reads the record AND the cap policy."""

  def test_missing_field_is_zero(self):
    e = _entry()  # no cooldown_archs written
    self.assertEqual(R.arch_cooldown_surcharge_for(e, 'v4', _NOW), 0.0)

  def test_reads_strikes_until_and_cap(self):
    e = _entry(cooldown_archs={'v4': {'until': _NOW + _CD, 'strikes': 2}})
    # v4 cap is 5 -> 2 * 0.15 * 5 = 1.5
    self.assertAlmostEqual(R.arch_cooldown_surcharge_for(e, 'v4', _NOW),
                           2 * _PCT * 5.0)
    self.assertAlmostEqual(R.arch_cooldown_surcharge_for(e, 'V4', _NOW),
                           2 * _PCT * 5.0)                     # case-insensitive
    self.assertEqual(R.arch_cooldown_surcharge_for(e, 'v7', _NOW), 0.0)  # other

  def test_expired_record_is_zero(self):
    e = _entry(cooldown_archs={'v4': {'until': _NOW - 1, 'strikes': 4}})
    self.assertEqual(R.arch_cooldown_surcharge_for(e, 'v4', _NOW), 0.0)


class MarkRerouteArchTest(googletest.TestCase):
  """mark_reroute writes the arch cooldown, stacking within the window."""

  def _placed(self, arch='v4', cell='nm'):
    e = _entry()
    e.arch = arch
    e.cell = cell
    e.submitted_at = _NOW - 100.0
    return e

  def test_first_reroute_records_one_strike(self):
    e = self._placed()
    R.mark_reroute(e, _NOW, _CD)
    self.assertEqual(e.cooldown_archs['v4']['strikes'], 1)
    self.assertEqual(e.cooldown_archs['v4']['until'], _NOW + _CD)

  def test_strikes_STACK_when_rerouted_again_within_window(self):
    e = self._placed()
    R.mark_reroute(e, _NOW, _CD)                 # strike 1
    e.arch = 'v4'
    e.cell = 'nf'                                # a DIFFERENT v4 cell
    R.mark_reroute(e, _NOW + 1000.0, _CD)        # still inside window
    self.assertEqual(e.cooldown_archs['v4']['strikes'], 2)   # stacked
    self.assertEqual(e.cooldown_archs['v4']['until'], _NOW + 1000.0 + _CD)

  def test_NEGCTL_strikes_RESET_after_a_quiet_window(self):
    e = self._placed()
    R.mark_reroute(e, _NOW, _CD)                 # strike 1, until NOW+CD
    e.arch = 'v4'
    e.cell = 'nm'
    R.mark_reroute(e, _NOW + _CD + 1.0, _CD)     # after the window elapsed
    self.assertEqual(e.cooldown_archs['v4']['strikes'], 1)   # reset, not 2

  def test_cell_cooldown_still_written_too(self):
    e = self._placed(cell='nm')
    R.mark_reroute(e, _NOW, _CD)
    self.assertEqual(e.cooldown_cells['nm'], _NOW + _CD)

  def test_NEGCTL_no_arch_no_write(self):
    e = _entry()
    e.arch = None
    e.cell = None
    e.submitted_at = _NOW
    R.mark_reroute(e, _NOW, _CD)
    self.assertEqual(e.cooldown_archs, {})


class CandidateShapesArchCooldownTest(googletest.TestCase):
  """The payoff: a struck arch sorts BELOW an arch it was not knocked off."""

  # Realistic per-arch prices (arch-global PROD, 2026-09-10): v4 cheapest.
  _PRICE = {'v4': 4.29, 'v5p': 8.0, 'v6p': 13.94, 'v7': 35.74}
  _POOL = {'v4': 1000.0, 'v5p': 1000.0, 'v6p': 1000.0, 'v7': 1000.0}

  def _shapes(self, e, pct=None):
    saved = os.environ.get(R.ARCH_COOLDOWN_PCT_ENV)
    if pct is not None:
      os.environ[R.ARCH_COOLDOWN_PCT_ENV] = str(pct)
    else:
      os.environ.pop(R.ARCH_COOLDOWN_PCT_ENV, None)
    try:
      return R.candidate_shapes(e, arch_price=self._PRICE, arch_pool=self._POOL,
                                now=_NOW)
    finally:
      if saved is None:
        os.environ.pop(R.ARCH_COOLDOWN_PCT_ENV, None)
      else:
        os.environ[R.ARCH_COOLDOWN_PCT_ENV] = saved

  def test_NEGCTL_without_cooldown_cheapest_arch_wins(self):
    e = _entry(power='v4-128', archs=('v4', 'v5p', 'v6p', 'v7'))
    self.assertEqual(self._shapes(e)[0][0], 'v4')

  def test_default_pct_is_a_gentle_nudge_not_an_escape(self):
    # At 15%, v4 cap 5 -> +0.75/strike; capped at 4 strikes -> 4.29 + 3.0 = 7.29,
    # still below v6p 13.94. This documents the operator's observation that 15%
    # de-ranks but does not cross generations for a cheap arch.
    e = _entry(power='v4-128', archs=('v4', 'v6p', 'v7'),
               cooldown_archs={'v4': {'until': _NOW + _CD, 'strikes': 99}})
    self.assertEqual(self._shapes(e)[0][0], 'v4')   # 7.29 < 13.94

  def test_higher_pct_escapes_v4_to_v6p(self):
    # The env knob makes it as aggressive as wanted: at 50%, v4 cap 5 ->
    # +2.5/strike, 4 strikes -> 4.29 + 10.0 = 14.29 > v6p 13.94: escapes.
    e = _entry(power='v4-128', archs=('v4', 'v6p', 'v7'),
               cooldown_archs={'v4': {'until': _NOW + _CD, 'strikes': 4}})
    order = [a for a, _ in self._shapes(e, pct=0.5)]
    self.assertEqual(order[0], 'v6p')
    self.assertLess(order.index('v6p'), order.index('v4'))

  def test_NEGCTL_expired_strike_restores_v4_lead(self):
    e = _entry(power='v4-128', archs=('v4', 'v5p', 'v6p', 'v7'),
               cooldown_archs={'v4': {'until': _NOW - 1, 'strikes': 4}})
    self.assertEqual(self._shapes(e)[0][0], 'v4')


class SurchargeIsSortOnlyNotAGateTest(googletest.TestCase):
  """OPERATOR RED LINE: the surcharge de-prefers; the limit order caps the REAL
  price only. The penalised sort price must NEVER be fed to the price-cap gate.
  """

  def _avail(self, cell, arch, price, free=256, metro='sin'):
    return R.CellAvail(cell=cell, arch=arch, metro=metro, free_chips=free,
                       price=price, oversold=False)

  def test_penalised_arch_still_PLACES_when_real_price_is_under_cap(self):
    # v4 cap is 5. A cell priced 4.0 is UNDER the cap and must place, even with
    # a heavy arch cooldown whose surcharge (strikes*pct*cap) would, if it were
    # added to the real price, push 4.0 above the cap and wrongly exclude it.
    e = _entry(power='v4-128', archs=('v4',),
               cooldown_archs={'v4': {'until': _NOW + _CD, 'strikes': 99}})
    os.environ[R.ARCH_COOLDOWN_PCT_ENV] = '0.5'   # +10 surcharge at the cap
    try:
      got = R.best_cell_for_shape('v4', 128, e, {'nm': self._avail('nm', 'v4', 4.0)},
                                  now=_NOW)
    finally:
      os.environ.pop(R.ARCH_COOLDOWN_PCT_ENV, None)
    self.assertIsNotNone(got)          # NOT price-capped out by the surcharge
    self.assertEqual(got[0].cell, 'nm')

  def test_NEGCTL_real_over_cap_is_still_excluded_independently(self):
    # Sanity that the cap gate itself still works: a cell whose REAL price is
    # over the v4 cap (5) is excluded regardless of any cooldown.
    e = _entry(power='v4-128', archs=('v4',))
    got = R.best_cell_for_shape('v4', 128, e, {'nm': self._avail('nm', 'v4', 9.0)},
                                now=_NOW)
    self.assertIsNone(got)


class SurchargeSurvivesPriceZeroTest(googletest.TestCase):
  """OPERATOR: a multiplier `k*price` collapses to 0 at price 0, so the free-but-
  stuck arch that most needs pushing off got no penalty. The cap-keyed additive
  surcharge does not have this hole.
  """

  def test_price_zero_arch_is_still_penalised(self):
    # Two archs both cleared FREE (price 0.0) this cycle. v4 has 4 strikes; v6p
    # is clean. A multiplier would tie them at 0; the additive cap-keyed
    # surcharge lifts v4 above v6p so the stuck free arch is still de-preferred.
    price = {'v4': 0.0, 'v6p': 0.0}
    pool = {'v4': 1000.0, 'v6p': 1000.0}
    e = _entry(power='v4-128', archs=('v4', 'v6p'),
               cooldown_archs={'v4': {'until': _NOW + _CD, 'strikes': 4}})
    shapes = R.candidate_shapes(e, arch_price=price, arch_pool=pool, now=_NOW)
    order = [a for a, _ in shapes]
    self.assertLess(order.index('v6p'), order.index('v4'))   # v4 pushed down

  def test_NEGCTL_price_zero_no_strikes_keeps_arch_pref_order(self):
    price = {'v4': 0.0, 'v6p': 0.0}
    pool = {'v4': 1000.0, 'v6p': 1000.0}
    e = _entry(power='v4-128', archs=('v4', 'v6p'))   # no cooldown at all
    order = [a for a, _ in
             R.candidate_shapes(e, arch_price=price, arch_pool=pool, now=_NOW)]
    # Both free -> tie on price 0, ARCH_PREF breaks it: v6p (newer) leads.
    self.assertEqual(order[0], 'v6p')


class RoundTripSerializationTest(googletest.TestCase):
  """cooldown_archs must survive the JSON queue file round trip."""

  def test_cooldown_archs_survives_to_dict_from_dict(self):
    e = _entry(cooldown_archs={'v4': {'until': _NOW + _CD, 'strikes': 3}})
    e2 = R.QueueEntry.from_dict(e.to_dict())
    self.assertEqual(e2.cooldown_archs, {'v4': {'until': _NOW + _CD, 'strikes': 3}})

  def test_NEGCTL_old_row_without_field_defaults_empty(self):
    d = _entry().to_dict()
    d.pop('cooldown_archs', None)
    e = R.QueueEntry.from_dict(d)
    self.assertEqual(e.cooldown_archs, {})
    self.assertEqual(R.arch_cooldown_surcharge_for(e, 'v4', _NOW), 0.0)


if __name__ == '__main__':
  googletest.main()
