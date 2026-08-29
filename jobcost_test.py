"""Tests for the single pricing path and its consistency assertion.

These run against the LIVE price table, so they assert relationships and failure modes rather
than absolute numbers -- a test pinned to today's credits would fail on a market tick and
teach everyone to ignore it.
"""

import unittest

try:
  from google3.experimental.users.qiaos.tpu_utils import jobcost as jc
except ImportError:  # pragma: no cover - flat-layout fallback
  import jobcost as jc  # type: ignore



class SinglePath(unittest.TestCase):
  def test_prices_a_known_family(self):
    cost, basis = jc.price_job('b200-8', 'PROD')
    self.assertGreater(cost, 0)
    self.assertIn(basis, ('market', 'cap'))

  def test_batch_is_free_at_the_gate(self):
    """BATCH bills the group but does not draw on the PROD credit bar this gate protects."""
    self.assertEqual(jc.price_job('v6p-32', 'BATCH')[0], 0.0)

  def test_a_price_is_never_zero_whatever_the_basis(self):
    """★A `0.00 (free pool)` quote is deliberately NOT parsed as a market price: a price of
    zero makes the budget gate STRUCTURALLY unable to refuse that family, so it falls back to
    the policy cap instead.

    ★This asserts the INVARIANT (never zero), not which family is currently free -- an
    earlier version pinned a100 to basis='cap' and broke the moment a100 started quoting
    0.02 Credits/hr. The market moved; the code was fine. A test that fails on a real price
    change teaches everyone to ignore it.
    """
    for arch in ('a100-8', 'h100-8', 'b200-8', 'v6p-32'):
      with self.subTest(arch=arch):
        cost, basis = jc.price_job(arch, 'PROD')
        self.assertGreater(cost, 0, f'{arch} priced at zero via {basis}')
        self.assertIn(basis, ('market', 'cap'))

  def test_scales_with_chip_count(self):
    small, _ = jc.price_job('v6p-16', 'PROD')
    big, _ = jc.price_job('v6p-32', 'PROD')
    self.assertAlmostEqual(big / small, 2.0, places=3)


class ConsistencyAssertion(unittest.TestCase):
  """The check that was missing when a healthy six-hour job was cancelled at a fake price."""

  def test_a_large_disagreement_is_raised(self):
    ours, _ = jc.price_job('b200-8', 'PROD')
    with self.assertRaises(jc.PricingDisagreement) as e:
      jc.assert_consistent('b200-8', 'PROD', ours * 70, other_name='stale-import canceller')
    self.assertIn('disagreement', str(e.exception))

  def test_agreement_is_silent(self):
    """Negative control: an assertion that fires on everything is as useless as one that
    never fires."""
    ours, _ = jc.price_job('b200-8', 'PROD')
    self.assertEqual(jc.assert_consistent('b200-8', 'PROD', ours, other_name='same'), ours)

  def test_a_market_tick_does_not_trip_it(self):
    ours, _ = jc.price_job('v6p-32', 'PROD')
    jc.assert_consistent('v6p-32', 'PROD', ours * 1.02, other_name='one tick later')

  def test_both_zero_agree(self):
    jc.assert_consistent('v6p-32', 'BATCH', 0.0, other_name='other')

  def test_the_error_names_both_numbers_and_the_ratio(self):
    """A disagreement report that does not say which two numbers disagreed cannot be acted
    on -- and 'do not act on either' is the correct instruction, since either may be stale."""
    ours, _ = jc.price_job('h100-8', 'PROD')
    try:
      jc.assert_consistent('h100-8', 'PROD', ours * 100, other_name='X')
      self.fail('expected PricingDisagreement')
    except jc.PricingDisagreement as e:
      msg = str(e)
      self.assertIn('X priced it', msg)
      self.assertIn('shared path prices it', msg)
      self.assertIn('Do NOT act on either', msg)


class BeliefReport(unittest.TestCase):
  """C11: a long-lived process must periodically state the price it BELIEVES, so a stale
  import is visible from outside instead of only in its effects."""

  def test_report_carries_price_basis_and_pid(self):
    line = jc.belief_report('v6p-32', 'PROD')
    self.assertIn('cr/hr', line)
    self.assertIn('basis=', line)
    self.assertIn('pid=', line)

  def test_unavailable_is_reported_as_such_not_as_a_number(self):
    line = jc.belief_report('not-a-real-arch-99', 'PROD')
    self.assertTrue('UNAVAILABLE' in line or 'cr/hr' in line)
