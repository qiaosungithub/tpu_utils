"""Cell selection ranks by effective PRICE, not by roominess alone.

WHY THIS EXISTS. `best_cell_for_shape` used to sort `(-n_slices, price,
-free_chips)`: roominess first, price consulted only on an exact tie. That was
harmless while every cell of an arch reported the SAME price -- the provider
read the market layer's `global` field and copied it to every cell, so the
price key could never fire. Per-cell prices differ by up to 3.2x in real data
(v6e at 15.999 and 51.923 in one snapshot), so making the provider carry them
WITHOUT changing this sort would have actively made placement worse: the router
would have paid 3.2x to fit one more slice. Data and ranking ship together.

Every test below has a negative control asserting the opposite outcome under
the opposite input, so a test that cannot fail is visible as such.
"""

from google3.experimental.users.qiaos.tpu_utils import route_lib as R
from google3.testing.pybase import googletest as absltest


def _entry(**kw):
  e = R.QueueEntry(job_id='j', power='v6e-16', allowed_archs=['v6e'], **kw)
  e.state = R.JobState.QUEUED
  return e


def _avail(specs, arch='v6e'):
  """specs = [(cell, free_chips, price)] -> the provider's avail_by_cell map."""
  return {f'{c}|{arch}': R.CellAvail(cell=c, arch=arch, free_chips=f,
                                     oversold=False, price=p, metro='xx')
          for c, f, p in specs}


class CellScoreTest(absltest.TestCase):

  # --- 1. price beats roominess when the gap is large ------------------------

  def test_cheap_and_tight_beats_dear_and_roomy(self):
    """The real v6e spread: 15.999 vs 51.923. No amount of room buys that."""
    av = _avail([('dear', 64, 51.923), ('cheap', 16, 15.999)])
    got, n = R.best_cell_for_shape('v6e', 16, _entry(), av, now=1e9)
    self.assertEqual(got.cell, 'cheap',
                     'a 3.2x price gap must not be outweighed by roominess')
    self.assertEqual(n, 1)

  def test_negctl_roomier_wins_when_prices_are_close(self):
    """NEGATIVE CONTROL for #1: the bonus must actually DO something.

    Without this, a bug that ignored n_slices entirely would still pass #1."""
    av = _avail([('roomy', 128, 17.0), ('tight', 16, 16.0)])
    got, _ = R.best_cell_for_shape('v6e', 16, _entry(), av, now=1e9)
    self.assertEqual(got.cell, 'roomy',
                     'within SLICE_BONUS, the roomier cell must win')

  def test_bonus_ceiling_is_exactly_slice_bonus(self):
    """The ceiling is the promise: roominess is worth AT MOST SLICE_BONUS.

    A cell 1.5x+ dearer must lose no matter how vast it is -- that is what
    makes the trade sayable out loud ('up to 50% more, never beyond')."""
    self.assertAlmostEqual(R.slice_weight(10 ** 6), 1.0 + R.SLICE_BONUS)
    av = _avail([('vast', 10 ** 6, 16.0 * 1.51), ('tight', 16, 16.0)])
    got, _ = R.best_cell_for_shape('v6e', 16, _entry(), av, now=1e9)
    self.assertEqual(got.cell, 'tight', 'past the ceiling, price must win')

  # --- 2. the eviction penalty ----------------------------------------------

  def test_recent_eviction_pushes_a_cell_down(self):
    av = _avail([('evicter', 64, 16.0), ('other', 64, 20.0)])
    e = _entry()
    e.evictions = {'evicter': {'strikes': 1, 'last': 1e9 - 60}}   # 1 min ago
    got, _ = R.best_cell_for_shape('v6e', 16, e, av, now=1e9)
    self.assertEqual(got.cell, 'other',
                     'a cell that just evicted this job reads as ~2x dearer')

  def test_negctl_eviction_decays_to_nothing_in_half_an_hour(self):
    """NEGATIVE CONTROL: the penalty must EXPIRE, or it becomes a permanent
    verdict on a cell -- and preemption is a fact about a moment, not a cell."""
    av = _avail([('evicter', 64, 16.0), ('other', 64, 20.0)])
    e = _entry()
    e.evictions = {'evicter': {'strikes': 1, 'last': 1e9 - R.EVICT_DECAY_S}}
    got, _ = R.best_cell_for_shape('v6e', 16, e, av, now=1e9)
    self.assertEqual(got.cell, 'evicter',
                     'after EVICT_DECAY_S the cheaper cell must win again')
    self.assertEqual(R.evict_penalty(1, R.EVICT_DECAY_S), 1.0)
    self.assertGreater(R.evict_penalty(1, 0.0), 1.9)

  def test_negctl_no_eviction_record_is_inert(self):
    """NEGATIVE CONTROL, the one that keeps this CL honest: nothing writes
    `evictions` yet, so the penalty must be exactly 1.0 everywhere and today's
    ranking must be decided by price and roominess ALONE."""
    self.assertEqual(R.evict_penalty(0, None), 1.0)
    self.assertEqual(R.evict_penalty(0, 5.0), 1.0)
    self.assertEqual(R.evict_penalty(3, None), 1.0)
    av = _avail([('a', 64, 16.0), ('b', 64, 20.0)])
    got, _ = R.best_cell_for_shape('v6e', 16, _entry(), av, now=1e9)
    self.assertEqual(got.cell, 'a')

  def test_penalty_is_soft_not_exclusion(self):
    """When EVERY candidate has struck, a choice must still be made -- the old
    hard cooldown could leave a job with nowhere to go."""
    av = _avail([('bad1', 64, 16.0), ('bad2', 64, 16.0)])
    e = _entry()
    e.evictions = {'bad1': {'strikes': 3, 'last': 1e9 - 10},
                   'bad2': {'strikes': 1, 'last': 1e9 - 1700}}
    got, _ = R.best_cell_for_shape('v6e', 16, e, av, now=1e9)
    self.assertEqual(got.cell, 'bad2', 'the least-recent offender still wins')

  # --- 3. unknown price, and the hard gates that outrank the score -----------

  def test_unknown_price_sorts_last_not_first(self):
    """A cell we cannot cost is not a bargain. Treating a missing number as 0
    is how the cheapest-looking option becomes the unpriceable one."""
    self.assertEqual(R.cell_score(None, 99), float('inf'))
    av = _avail([('nopriced', 10 ** 6, None), ('priced', 16, 40.0)])
    got, _ = R.best_cell_for_shape('v6e', 16, _entry(), av, now=1e9)
    self.assertEqual(got.cell, 'priced')

  def test_negctl_hard_gates_still_outrank_any_score(self):
    """NEGATIVE CONTROL: oversold / max_price / cooldown are PROHIBITIONS. A
    score must never rescue a cell that failed one, or the gates become
    suggestions."""
    av = _avail([('cheap', 64, 1.0), ('ok', 16, 40.0)])
    av['cheap|v6e'] = R.CellAvail(cell='cheap', arch='v6e', free_chips=64,
                                  oversold=True, price=1.0, metro='xx')
    got, _ = R.best_cell_for_shape('v6e', 16, _entry(), av, now=1e9)
    self.assertEqual(got.cell, 'ok', 'oversold must stay excluded')

    av2 = _avail([('cheap', 64, 1.0), ('ok', 16, 40.0)])
    e = _entry()
    e.cooldown_cells = {'cheap': 1e9 + 600}
    got, _ = R.best_cell_for_shape('v6e', 16, e, av2, now=1e9)
    self.assertEqual(got.cell, 'ok', 'cooldown must stay excluded')

    e2 = _entry(max_price=10.0)
    av3 = _avail([('dear', 64, 40.0), ('fine', 16, 9.0)])
    got, _ = R.best_cell_for_shape('v6e', 16, e2, av3, now=1e9)
    self.assertEqual(got.cell, 'fine', 'the limit-order cap must still bite')

  # --- 4. the arch-level score is a SEPARATE thing ---------------------------

  def test_arch_and_cell_scores_are_independent(self):
    """Two scores, two questions. The arch score uses one global price and the
    pool bonus; the cell score uses per-cell prices and the slice bonus.
    Raising one ceiling must not move the other."""
    self.assertAlmostEqual(R.pool_weight(10 ** 9), 1.0 + R.POOL_BONUS)
    self.assertAlmostEqual(R.slice_weight(10 ** 9), 1.0 + R.SLICE_BONUS)
    self.assertAlmostEqual(R.effective_price(30.0, 10 ** 9),
                           30.0 / (1.0 + R.POOL_BONUS))
    # a full-bonus pool forgives POOL_BONUS of price, and no more
    self.assertGreater(R.effective_price(1.0 + R.POOL_BONUS + 0.01, 10 ** 9),
                       R.effective_price(1.0, 0.0))


if __name__ == '__main__':
  absltest.main()
