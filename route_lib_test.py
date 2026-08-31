"""Unit tests for the pure scheduling core. No I/O, no RPC, no real clock."""

import random
from typing import Optional, TypeVar
import unittest

from google3.experimental.users.qiaos.tpu_utils import route_lib as R

_T = TypeVar('_T')


def _ok(x: Optional[_T]) -> _T:
  """Assert a router result is not None and return it type-narrowed.

  plan_one/best_cell return Optional; a test that then reads `.cell` both
  asserts placement happened AND satisfies the strict type checker."""
  assert x is not None, 'expected a placement, got None'
  return x


def _avail(cell, arch, free, oversold=False, price=25.0, metro=''):
  return R.CellAvail(cell=cell, arch=arch, free_chips=free, oversold=oversold,
                     price=price, metro=metro or cell)


def _entry(job_id='j1', power='v7-32', archs=('v7',), **kw):
  return R.QueueEntry(job_id=job_id, power=power, allowed_archs=list(archs), **kw)


class PowerTest(unittest.TestCase):

  def test_parse_power(self):
    self.assertEqual(R.parse_power('v5p-32'), 32.0)
    self.assertEqual(R.parse_power('v7-32'), 4.34 * 32)
    self.assertEqual(R.parse_power('64'), 64.0)

  def test_candidate_shapes_orders_newer_first(self):
    # power v6p-32 == 138.88 v5p-eq; v7-32 == same. Both accepted, v7 first.
    e = _entry(power='v6p-32', archs=('v6p', 'v7'))
    shapes = R.candidate_shapes(e)
    self.assertEqual(shapes[0][0], 'v7')       # ARCH_PREF puts v7 ahead
    self.assertIn(('v6p', 32), shapes)

  def test_candidate_shapes_respects_allowed_archs(self):
    e = _entry(power='v7-32', archs=('v6p',))   # only v6p allowed
    shapes = R.candidate_shapes(e)
    self.assertTrue(all(a == 'v6p' for a, _ in shapes))


class PlacementTest(unittest.TestCase):

  def test_oversold_cell_is_skipped(self):
    e = _entry()
    avail = {
        'yulpptr': _avail('yulpptr', 'v7', free=5, oversold=True),   # THE bug cell
        'yukulwh': _avail('yukulwh', 'v7', free=3244, oversold=False),
    }
    p = _ok(R.plan_one(e, avail, now=0.0))
    self.assertIsNotNone(p)
    self.assertEqual(p.cell, 'yukulwh')         # NOT the oversold one

  def test_fragmentation_zero_slices_skipped(self):
    # free chips present but < one slice (32): unplaceable.
    e = _entry()
    avail = {'c1': _avail('c1', 'v7', free=31)}   # 31 < 32 -> 0 slices
    self.assertIsNone(R.plan_one(e, avail, now=0.0))

  def test_obtainable_does_not_help_only_free_counts(self):
    # We never pass obtainable in; a cell with 0 free is unplaceable regardless.
    e = _entry()
    avail = {'c1': _avail('c1', 'v7', free=0)}
    self.assertIsNone(R.plan_one(e, avail, now=0.0))

  def test_price_cap_excludes_expensive_cell(self):
    e = _entry(max_price=20.0)
    avail = {
        'pricey': _avail('pricey', 'v7', free=3200, price=25.0),   # over cap
        'cheap': _avail('cheap', 'v7', free=64, price=15.0),
    }
    p = _ok(R.plan_one(e, avail, now=0.0))
    self.assertEqual(p.cell, 'cheap')

  def test_metro_filter(self):
    e = _entry(allowed_metros=['cbf'])
    avail = {
        'yukulwh': _avail('yukulwh', 'v7', free=3244, metro='kul'),
        'yucbfiv': _avail('yucbfiv', 'v7', free=64, metro='cbf'),
    }
    p = _ok(R.plan_one(e, avail, now=0.0))
    self.assertEqual(p.cell, 'yucbfiv')

  def test_rank_prefers_more_slices_then_cheaper(self):
    e = _entry()
    avail = {
        'few': _avail('few', 'v7', free=64, price=10.0),     # 2 slices, cheap
        'many': _avail('many', 'v7', free=3200, price=25.0),  # 100 slices
    }
    p = _ok(R.plan_one(e, avail, now=0.0))
    self.assertEqual(p.cell, 'many')     # slices dominate price

  def test_tpu_type_fallback_when_first_arch_unavailable(self):
    # v7 allowed first but no v7 anywhere; v6p available -> falls back.
    e = _entry(power='v7-32', archs=('v7', 'v6p'))
    avail = {'c1': _avail('c1', 'v6p', free=320)}   # v6p-32 fits (138.88 pw)
    p = _ok(R.plan_one(e, avail, now=0.0))
    self.assertEqual(p.arch, 'v6p')
    self.assertEqual(p.chips, 32)

  def test_cooldown_cell_skipped(self):
    e = _entry()
    e.cooldown_cells = {'hot': 100.0}
    avail = {
        'hot': _avail('hot', 'v7', free=3200),
        'cool': _avail('cool', 'v7', free=64),
    }
    self.assertEqual(_ok(R.plan_one(e, avail, now=50.0)).cell, 'cool')  # hot on cooldown
    # after cooldown expires, hot (more slices) wins
    self.assertEqual(_ok(R.plan_one(e, avail, now=150.0)).cell, 'hot')


class BatchSchedulingTest(unittest.TestCase):

  def test_priority_high_first(self):
    lo = _entry('lo', priority=0)
    hi = _entry('hi', priority=10)
    avail = {'c': _avail('c', 'v7', free=32)}   # exactly ONE slice
    plcs = R.select_and_plan([lo, hi], avail, now=0.0, rng=random.Random(0))
    self.assertEqual(len(plcs), 1)
    self.assertEqual(plcs[0].job_id, 'hi')      # high priority took the one slice

  def test_equal_priority_random_but_fair(self):
    # two equal-priority jobs, one slice: which one wins should vary with seed.
    a = _entry('a', priority=5)
    b = _entry('b', priority=5)
    avail = {'c': _avail('c', 'v7', free=32)}
    winners = set()
    for seed in range(20):
      plcs = R.select_and_plan([a, b], avail, now=0.0, rng=random.Random(seed))
      winners.add(plcs[0].job_id)
    self.assertEqual(winners, {'a', 'b'})       # both win under some seed

  def test_thundering_herd_drawdown(self):
    # one cell with exactly 2 slices, three jobs -> only 2 placed, and NOT all
    # into the same cell beyond capacity.
    jobs = [_entry(f'j{i}') for i in range(3)]
    avail = {'c': _avail('c', 'v7', free=64)}    # 2 slices
    plcs = R.select_and_plan(jobs, avail, now=0.0, rng=random.Random(1))
    self.assertEqual(len(plcs), 2)               # third stays queued

  def test_drawdown_spreads_across_cells(self):
    jobs = [_entry(f'j{i}') for i in range(2)]
    avail = {
        'a': _avail('a', 'v7', free=32),   # 1 slice
        'b': _avail('b', 'v7', free=32),   # 1 slice
    }
    plcs = R.select_and_plan(jobs, avail, now=0.0, rng=random.Random(2))
    self.assertEqual(len(plcs), 2)
    self.assertEqual({p.cell for p in plcs}, {'a', 'b'})  # one each, not both on one

  def test_drawdown_composite_key_dual_arch_cell(self):
    # A cell keyed per (cell, arch) as 'cell|arch' -- the shape the live
    # provider emits for a dual-generation cell like `je` (v6e + v7). The
    # decrement must find the right entry BY CONTENT, not by assuming the key
    # is the bare cell name, or the second job sees stale free chips.
    jobs = [_entry(f'j{i}', power='v7-32', archs=('v7',)) for i in range(3)]
    avail = {
        'je|v6e': _avail('je', 'v6e', free=999),   # same cell, other gen
        'je|v7': _avail('je', 'v7', free=64),      # 2 v7 slices only
    }
    plcs = R.select_and_plan(jobs, avail, now=0.0, rng=random.Random(3))
    self.assertEqual(len(plcs), 2)                 # v7 drawn down to 0, third waits
    self.assertTrue(all(p.arch == 'v7' and p.cell == 'je' for p in plcs))


class RerouteTest(unittest.TestCase):

  def test_needs_reroute_timing(self):
    e = _entry()
    e.state = R.JobState.SUBMITTED
    e.submitted_at = 1000.0
    self.assertFalse(R.needs_reroute(e, now=1000.0 + 599, reroute_after_s=600))
    self.assertTrue(R.needs_reroute(e, now=1000.0 + 600, reroute_after_s=600))

  def test_needs_reroute_only_for_submitted(self):
    e = _entry()
    e.state = R.JobState.QUEUED
    e.submitted_at = 0.0
    self.assertFalse(R.needs_reroute(e, now=1e9, reroute_after_s=600))

  def test_mark_reroute_sets_cooldown_and_requeues(self):
    e = _entry()
    e.state = R.JobState.SUBMITTED
    e.cell = 'yulpptr'
    e.submitted_at = 0.0
    e.xid = '12345'
    R.mark_reroute(e, now=700.0, cooldown_s=1800.0)
    self.assertEqual(e.state, R.JobState.QUEUED)
    self.assertIsNone(e.xid)
    self.assertGreater(e.cooldown_cells['yulpptr'], 700.0)
    # ★A re-route is NOT a build failure: `attempts` feeds the 3-strikes brake
    # that parks a row HELD, so bumping it here parked healthy jobs that the
    # router had merely moved between oversold cells (infra-v17, measured on
    # elt's cars: attempts=3 with zero real build failures). Re-routes are
    # counted separately, and the old XID is preserved for the audit.
    self.assertEqual(e.attempts, 0)
    self.assertEqual(e.reroutes, 1)
    self.assertEqual(e.prior_xids, ['12345'])

  def test_repeated_reroutes_never_trip_the_build_brake(self):
    # Regression: an oversold-cell rotation must not park a healthy job.
    e = _entry()
    e.state = R.JobState.SUBMITTED
    e.submitted_at = 0.0
    for i in range(10):
      e.cell = f'cell{i}'
      e.xid = str(1000 + i)
      R.mark_reroute(e, now=700.0 + i, cooldown_s=1800.0)
    self.assertEqual(e.attempts, 0)
    self.assertEqual(e.reroutes, 10)
    self.assertEqual(len(e.prior_xids), 10)

  def test_reroute_then_replan_avoids_hot_cell(self):
    # end-to-end: job stuck in yulpptr, re-routed, next plan picks another cell.
    e = _entry()
    e.state = R.JobState.SUBMITTED
    e.cell = 'yulpptr'
    e.submitted_at = 0.0
    R.mark_reroute(e, now=700.0, cooldown_s=1800.0)
    avail = {
        'yulpptr': _avail('yulpptr', 'v7', free=3200),   # now looks free but on cooldown
        'yukulwh': _avail('yukulwh', 'v7', free=3200),
    }
    p = _ok(R.plan_one(e, avail, now=800.0))
    self.assertEqual(p.cell, 'yukulwh')      # avoided the cooled-down cell

  # --- hardening pure logic (2026-08-24) ---
  def test_output_is_fresh_within_window(self):
    self.assertTrue(R.output_is_fresh(latest_mtime=640.0, now=700.0,
                                      fresh_within_s=1200.0))   # 60s ago

  def test_output_is_fresh_none_means_no_evidence(self):
    self.assertFalse(R.output_is_fresh(latest_mtime=None, now=700.0,
                                       fresh_within_s=1200.0))  # missing = not alive

  def test_output_is_fresh_boundary_is_stale(self):
    # EXACTLY fresh_within_s ago counts as stale, so the window can never make
    # reroute a permanent no-op.
    self.assertFalse(R.output_is_fresh(latest_mtime=800.0, now=2000.0,
                                       fresh_within_s=1200.0))  # 1200s ago == boundary
    self.assertTrue(R.output_is_fresh(latest_mtime=801.0, now=2000.0,
                                      fresh_within_s=1200.0))   # 1199s ago < window

  def test_decide_reroute_both_pending_no_output_reroutes(self):
    self.assertTrue(R.decide_reroute('PENDING', 'PENDING', output_fresh=False))

  def test_decide_reroute_fresh_output_blocks(self):
    self.assertFalse(R.decide_reroute('PENDING', 'PENDING', output_fresh=True))

  def test_decide_reroute_second_running_blocks(self):
    self.assertFalse(R.decide_reroute('PENDING', 'RUNNING', output_fresh=False))

  def test_decide_reroute_second_unknown_blocks(self):
    # ambiguity protects: never cancel on a second UNKNOWN.
    self.assertFalse(R.decide_reroute('PENDING', 'UNKNOWN', output_fresh=False))

  def test_decide_reroute_first_not_pending_never_reroutes(self):
    self.assertFalse(R.decide_reroute('RUNNING', None, output_fresh=False))
    self.assertFalse(R.decide_reroute('TERMINAL', None, output_fresh=False))

  def test_decide_reroute_none_second_is_defensive_noop(self):
    self.assertFalse(R.decide_reroute('PENDING', None, output_fresh=False))


class SerdeTest(unittest.TestCase):

  def test_roundtrip(self):
    e = _entry(priority=3, allowed_metros=['cbf', 'tul'],
               launch_kwargs={'config': 'remote_run', 'exp_name': 'x'})
    e.state = R.JobState.SUBMITTED
    e.cooldown_cells = {'c': 5.0}
    d = e.to_dict()
    self.assertEqual(d['state'], 'SUBMITTED')
    e2 = R.QueueEntry.from_dict(d)
    self.assertEqual(e2.priority, 3)
    self.assertEqual(e2.state, R.JobState.SUBMITTED)
    self.assertEqual(e2.launch_kwargs['config'], 'remote_run')
    self.assertEqual(e2.cooldown_cells, {'c': 5.0})



class TopologyLockTest(unittest.TestCase):

  def test_geometry_equivalence(self):
    self.assertTrue(R.same_topology('v6p', 32, 'v7', 32))    # both 2x4x4
    self.assertTrue(R.same_topology('v4', 32, 'v5p', 32))    # both 2x4x4
    self.assertFalse(R.same_topology('v6p', 32, 'v6e', 32))  # 2x4x4 vs 4_8
    self.assertFalse(R.same_topology('v6p', 32, 'v6p', 64))  # 2x4x4 vs 4x4x4

  def test_unlocked_job_ignores_geometry(self):
    # not locked: v6e is a valid fallback even though its geometry differs from
    # v6p. Power math: v6p-32 == 138.88 v5p-eq, window [104.16, 208.32]; the
    # equivalent v6e shape is v6e-64 (128), NOT v6e-32 (64, below window). This
    # doubles as a check that power-equivalence picks the right chip count.
    e = _entry(power='v6p-32', archs=('v6e',))
    shapes = R.candidate_shapes(e)
    self.assertIn(('v6e', 64), shapes)
    self.assertNotEqual(R.geometry_of('v6e', 64), '2x4x4')  # geometry differs, still allowed

  def test_locked_job_restricts_to_pinned_geometry(self):
    # paligemma-style: locked on 2x4x4, allows v6p+v7+v6e families
    e = _entry(power='v6p-32', archs=('v7', 'v6p', 'v6e'),
               topology_locked=True, locked_geometry='2x4x4')
    shapes = R.candidate_shapes(e)
    # v7-32 and v6p-32 are 2x4x4 -> allowed; v6e-* (4_8/etc) -> excluded
    self.assertIn(('v7', 32), shapes)
    self.assertIn(('v6p', 32), shapes)
    self.assertTrue(all(R.geometry_of(a, c) == '2x4x4' for a, c in shapes))

  def test_locked_job_can_move_v6p_to_v7(self):
    # the operator's example: locked job pending on v6p-32 re-routes to v7-32
    e = _entry(power='v6p-32', archs=('v7', 'v6p'),
               topology_locked=True, locked_geometry='2x4x4')
    avail = {'c': _avail('c', 'v7', free=320)}   # only v7 free
    p = _ok(R.plan_one(e, avail, now=0.0))
    self.assertIsNotNone(p)
    self.assertEqual((p.arch, p.chips), ('v7', 32))

  def test_locked_job_refuses_different_geometry_even_if_free(self):
    # locked on 2x4x4; only v6e (4_8) is free -> must NOT place
    e = _entry(power='v6p-32', archs=('v6p', 'v6e'),
               topology_locked=True, locked_geometry='2x4x4')
    avail = {'c': _avail('c', 'v6e', free=3200)}   # tons of v6e free
    self.assertIsNone(R.plan_one(e, avail, now=0.0))

  def test_locked_unpinned_uses_power_spec_geometry_as_anchor(self):
    # REGRESSION: a locked v6p-32 (2x4x4) job NOT yet placed, allowing v6e too.
    # Only a v6e-64 (8_8) slice is free and its power (64 v5p-eq) falls inside
    # the v6p-32 tolerance window -- but 8_8 != 2x4x4, so it MUST be refused.
    # Before the fix the unpinned branch applied no geometry filter and this
    # job landed on v6e-64, unrestorable for a 2x4x4-sharded checkpoint.
    e = _entry(power='v6p-32', archs=('v7', 'v6p', 'v6e'),
               topology_locked=True)
    self.assertIsNone(e.locked_geometry)
    avail = {'c': _avail('c', 'v6e', free=3200)}   # only v6e free
    self.assertIsNone(R.plan_one(e, avail, now=0.0))
    # candidate_shapes for this job must all be 2x4x4, never a v6e shape
    shapes = R.candidate_shapes(e)
    self.assertTrue(shapes)                          # v7-32 / v6p-32 qualify
    self.assertTrue(all(R.geometry_of(a, c) == '2x4x4' for a, c in shapes))
    self.assertFalse(any(a == 'v6e' for a, _ in shapes))

  def test_locked_bare_int_power_is_unplaceable(self):
    # A locked job whose power is a bare int names no arch => no anchor mesh.
    # Placing it would guess a geometry for a sharded checkpoint, so refuse.
    e = _entry(power='32', archs=('v6p', 'v7'), topology_locked=True)
    avail = {'c': _avail('c', 'v7', free=320)}
    self.assertIsNone(R.plan_one(e, avail, now=0.0))
    self.assertEqual(R.candidate_shapes(e), [])

  def test_power_geometry_helper(self):
    self.assertEqual(R.power_geometry('v6p-32'), '2x4x4')
    self.assertEqual(R.power_geometry('v7-32'), '2x4x4')
    self.assertEqual(R.power_geometry('v6e-64'), '8_8')
    self.assertIsNone(R.power_geometry('32'))          # bare int: no geometry

  def test_apply_placement_freezes_geometry_on_first_submit(self):
    # locked but not yet pinned: first placement freezes the mesh
    e = _entry(power='v6p-32', archs=('v7', 'v6p'), topology_locked=True)
    self.assertIsNone(e.locked_geometry)
    avail = {'c': _avail('c', 'v7', free=320)}
    p = _ok(R.plan_one(e, avail, now=0.0))
    R.apply_placement(e, p, xid='999', now=10.0)
    self.assertEqual(e.locked_geometry, '2x4x4')   # frozen from v7-32
    # after re-route, it stays pinned to 2x4x4
    R.mark_reroute(e, now=700.0, cooldown_s=1800.0)
    self.assertEqual(e.locked_geometry, '2x4x4')
    shapes = R.candidate_shapes(e)
    self.assertTrue(all(R.geometry_of(a, c) == '2x4x4' for a, c in shapes))

  def test_apply_placement_unlocked_does_not_pin(self):
    e = _entry(power='v6p-32', archs=('v7',))   # not locked
    avail = {'c': _avail('c', 'v7', free=320)}
    p = _ok(R.plan_one(e, avail, now=0.0))
    R.apply_placement(e, p, xid='1', now=0.0)
    self.assertIsNone(e.locked_geometry)



class TypeSelectionWeightTest(unittest.TestCase):

  def test_pool_weight_bounds(self):
    self.assertEqual(R.pool_weight(0), 1.0)
    self.assertAlmostEqual(R.pool_weight(4096), 1.20, places=2)
    self.assertGreater(R.pool_weight(4096 * 10), 1.20 - 1e-9)
    mid = R.pool_weight(64)
    self.assertGreater(mid, 1.0)
    self.assertLess(mid, 1.20)

  def test_effective_price_big_pool_reads_cheaper(self):
    # big pool (full 20% bonus) at 24 vs thin pool at 23: 24/1.2=20.0 beats
    # 23/~1.05=~21.9. A big pool rescues a modestly-pricier type.
    big = R.effective_price(24.0, 4096)
    thin = R.effective_price(23.0, 10)
    self.assertLess(big, thin)

  def test_effective_price_respects_20pct_ceiling(self):
    big = R.effective_price(30.0, 1e9)
    cheap = R.effective_price(24.0, 0)
    self.assertLess(cheap, big)

  def test_candidate_shapes_effective_price_ordering(self):
    e = _entry(power='v6p-32', archs=('v7', 'v6p'))
    shapes = R.candidate_shapes(
        e, arch_price={'v7': 20.0, 'v6p': 9.77},
        arch_pool={'v7': 4096, 'v6p': 30})
    self.assertEqual(shapes[0][0], 'v6p')

  def test_candidate_shapes_big_pool_wins_when_close(self):
    e = _entry(power='v6p-32', archs=('v7', 'v6p'))
    shapes = R.candidate_shapes(
        e, arch_price={'v7': 20.0, 'v6p': 18.0},
        arch_pool={'v7': 8192, 'v6p': 20})
    self.assertEqual(shapes[0][0], 'v7')

  def test_no_market_data_falls_back_to_arch_pref(self):
    e = _entry(power='v6p-32', archs=('v6p', 'v7'))
    shapes = R.candidate_shapes(e)
    self.assertEqual(shapes[0][0], 'v7')

  def test_plan_one_uses_effective_price_to_pick_type(self):
    e = _entry(power='v6p-32', archs=('v7', 'v6p'))
    avail = {
        'v7cell': _avail('v7cell', 'v7', free=3200, price=20.0),
        'v6pcell': _avail('v6pcell', 'v6p', free=3200, price=9.77),
    }
    p = _ok(R.plan_one(e, avail, now=0.0,
                       arch_price={'v7': 20.0, 'v6p': 9.77},
                       arch_pool={'v7': 4096, 'v6p': 30}))
    self.assertEqual(p.arch, 'v6p')


class SerialWorkerInvariantTest(unittest.TestCase):

  def _q(self, job_id, state, **kw):
    e = _entry(job_id)
    e.state = state
    for k, v in kw.items():
      setattr(e, k, v)
    return e

  def test_count_building(self):
    es = [self._q('a', R.JobState.QUEUED),
          self._q('b', R.JobState.BUILDING, build_started_at=100.0),
          self._q('c', R.JobState.SUBMITTED)]
    self.assertEqual(R.count_building(es), 1)

  def test_can_claim_false_when_live_build(self):
    es = [self._q('b', R.JobState.BUILDING, build_started_at=100.0)]
    self.assertFalse(R.can_claim_build(es, now=150.0, stale_after_s=1800.0))

  def test_can_claim_true_when_build_is_stale(self):
    es = [self._q('b', R.JobState.BUILDING, build_started_at=100.0)]
    # 100 + 1800 = 1900 < 2000 -> stale, slot is free again
    self.assertTrue(R.can_claim_build(es, now=2000.0, stale_after_s=1800.0))

  def test_can_claim_true_when_none_building(self):
    es = [self._q('a', R.JobState.QUEUED)]
    self.assertTrue(R.can_claim_build(es, now=0.0, stale_after_s=1800.0))

  def test_building_no_timestamp_is_stale(self):
    e = self._q('b', R.JobState.BUILDING, build_started_at=None)
    self.assertTrue(R.building_is_stale(e, now=0.0, stale_after_s=1800.0))

  def test_reclaim_stale_building(self):
    es = [self._q('b', R.JobState.BUILDING, build_started_at=0.0, worker_id='w1')]
    reclaimed = R.reclaim_stale_building(es, now=2000.0, stale_after_s=1800.0)
    self.assertEqual([e.job_id for e in reclaimed], ['b'])
    self.assertEqual(es[0].state, R.JobState.QUEUED)
    self.assertIsNone(es[0].build_started_at)
    self.assertIsNone(es[0].worker_id)

  def test_reclaim_leaves_live_building(self):
    es = [self._q('b', R.JobState.BUILDING, build_started_at=1000.0)]
    reclaimed = R.reclaim_stale_building(es, now=1100.0, stale_after_s=1800.0)
    self.assertEqual(reclaimed, [])
    self.assertEqual(es[0].state, R.JobState.BUILDING)

  def test_next_queued_priority(self):
    es = [self._q('lo', R.JobState.QUEUED, priority=1),
          self._q('hi', R.JobState.QUEUED, priority=9),
          self._q('bld', R.JobState.BUILDING, build_started_at=0.0)]
    self.assertEqual(_ok(R.next_queued(es)).job_id, 'hi')

  def test_next_queued_none_when_all_building_or_terminal(self):
    es = [self._q('b', R.JobState.BUILDING, build_started_at=0.0),
          self._q('d', R.JobState.DONE)]
    self.assertIsNone(R.next_queued(es))

  def test_claim_for_build_marks_and_stamps(self):
    e = self._q('a', R.JobState.QUEUED)
    R.claim_for_build(e, now=500.0, worker_id='w7')
    self.assertEqual(e.state, R.JobState.BUILDING)
    self.assertEqual(e.build_started_at, 500.0)
    self.assertEqual(e.worker_id, 'w7')


class BuildRequestedBackpressureTest(unittest.TestCase):
  """Step1: BUILD_REQUESTED handoff token + backpressure counting."""

  def _q(self, job_id, state, **kw):
    e = _entry(job_id)
    e.state = state
    for k, v in kw.items():
      setattr(e, k, v)
    return e

  def test_mark_build_requested_transitions_without_touching_attempts(self):
    e = self._q('a', R.JobState.QUEUED, attempts=2)
    R.mark_build_requested(e)
    self.assertEqual(e.state, R.JobState.BUILD_REQUESTED)
    self.assertEqual(e.attempts, 2)  # dispatch is not a failure

  def test_count_build_pending_counts_requested_and_building(self):
    es = [self._q('a', R.JobState.QUEUED),
          self._q('b', R.JobState.BUILD_REQUESTED),
          self._q('c', R.JobState.BUILDING, build_started_at=1.0),
          self._q('d', R.JobState.SUBMITTED),
          self._q('e', R.JobState.BUDGET_DEFERRED)]
    self.assertEqual(R.count_build_pending(es), 2)

  def test_count_build_pending_zero_means_builder_drained(self):
    es = [self._q('a', R.JobState.QUEUED), self._q('d', R.JobState.SUBMITTED)]
    self.assertEqual(R.count_build_pending(es), 0)

  def test_next_build_requested_priority_then_none(self):
    es = [self._q('lo', R.JobState.BUILD_REQUESTED, priority=1),
          self._q('hi', R.JobState.BUILD_REQUESTED, priority=9),
          self._q('q', R.JobState.QUEUED, priority=99)]  # QUEUED not eligible
    self.assertEqual(_ok(R.next_build_requested(es)).job_id, 'hi')
    for e in es:
      if e.state == R.JobState.BUILD_REQUESTED:
        e.state = R.JobState.BUILDING
    self.assertIsNone(R.next_build_requested(es))

  def test_next_queued_ignores_build_requested(self):
    # BUILD_REQUESTED must NOT be re-picked by next_queued (only the builder
    # claims it) -- otherwise a job dispatched this round gets double-dispatched.
    es = [self._q('r', R.JobState.BUILD_REQUESTED, priority=9),
          self._q('q', R.JobState.QUEUED, priority=1)]
    self.assertEqual(_ok(R.next_queued(es)).job_id, 'q')


class BudgetDeferredTest(unittest.TestCase):
  """Step1: BUDGET_DEFERRED soft park + per-round promote."""

  def _q(self, job_id, state, **kw):
    e = _entry(job_id)
    e.state = state
    for k, v in kw.items():
      setattr(e, k, v)
    return e

  def test_mark_budget_deferred_does_not_increment_attempts(self):
    e = self._q('a', R.JobState.BUILDING, attempts=1, build_started_at=5.0,
                worker_id='w1')
    R.mark_budget_deferred(e)
    self.assertEqual(e.state, R.JobState.BUDGET_DEFERRED)
    self.assertEqual(e.attempts, 1)          # budget refusal is NOT a failure
    self.assertIsNone(e.build_started_at)     # slot freed
    self.assertIsNone(e.worker_id)

  def test_mark_budget_deferred_never_becomes_held(self):
    # Even after many rounds of deferral, a job never accrues attempts toward HELD.
    e = self._q('a', R.JobState.QUEUED, attempts=0)
    for _ in range(10):
      R.mark_budget_deferred(e)
      R.promote_deferred([e])
    self.assertEqual(e.attempts, 0)
    self.assertNotEqual(e.state, R.JobState.HELD)

  def test_promote_deferred_returns_to_queued(self):
    es = [self._q('a', R.JobState.BUDGET_DEFERRED),
          self._q('b', R.JobState.QUEUED),
          self._q('c', R.JobState.BUDGET_DEFERRED)]
    promoted = R.promote_deferred(es)
    self.assertEqual(sorted(e.job_id for e in promoted), ['a', 'c'])
    self.assertTrue(all(e.state == R.JobState.QUEUED for e in es))

  def test_promote_deferred_noop_when_none_deferred(self):
    es = [self._q('b', R.JobState.QUEUED), self._q('r', R.JobState.RUNNING)]
    self.assertEqual(R.promote_deferred(es), [])


class ReconcileTest(unittest.TestCase):
  """Step1: XM-truth reconcile pure decision (R3 zombie cleanup)."""

  def _q(self, job_id, state, **kw):
    e = _entry(job_id)
    e.state = state
    for k, v in kw.items():
      setattr(e, k, v)
    return e

  # --- decide_reconcile truth table ---
  def test_terminal_from_running_is_failed(self):
    self.assertEqual(
        R.decide_reconcile(R.JobState.RUNNING, 'TERMINAL'), R.JobState.FAILED)

  def test_terminal_from_submitted_is_failed(self):
    self.assertEqual(
        R.decide_reconcile(R.JobState.SUBMITTED, 'TERMINAL'), R.JobState.FAILED)

  def test_running_promotes_submitted(self):
    self.assertEqual(
        R.decide_reconcile(R.JobState.SUBMITTED, 'RUNNING'), R.JobState.RUNNING)

  def test_running_running_is_noop(self):
    self.assertIsNone(R.decide_reconcile(R.JobState.RUNNING, 'RUNNING'))

  def test_pending_is_noop(self):
    # reroute (not reconcile) owns pending>deadline; reconcile leaves it.
    self.assertIsNone(R.decide_reconcile(R.JobState.SUBMITTED, 'PENDING'))

  def test_unknown_never_acts(self):
    # THE safety rule: a probe hiccup must never mark a live job dead.
    self.assertIsNone(R.decide_reconcile(R.JobState.RUNNING, 'UNKNOWN'))
    self.assertIsNone(R.decide_reconcile(R.JobState.SUBMITTED, 'UNKNOWN'))

  def test_unrecognised_status_is_noop(self):
    self.assertIsNone(R.decide_reconcile(R.JobState.RUNNING, 'WAT'))

  # --- reconcile_entry mutator ---
  def test_reconcile_entry_cleans_zombie(self):
    e = self._q('z', R.JobState.RUNNING, xid='123')
    self.assertTrue(R.reconcile_entry(e, 'TERMINAL'))
    self.assertEqual(e.state, R.JobState.FAILED)
    self.assertIn('zombie', e.last_reason)

  def test_reconcile_entry_promotes_placement(self):
    e = self._q('p', R.JobState.SUBMITTED, xid='123')
    self.assertTrue(R.reconcile_entry(e, 'RUNNING'))
    self.assertEqual(e.state, R.JobState.RUNNING)

  def test_reconcile_entry_unknown_leaves_unchanged(self):
    e = self._q('u', R.JobState.RUNNING, xid='123')
    self.assertFalse(R.reconcile_entry(e, 'UNKNOWN'))
    self.assertEqual(e.state, R.JobState.RUNNING)

  def test_reconcile_entry_skips_non_reconcilable(self):
    # QUEUED/HELD/BUDGET_DEFERRED/DONE/FAILED are never reconciled (no live xid).
    for st in (R.JobState.QUEUED, R.JobState.HELD, R.JobState.BUDGET_DEFERRED,
               R.JobState.DONE, R.JobState.FAILED, R.JobState.BUILD_REQUESTED):
      e = self._q('x', st)
      self.assertFalse(R.reconcile_entry(e, 'TERMINAL'),
                       f'{st} should not be reconciled')
      self.assertEqual(e.state, st)

  def test_reconcilable_states_membership(self):
    self.assertEqual(
        R.RECONCILABLE_STATES,
        frozenset({R.JobState.RUNNING, R.JobState.SUBMITTED, R.JobState.BUILDING}))


class NewStateSerdeTest(unittest.TestCase):
  """Step1: the two new states survive the JSON round-trip (readable strings)."""

  def test_build_requested_roundtrip(self):
    e = _entry()
    e.state = R.JobState.BUILD_REQUESTED
    d = e.to_dict()
    self.assertEqual(d['state'], 'BUILD_REQUESTED')
    self.assertEqual(R.QueueEntry.from_dict(d).state, R.JobState.BUILD_REQUESTED)

  def test_budget_deferred_roundtrip(self):
    e = _entry()
    e.state = R.JobState.BUDGET_DEFERRED
    d = e.to_dict()
    self.assertEqual(d['state'], 'BUDGET_DEFERRED')
    self.assertEqual(R.QueueEntry.from_dict(d).state, R.JobState.BUDGET_DEFERRED)


class PlanDispatchTest(unittest.TestCase):
  """Step3: greedy dispatch with in-memory pre-debit (route_lib.plan_dispatch)."""

  def _q(self, job_id, priority=0):
    e = _entry(job_id)
    e.state = R.JobState.QUEUED
    e.priority = priority
    return e

  def _plan(self, entries, headroom, costs, exempt=()):
    cost_of = lambda e: costs.get(e.job_id, 0.0)
    is_exempt = lambda e: e.job_id in exempt
    return R.plan_dispatch(entries, headroom, cost_of, is_exempt)

  def test_empty_is_empty(self):
    self.assertEqual(self._plan([], 100.0, {}), [])

  def test_all_fit_all_dispatched(self):
    es = [self._q('a'), self._q('b')]
    out = self._plan(es, 100.0, {'a': 30.0, 'b': 40.0})
    self.assertTrue(all(d.decision == R.JobState.BUILD_REQUESTED for d in out))
    self.assertAlmostEqual(out[-1].headroom_after, 30.0)  # 100-30-40

  def test_pre_debit_stops_over_dispatch(self):
    # Two 60-cost jobs, headroom 100: only the first fits (60), second deferred
    # (60 > 40 left). Without pre-debit both would be admitted against 100.
    es = [self._q('a', priority=2), self._q('b', priority=1)]
    out = self._plan(es, 100.0, {'a': 60.0, 'b': 60.0})
    self.assertEqual(out[0].decision, R.JobState.BUILD_REQUESTED)
    self.assertEqual(out[1].decision, R.JobState.BUDGET_DEFERRED)

  def test_priority_order(self):
    es = [self._q('lo', priority=1), self._q('hi', priority=9)]
    out = self._plan(es, 1000.0, {'lo': 1.0, 'hi': 1.0})
    self.assertEqual(out[0].job_id, 'hi')  # highest priority first

  def test_head_of_line_does_not_block_smaller(self):
    # Big expensive job at head does NOT fit; a smaller cheaper job behind it
    # STILL gets dispatched (fixes H2 starvation).
    es = [self._q('big', priority=9), self._q('small', priority=1)]
    out = self._plan(es, 100.0, {'big': 500.0, 'small': 50.0})
    by = {d.job_id: d.decision for d in out}
    self.assertEqual(by['big'], R.JobState.BUDGET_DEFERRED)
    self.assertEqual(by['small'], R.JobState.BUILD_REQUESTED)  # not blocked

  def test_exempt_dispatched_without_debit(self):
    # An exempt job (g5/BATCH/CPU) is dispatched and does NOT consume headroom.
    es = [self._q('ex', priority=9), self._q('paid', priority=1)]
    out = self._plan(es, 50.0, {'ex': 999.0, 'paid': 50.0}, exempt={'ex'})
    by = {d.job_id: d for d in out}
    self.assertEqual(by['ex'].decision, R.JobState.BUILD_REQUESTED)
    self.assertEqual(by['ex'].cost, 0.0)                 # no debit for exempt
    self.assertEqual(by['paid'].decision, R.JobState.BUILD_REQUESTED)  # 50<=50 still

  def test_all_over_bar_all_deferred(self):
    es = [self._q('a'), self._q('b')]
    out = self._plan(es, 10.0, {'a': 100.0, 'b': 100.0})
    self.assertTrue(all(d.decision == R.JobState.BUDGET_DEFERRED for d in out))

  def test_zero_headroom_defers_paid_admits_exempt(self):
    es = [self._q('paid'), self._q('ex')]
    out = self._plan(es, 0.0, {'paid': 1.0, 'ex': 1.0}, exempt={'ex'})
    by = {d.job_id: d.decision for d in out}
    self.assertEqual(by['paid'], R.JobState.BUDGET_DEFERRED)
    self.assertEqual(by['ex'], R.JobState.BUILD_REQUESTED)

  def test_exact_fit_admitted(self):
    es = [self._q('a')]
    out = self._plan(es, 50.0, {'a': 50.0})
    self.assertEqual(out[0].decision, R.JobState.BUILD_REQUESTED)  # <= is inclusive
    self.assertAlmostEqual(out[0].headroom_after, 0.0)


if __name__ == '__main__':
  unittest.main()
