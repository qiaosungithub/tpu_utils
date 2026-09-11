# Copyright 2026 Google LLC. All Rights Reserved.
"""Re-route churn brakes: per-row backoff and a global rate cap.

Measured 2026-09-01: `QueueEntry.reroutes` was WRITTEN by mark_reroute and read
by nothing, so attempt 1 and attempt 7 got the same 600s of patience. One eval
job burned 7 XIDs in 2.5 hours -- placed, cancelled at ~600s, re-queued, placed
again -- while the queue showed reroutes=2 because the first 5 lived on a row
that had since been dequeued. Two brakes, because each one alone has a hole:

  * per-row backoff   -- doubles patience per move; RESET by a re-enqueue.
  * global rate cap   -- keyed on NOTHING, counts what the re-router itself did,
                         so renaming or re-enqueuing the car cannot evade it.

There is deliberately NO give-up bound (removed 2026-09-11 by operator request):
a job is re-routed however many times it takes and is never auto-parked as HELD
for churning. entry.reroutes is surfaced on the board instead, so a human can
see a high count and step in -- that is the safety valve that replaced the bound.

The NEGATIVE CONTROLS carry this file. Every brake here can only make the
re-router act LESS, and the failure mode of "less" is a genuinely stuck car
that nobody rescues -- so each brake is paired with a test proving a first
offence still gets re-routed normally.
"""

import time as _time

from google3.experimental.users.qiaos.tpu_utils import route_check as RC
from google3.experimental.users.qiaos.tpu_utils import route_lib as R
from google3.testing.pybase import googletest

_BASE = 600.0


def _entry(reroutes=0, age_s=10_000.0, now=1_000_000.0,
           state=R.JobState.SUBMITTED):
  e = R.QueueEntry(job_id='j', power='h100-8', allowed_archs=['h100'])
  e.state = state
  e.xid = '285502876'
  e.reroutes = reroutes
  e.submitted_at = now - age_s
  return e


class BackoffTest(googletest.TestCase):

  def test_deadline_doubles_per_reroute(self):
    for n, want in ((0, 600.0), (1, 1200.0), (2, 2400.0), (3, 4800.0)):
      self.assertEqual(R.reroute_deadline_s(_entry(reroutes=n), _BASE), want)

  def test_deadline_is_capped_not_unbounded(self):
    cap = _BASE * (2 ** R.REROUTE_BACKOFF_MAX_DOUBLINGS)
    for n in (R.REROUTE_BACKOFF_MAX_DOUBLINGS, 20, 999):
      self.assertEqual(R.reroute_deadline_s(_entry(reroutes=n), _BASE), cap)

  def test_second_attempt_is_NOT_rerouted_at_the_old_deadline(self):
    # 700s in: past the flat 600s rule, inside the doubled 1200s one.
    now = 1_000_000.0
    self.assertFalse(
        R.needs_reroute(_entry(reroutes=1, age_s=700.0, now=now), now, _BASE))

  def test_second_attempt_IS_rerouted_once_its_own_deadline_passes(self):
    now = 1_000_000.0
    self.assertTrue(
        R.needs_reroute(_entry(reroutes=1, age_s=1300.0, now=now), now, _BASE))

  # --- NEGATIVE CONTROL: a FIRST offence must still be re-routed on time ---
  def test_NEGCTL_first_attempt_unchanged_at_600s(self):
    now = 1_000_000.0
    self.assertTrue(
        R.needs_reroute(_entry(reroutes=0, age_s=601.0, now=now), now, _BASE))
    self.assertFalse(
        R.needs_reroute(_entry(reroutes=0, age_s=599.0, now=now), now, _BASE))

  # --- NEGATIVE CONTROL: non-SUBMITTED / no submitted_at still never re-route
  def test_NEGCTL_preexisting_guards_intact(self):
    now = 1_000_000.0
    self.assertFalse(R.needs_reroute(
        _entry(state=R.JobState.RUNNING, now=now), now, _BASE))
    e = _entry(now=now)
    e.submitted_at = None
    self.assertFalse(R.needs_reroute(e, now, _BASE))


class NoGiveUpBoundTest(googletest.TestCase):
  """The give-up bound was REMOVED 2026-09-11 (operator request). A job is
  re-routed as many times as it takes and is never auto-parked as HELD for
  churning; the per-row backoff and the global rate cap are the only brakes."""

  def test_high_reroute_count_still_reroutes(self):
    # 99 moves in -- far past the old bound of 6 -- and it is STILL eligible
    # once its (capped) backoff deadline has passed. No bound turns it off.
    now = 1_000_000.0
    e = _entry(reroutes=99, age_s=999_999.0, now=now)
    self.assertTrue(R.needs_reroute(e, now, _BASE))

  def test_a_thousand_moves_in_still_reroutes(self):
    # There is no ceiling at all: even absurd counts stay eligible. The full
    # run_reroute path (never parking a high-count row as HELD) is covered by
    # route_check_test.test_high_reroute_count_is_still_rechecked_and_rerouted.
    now = 1_000_000.0
    e = _entry(reroutes=1000, age_s=999_999.0, now=now)
    self.assertTrue(R.needs_reroute(e, now, _BASE))


class GlobalBrakeTest(googletest.TestCase):

  def test_brake_trips_at_the_limit(self):
    now = 1_000_000.0
    times = [now - i * 60 for i in range(R.REROUTE_GLOBAL_MAX_PER_HOUR)]
    self.assertTrue(R.global_reroute_brake(times, now))

  # --- NEGATIVE CONTROL: below the limit the brake must stay OFF ---
  def test_NEGCTL_below_limit_does_not_trip(self):
    now = 1_000_000.0
    times = [now - i * 60 for i in range(R.REROUTE_GLOBAL_MAX_PER_HOUR - 1)]
    self.assertFalse(R.global_reroute_brake(times, now))

  # --- NEGATIVE CONTROL: OLD churn must age out of the window ---
  def test_NEGCTL_stale_history_does_not_trip_it(self):
    now = 1_000_000.0
    times = [now - 3601.0 - i for i in range(50)]   # all just outside the hour
    self.assertFalse(R.global_reroute_brake(times, now))

  # --- NEGATIVE CONTROL: empty history fails OPEN, never suspends ---
  def test_NEGCTL_empty_history_fails_open(self):
    self.assertFalse(R.global_reroute_brake([], 1_000_000.0))

  def test_tripped_brake_leaves_the_candidate_alone(self):
    now = _time.time()
    hist = [now - i for i in range(R.REROUTE_GLOBAL_MAX_PER_HOUR + 2)]
    path = self.create_tempfile().full_path
    RC._save_reroute_history(hist, path)
    e = _entry(reroutes=0, age_s=99_999.0, now=now)
    entries, log = RC.run_reroute([e], now=now, probe=_Probe('PENDING'),
                                  dry_run=False, history_file=path)
    self.assertEqual(entries[0].state, R.JobState.SUBMITTED)  # untouched
    self.assertIn('GLOBAL BRAKE ENGAGED', '\n'.join(log))


class HistoryPersistenceTest(googletest.TestCase):

  def test_roundtrip_keeps_only_the_window(self):
    now = _time.time()
    path = self.create_tempfile().full_path
    RC._save_reroute_history([now - 10, now - 20, now - 7200], path)
    got = RC._load_reroute_history(path)
    self.assertLen(got, 2)

  # --- NEGATIVE CONTROL: a corrupt/missing file must fail OPEN (empty) ---
  def test_NEGCTL_corrupt_history_reads_empty_not_crash(self):
    path = self.create_tempfile(content='{not json').full_path
    self.assertEqual(RC._load_reroute_history(path), [])
    self.assertEqual(RC._load_reroute_history('/nonexistent/x.json'), [])


class _Probe:

  def __init__(self, state):
    self._s = state

  def status(self, xid):
    del xid
    return self._s


if __name__ == '__main__':
  googletest.main()
