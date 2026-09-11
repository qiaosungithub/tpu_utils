# Copyright 2026 Google LLC. All Rights Reserved.
"""A resolvable experiment with ZERO work units is GONE, not UNKNOWN.

Measured 2026-08-31: 13 local-queue rows sat SUBMITTED for 5-8 days in a
deadlock. `xmanager list --archived=true` reported every one of them
NOT_RUNNING, but the probe returned UNKNOWN because `get_experiment`
SUCCEEDED and then yielded no work units. reconcile's "never act on
UNKNOWN" guard therefore left them alone forever, and `tpu dequeue`
refuses any row that still carries an xid -- so no path could clean them
up. The 13 rows polluted every group/state reading taken off the queue.

The NEGATIVE CONTROLS are the load-bearing half of this file. Turning
"cannot read" into "it is dead" is exactly the mistake that would cancel
live cars, so:
  * a probe that FAILS (exception) must still be UNKNOWN and act on nothing;
  * a RUNNING car must still be RUNNING;
  * a JUST-SUBMITTED car -- briefly 0-WU too -- must NOT be reaped;
  * a genuinely failed car must still reach FAILED by the old TERMINAL path.
A test that only asserted "the 13 rows now clear" would pass just as well
if reconcile had started reaping everything.
"""

from google3.experimental.users.qiaos.tpu_utils import route_check as RC
from google3.experimental.users.qiaos.tpu_utils import route_lib as R
from google3.testing.pybase import googletest

_HOUR = 3600.0


class _WU:
  """Only the booleans XManagerStatusProbe.status() reads off a work unit."""

  def __init__(self, pending=False, running=False, failed=False,
               completed=False, stopped=False):
    self.is_pending = pending
    self.is_running = running
    self.is_failed = failed
    self.is_completed = completed
    self.is_stopped = stopped


class _Exp:

  def __init__(self, wus):
    self._wus = wus

  def get_work_units(self, populate_detailed_executable_status=False):
    del populate_detailed_executable_status
    return list(self._wus)


class _FakeClient:
  """get_experiment either returns an _Exp or raises (probe failure)."""

  def __init__(self, by_xid):
    self._by_xid = by_xid

  def get_experiment(self, xid):
    v = self._by_xid[int(xid)]
    if isinstance(v, Exception):
      raise v
    return v


def _probe(by_xid):
  p = RC.XManagerStatusProbe()
  p._client = _FakeClient(by_xid)  # pylint: disable=protected-access
  return p


def _entry(job_id='j', state=R.JobState.SUBMITTED, xid='1', submitted_at=0.0):
  e = R.QueueEntry(job_id=job_id, power='v6e-16', allowed_archs=['v6e'])
  e.state = state
  e.xid = xid
  e.submitted_at = submitted_at
  return e


class ProbeGoneTest(googletest.TestCase):

  def test_zero_work_units_is_GONE(self):
    p = _probe({1: _Exp([])})
    self.assertEqual(p.status('1'), RC.STATUS_GONE)

  # --- NEGATIVE CONTROL: a probe FAILURE is still UNKNOWN, never GONE ---
  def test_NEGCTL_probe_exception_still_UNKNOWN(self):
    p = _probe({1: RuntimeError('rpc down')})
    self.assertEqual(p.status('1'), RC.STATUS_UNKNOWN)

  # --- NEGATIVE CONTROL: a live car is untouched by this change ---
  def test_NEGCTL_running_car_still_RUNNING(self):
    p = _probe({1: _Exp([_WU(running=True)])})
    self.assertEqual(p.status('1'), RC.STATUS_RUNNING)

  def test_NEGCTL_pending_car_still_PENDING(self):
    p = _probe({1: _Exp([_WU(pending=True)])})
    self.assertEqual(p.status('1'), RC.STATUS_PENDING)


class DecideReconcileGoneTest(googletest.TestCase):

  def test_old_gone_row_becomes_FAILED(self):
    self.assertEqual(
        R.decide_reconcile(R.JobState.SUBMITTED, 'GONE', age_s=8 * 24 * _HOUR),
        R.JobState.FAILED)

  # --- NEGATIVE CONTROL: a JUST-submitted car is 0-WU too. Do not reap it. ---
  def test_NEGCTL_young_gone_row_is_left_alone(self):
    self.assertIsNone(
        R.decide_reconcile(R.JobState.SUBMITTED, 'GONE', age_s=60.0))

  # --- NEGATIVE CONTROL: age unknown (re-routed row) acts on nothing ---
  def test_NEGCTL_missing_age_is_left_alone(self):
    self.assertIsNone(R.decide_reconcile(R.JobState.SUBMITTED, 'GONE'))

  # --- NEGATIVE CONTROL: UNKNOWN still never acts, at any age ---
  def test_NEGCTL_unknown_never_acts_however_old(self):
    self.assertIsNone(
        R.decide_reconcile(R.JobState.SUBMITTED, 'UNKNOWN',
                           age_s=99 * 24 * _HOUR))

  # --- NEGATIVE CONTROL: the old verdicts are unchanged ---
  def test_NEGCTL_terminal_still_FAILED_and_completed_still_DONE(self):
    self.assertEqual(R.decide_reconcile(R.JobState.SUBMITTED, 'TERMINAL'),
                     R.JobState.FAILED)
    self.assertEqual(R.decide_reconcile(R.JobState.RUNNING, 'COMPLETED'),
                     R.JobState.DONE)
    self.assertEqual(R.decide_reconcile(R.JobState.SUBMITTED, 'RUNNING'),
                     R.JobState.RUNNING)
    self.assertIsNone(R.decide_reconcile(R.JobState.SUBMITTED, 'PENDING'))


class RunReconcileGoneTest(googletest.TestCase):
  """End to end through run_reconcile, which is what the daemon calls."""

  def test_the_13_row_deadlock_clears(self):
    now = 10 * 24 * _HOUR
    old = _entry('stale', xid='282431624', submitted_at=now - 8 * 24 * _HOUR)
    entries, log = RC.run_reconcile(
        [old], now=now, probe=_probe({282431624: _Exp([])}), dry_run=False)
    self.assertEqual(entries[0].state, R.JobState.FAILED)
    self.assertIn('GONE', '\n'.join(log))
    self.assertIn('ZERO work units', entries[0].last_reason)

  # --- NEGATIVE CONTROL: the same pass must not touch a live car ---
  def test_NEGCTL_live_car_survives_the_same_pass(self):
    now = 10 * 24 * _HOUR
    live = _entry('live', xid='285023898', submitted_at=now - 8 * 24 * _HOUR)
    entries, _ = RC.run_reconcile(
        [live], now=now,
        probe=_probe({285023898: _Exp([_WU(running=True)])}), dry_run=False)
    self.assertEqual(entries[0].state, R.JobState.RUNNING)

  # --- NEGATIVE CONTROL: a young 0-WU car survives ---
  def test_NEGCTL_young_car_survives_the_same_pass(self):
    now = 10 * 24 * _HOUR
    young = _entry('young', xid='999', submitted_at=now - 120.0)
    entries, _ = RC.run_reconcile(
        [young], now=now, probe=_probe({999: _Exp([])}), dry_run=False)
    self.assertEqual(entries[0].state, R.JobState.SUBMITTED)

  # --- NEGATIVE CONTROL: dry_run mutates nothing ---
  def test_NEGCTL_dry_run_does_not_mutate(self):
    now = 10 * 24 * _HOUR
    old = _entry('stale', xid='282431624', submitted_at=now - 8 * 24 * _HOUR)
    entries, _ = RC.run_reconcile(
        [old], now=now, probe=_probe({282431624: _Exp([])}), dry_run=True)
    self.assertEqual(entries[0].state, R.JobState.SUBMITTED)


if __name__ == '__main__':
  googletest.main()
