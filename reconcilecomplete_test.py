# Copyright 2026 Google LLC. All Rights Reserved.
"""reconcile must tell 'it finished' apart from 'it died'.

Both were STATUS_TERMINAL until it was measured that 105 of 227 local-queue
rows had been reconciled to FAILED -- 100% of the rows carrying the 'zombie
cleaned up' reason -- including jobs whose results were already being used.
The success/failure bit was destroyed inside the probe (is_failed OR
is_completed OR is_stopped), so no downstream logic could recover it.

The NEGATIVE controls are the load-bearing half of this file: a job that
really failed must STILL be written FAILED, and a probe failure must still be
ignored. A test that only checks 'completed is no longer mislabelled' would
pass just as happily if reconcile stopped failing anything at all.
"""

from google3.experimental.users.qiaos.tpu_utils import route_check as RC
from google3.experimental.users.qiaos.tpu_utils import route_lib as R
from google3.testing.pybase import googletest


class _WU:
  """Only the booleans XManagerStatusProbe.status() reads off a work unit."""

  def __init__(self, pending=False, running=False, failed=False,
               completed=False, stopped=False):
    self.is_pending = pending
    self.is_running = running
    self.is_failed = failed
    self.is_completed = completed
    self.is_stopped = stopped


class ClassifyWuStatesTest(googletest.TestCase):

  def test_completed_wu_is_COMPLETED_not_TERMINAL(self):
    self.assertEqual(
        RC.classify_wu_states(False, False, False, True), RC.STATUS_COMPLETED)

  def test_completed_outranks_stopped_on_the_same_wu(self):
    # XManager can set is_completed AND is_stopped on one finished work unit.
    self.assertEqual(
        RC.classify_wu_states(False, False, True, True), RC.STATUS_COMPLETED)

  # --- NEGATIVE CONTROL: a real failure must still classify as TERMINAL ---
  def test_NEGCTL_failed_wu_still_TERMINAL(self):
    self.assertEqual(
        RC.classify_wu_states(False, False, True, False), RC.STATUS_TERMINAL)

  # --- NEGATIVE CONTROL: the pre-existing 3-arg contract is unchanged ---
  def test_NEGCTL_legacy_three_arg_calls_unchanged(self):
    self.assertEqual(RC.classify_wu_states(True, False, False),
                     RC.STATUS_PENDING)
    self.assertEqual(RC.classify_wu_states(False, True, False),
                     RC.STATUS_RUNNING)
    self.assertEqual(RC.classify_wu_states(True, True, True),
                     RC.STATUS_TERMINAL)
    self.assertEqual(RC.classify_wu_states(False, False, False),
                     RC.STATUS_RUNNING)


class DecideReconcileTest(googletest.TestCase):

  def test_completed_becomes_DONE(self):
    for local in (R.JobState.RUNNING, R.JobState.SUBMITTED,
                  R.JobState.BUILDING):
      self.assertEqual(
          R.decide_reconcile(local, 'COMPLETED'), R.JobState.DONE, str(local))

  # --- NEGATIVE CONTROL: a real failure must still become FAILED ---
  def test_NEGCTL_terminal_still_FAILED(self):
    for local in (R.JobState.RUNNING, R.JobState.SUBMITTED,
                  R.JobState.BUILDING):
      self.assertEqual(
          R.decide_reconcile(local, 'TERMINAL'), R.JobState.FAILED, str(local))

  # --- NEGATIVE CONTROL: UNKNOWN stays fail-closed; never act blind ---
  def test_NEGCTL_unknown_and_gibberish_never_act(self):
    for status in ('UNKNOWN', 'WAT', ''):
      self.assertIsNone(R.decide_reconcile(R.JobState.RUNNING, status), status)


class ReconcileEntryTest(googletest.TestCase):

  def _entry(self, state):
    entry = R.QueueEntry(job_id='t', power='h100-8', allowed_archs=['h100'])
    entry.state = state
    entry.xid = '1'
    return entry

  def test_completed_entry_is_DONE_and_not_called_a_zombie(self):
    entry = self._entry(R.JobState.RUNNING)
    self.assertTrue(R.reconcile_entry(entry, 'COMPLETED'))
    self.assertEqual(entry.state, R.JobState.DONE)
    self.assertIn('COMPLETED', entry.last_reason)
    self.assertNotIn('zombie', entry.last_reason)

  # --- NEGATIVE CONTROL: a real zombie keeps its wording and its state ---
  def test_NEGCTL_failed_entry_still_reads_zombie(self):
    entry = self._entry(R.JobState.RUNNING)
    self.assertTrue(R.reconcile_entry(entry, 'TERMINAL'))
    self.assertEqual(entry.state, R.JobState.FAILED)
    self.assertIn('zombie cleaned up', entry.last_reason)


if __name__ == '__main__':
  googletest.main()
