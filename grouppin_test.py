"""A caller's alloc-group PIN must beat the router's g5/g3-first preference.

WHY THIS EXISTS. The fleet-wide order ['5','3','9'] is right by default and
wrong for one specific job: g5 is `vqfree-xm`, whose GHOSTFISHLITE allotment
GQM can decline outright, so a job being preempted out of g5 needs the g9
floor even though g9 bills against the hard income/10 bar. Before the pin,
NO caller could say so: `tpu enqueue --group=9` was consumed by absl and never
reached the row, and a `launch_kwargs['group']` copy was dropped at build time
(correctly -- emitting it appended a SECOND `--group=`, and the last one wins).

The three tests below are the three ways this can silently go back to a no-op:
the pin is ignored, the pin leaks onto jobs that never asked, or the pin
reaches XM as a duplicate flag. Each has a NEGATIVE CONTROL asserting the
opposite outcome under the opposite input, so a test that cannot fail is
visible as such.
"""

import os
import tempfile

from google3.experimental.users.qiaos.tpu_utils import route_check as RC
from google3.experimental.users.qiaos.tpu_utils import route_lib as R
from google3.testing.pybase import googletest as absltest


def _mkq(path, n=1, pin=None, kwargs_group=None):
  """A queue of `n` QUEUED v7-32 rows, optionally carrying a pin."""
  es = []
  for i in range(n):
    lk = {'config': 'c'}
    if kwargs_group is not None:
      lk['group'] = kwargs_group
    e = R.QueueEntry(job_id=f'j{i}', power='v7-32', allowed_archs=['v7'],
                     launch_kwargs=lk)
    e.state = R.JobState.QUEUED
    e.arch, e.chips, e.tier = 'v7', 32, 'PROD'
    if pin is not None:
      e.pin_group = pin
    es.append(e)
  RC.save_queue(path, es)
  return path


def _q():
  return os.path.join(tempfile.mkdtemp(), 'q.json')


def _seam_all_fit(_t, tier='PROD', lo='', group=''):
  """Every pool admits. Under this seam the ORDER alone decides, so anything
  other than g5 can only come from a pin."""
  del tier, lo
  return {'headroom': 2000.0, 'new_cost': 216.0, 'fits': True,
          'exempt': group in ('3', '5'), 'bar': 2256.9, 'current': 21.0}


class GroupPinTest(absltest.TestCase):

  # --- 1. the pin is honoured, and its absence still yields g5 ---------------

  def test_pin_g9_beats_the_g5_first_order(self):
    q = _mkq(_q(), pin='9')
    RC.run_dispatch_once(q, now=1e9, budget_query_fn=_seam_all_fit,
                         dry_run=False)
    got = RC.load_queue(q)[0]
    self.assertEqual(got.group, '9',
                     'a caller pin of g9 must win over the g5-first order')
    self.assertEqual(got.state, R.JobState.BUILD_REQUESTED)

  def test_negctl_no_pin_still_picks_g5(self):
    """NEGATIVE CONTROL for #1: same seam, no pin -> the preference is intact.

    Without this, a bug that pinned everything to g9 would pass test #1."""
    q = _mkq(_q())
    RC.run_dispatch_once(q, now=1e9, budget_query_fn=_seam_all_fit,
                         dry_run=False)
    got = RC.load_queue(q)[0]
    self.assertEqual(got.group, '5',
                     'an unpinned job must still follow the g5/g3-first order')

  # --- 2. the older spelling, and the empty-pin trap -------------------------

  def test_launch_kwargs_group_is_read_as_a_pin(self):
    """The 90 rows already carrying launch_kwargs['group']='9' must work with
    ZERO caller changes -- that was the point of reading both spellings."""
    q = _mkq(_q(), kwargs_group='9')
    RC.run_dispatch_once(q, now=1e9, budget_query_fn=_seam_all_fit,
                         dry_run=False)
    self.assertEqual(RC.load_queue(q)[0].group, '9')

  def test_negctl_pin_to_g5_does_not_regress(self):
    """NEGATIVE CONTROL: a pin naming the pool the order would have chosen
    anyway must be a no-op, not an error -- pinning g5 stays g5."""
    q = _mkq(_q(), pin='5')
    RC.run_dispatch_once(q, now=1e9, budget_query_fn=_seam_all_fit,
                         dry_run=False)
    self.assertEqual(RC.load_queue(q)[0].group, '5')

  def test_negctl_blank_pin_is_not_a_pin(self):
    """NEGATIVE CONTROL: '' / '  ' must NOT pin to the empty group. An empty
    flag value is the single most likely way to accidentally submit under no
    group at all, and it would read as a deliberate choice on the row."""
    for blank in ('', '   '):
      self.assertIsNone(
          R.pinned_group(R.QueueEntry(job_id='b', power='v7-32',
                                      allowed_archs=['v7'], pin_group=blank)),
          f'blank pin {blank!r} must not count as a pin')
    q = _mkq(_q(), pin='')
    RC.run_dispatch_once(q, now=1e9, budget_query_fn=_seam_all_fit,
                         dry_run=False)
    self.assertEqual(RC.load_queue(q)[0].group, '5',
                     'a blank pin must fall through to the normal order')

  # --- 3. exactly ONE --group reaches XM, and the bar still binds ------------

  def test_pin_emits_exactly_one_group_flag(self):
    """The pin must not resurrect the duplicate-flag bug it replaces."""
    e = R.QueueEntry(job_id='j', power='v7-32', allowed_archs=['v7'],
                     launch_kwargs={'config': 'c', 'group': '9'})
    e.pin_group = '9'
    p = R.Placement(job_id='j', arch='v7', chips=32, cell='yucbfpv',
                    price=6.75, reason='r')
    argv = RC.build_tpu_queue_cmd(p, e, R.pinned_group(e) or '9')
    self.assertEqual([a for a in argv if a.startswith('--group=')],
                     ['--group=9'],
                     'exactly one --group= may be emitted; a second one wins '
                     'over the first and silently overrides the router')

  def test_negctl_pin_does_not_bypass_the_g9_bar(self):
    """NEGATIVE CONTROL, the expensive one: a pin buys a POOL, never a bypass
    of the income/10 gate. If g9 does not fit, a job pinned to g9 must be
    BUDGET_DEFERRED exactly like any other -- otherwise the pin becomes a way
    to spend past the one limit a human is held to."""
    def seam_nothing_fits(_t, tier='PROD', lo='', group=''):
      del tier, lo, group
      return {'headroom': 0.0, 'new_cost': 216.0, 'fits': False,
              'exempt': False, 'bar': 2256.9, 'current': 9999.0}
    q = _mkq(_q(), pin='9')
    RC.run_dispatch_once(q, now=1e9, budget_query_fn=seam_nothing_fits,
                         dry_run=False)
    got = RC.load_queue(q)[0]
    self.assertEqual(got.state, R.JobState.BUDGET_DEFERRED,
                     'the g9 income/10 gate must still defer a pinned job')

  # --- 4. the pin survives a concurrent reconcile snapshot -------------------

  def test_pin_survives_a_stale_merge(self):
    """merge_and_save_touched must carry the pin over from the LIVE row, the
    same way it already carries the router's `group`: a reconcile pass holding
    a pre-pin snapshot would otherwise write the row back unpinned."""
    q = _mkq(_q(), pin='9')
    stale = RC.load_queue(q)[0]
    stale.pin_group = None              # the snapshot taken before the pin
    stale.last_reason = 'reconciled'
    RC.merge_and_save_touched(q, [stale])
    self.assertEqual(RC.load_queue(q)[0].pin_group, '9',
                     'a stale snapshot must not erase the caller pin')


if __name__ == '__main__':
  absltest.main()
