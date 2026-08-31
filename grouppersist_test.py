from google3.testing.pybase import googletest as absltest
import os, tempfile
from google3.experimental.users.qiaos.tpu_utils import route_check as RC
from google3.experimental.users.qiaos.tpu_utils import route_lib as R

class GroupPersistTest(absltest.TestCase):
  """The group chosen at dispatch must SURVIVE the build+submit round-trip."""

  def test_group_survives_claim_and_submit(self):
    q = os.path.join(tempfile.mkdtemp(), 'q.json')
    e = R.QueueEntry(job_id='j', power='v7-32', allowed_archs=['v7'])
    e.state = R.JobState.QUEUED; e.arch='v7'; e.chips=32; e.tier='PROD'
    RC.save_queue(q, [e])

    def seam(t, tier='PROD', lo='', group=''):
      if group in ('5', '3'):
        return {'headroom':2000.0,'new_cost':32.0,'exempt':True,'fits':True}
      return {'headroom':0.0,'new_cost':32.0,'exempt':False,'fits':False}
    RC.run_dispatch_once(q, now=1e9, budget_query_fn=seam, dry_run=False)
    got = RC.load_queue(q)[0]
    self.assertEqual(got.group, '5', 'group must be persisted at BUILD_REQUESTED')

    # now simulate the builder recording a successful submit
    p = R.Placement(job_id='j', arch='v7', chips=32, cell='c1', price=1.0,
                    reason='r', geometry='2x4x4')
    def _rec(x: R.QueueEntry) -> None:
      R.apply_placement(x, p, '999', 1e9)
    RC.update_entry(q, 'j', _rec)
    after = RC.load_queue(q)[0]
    self.assertEqual(after.group, '5',
                     'group must SURVIVE apply_placement (it is what the '
                     'builder submitted under; losing it makes the whole '
                     'preference unauditable after the fact)')

  def test_stale_snapshot_does_not_erase_group(self):
    """Regression: reconcile reads a snapshot, does slow XM RPCs, then merges.

    If dispatch admitted the row under g5 during those RPCs, the stale copy
    (group=None) must NOT overwrite the live value."""
    q = os.path.join(tempfile.mkdtemp(), 'q.json')
    e = R.QueueEntry(job_id='j', power='v7-32', allowed_archs=['v7'])
    e.state = R.JobState.SUBMITTED; e.xid = '1'
    RC.save_queue(q, [e])
    stale = RC.load_queue(q)[0]          # snapshot taken BEFORE dispatch writes

    live = RC.load_queue(q)[0]           # dispatch admits it under g5
    live.group = '5'
    RC.save_queue(q, [live])

    stale.state = R.JobState.RUNNING     # the pass's own (legitimate) change
    RC.merge_and_save_touched(q, [stale])
    got = RC.load_queue(q)[0]
    self.assertEqual(got.state, R.JobState.RUNNING, 'the pass keeps its change')
    self.assertEqual(got.group, '5',
                     'a stale snapshot must not revert the group choice')

if __name__ == '__main__':
  absltest.main()
