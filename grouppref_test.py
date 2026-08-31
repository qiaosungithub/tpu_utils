from google3.testing.pybase import googletest as absltest
import os, tempfile
from google3.experimental.users.qiaos.tpu_utils import route_check as RC
from google3.experimental.users.qiaos.tpu_utils import route_lib as R

def mkq(path, n=2):
  es = []
  for i in range(n):
    e = R.QueueEntry(job_id=f'j{i}', power='v7-32', allowed_archs=['v7'])
    e.state = R.JobState.QUEUED; e.arch='v7'; e.chips=32; e.tier='PROD'
    es.append(e)
  RC.save_queue(path, es)

class GroupPrefTest(absltest.TestCase):
  def test_prefers_g5_when_it_fits(self):
    def seam(t, tier='PROD', lo='', group=''):
      if group in ('5', '3'):
        return {'headroom':2000.0,'new_cost':32.0,'exempt':True,'fits':True,'bar':2245.5,'current':180.0}
      return {'headroom':0.0,'new_cost':32.0,'exempt':False,'fits':False,'bar':2245.5,'current':9999.0}
    q = os.path.join(tempfile.mkdtemp(), 'q.json'); mkq(q)
    RC.run_dispatch_once(q, now=1e9, budget_query_fn=seam, dry_run=False)
    for e in RC.load_queue(q):
      self.assertEqual(e.group, '5', 'g5 must win when it fits')
      self.assertEqual(e.state, R.JobState.BUILD_REQUESTED)

  def test_falls_back_to_g3_then_g9(self):
    def seam(t, tier='PROD', lo='', group=''):
      if group == '3':
        return {'headroom':500.0,'new_cost':32.0,'exempt':True,'fits':True,'bar':2245.5,'current':0.0}
      return {'headroom':0.0,'new_cost':32.0,'exempt':(group=='5'),'fits':False,'bar':2245.5,'current':9999.0}
    q = os.path.join(tempfile.mkdtemp(), 'q.json'); mkq(q)
    RC.run_dispatch_once(q, now=1e9, budget_query_fn=seam, dry_run=False)
    for e in RC.load_queue(q):
      self.assertEqual(e.group, '3', 'g3 must win when g5 is full')

  def test_g9_bar_still_refuses_when_nothing_fits(self):
    # NEGATIVE CONTROL: the preference must never widen admission.
    def seam(t, tier='PROD', lo='', group=''):
      return {'headroom':0.0,'new_cost':32.0,'exempt':False,'fits':False,'bar':2245.5,'current':9999.0}
    q = os.path.join(tempfile.mkdtemp(), 'q.json'); mkq(q)
    RC.run_dispatch_once(q, now=1e9, budget_query_fn=seam, dry_run=False)
    for e in RC.load_queue(q):
      self.assertEqual(e.state, R.JobState.BUDGET_DEFERRED,
                       'g9 income/10 gate must still defer')

if __name__ == '__main__':
  absltest.main()
