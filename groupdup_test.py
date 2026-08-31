from google3.testing.pybase import googletest as absltest
from google3.experimental.users.qiaos.tpu_utils import route_check as RC
from google3.experimental.users.qiaos.tpu_utils import route_lib as R

class GroupDupTest(absltest.TestCase):
  def test_launch_kwargs_group_would_override_router(self):
    e = R.QueueEntry(job_id='j', power='h100-8', allowed_archs=['h100'])
    e.tier = 'PROD'
    e.launch_kwargs = {'group': '9', 'exp_name': 'x'}
    p = R.Placement(job_id='j', arch='h100', chips=8, cell='sh', price=1.0,
                    reason='r')
    argv = RC.build_tpu_queue_cmd(p, e, '5')   # router picked g5
    groups = [a for a in argv if a.startswith('--group')]
    print('ARGV:', ' '.join(argv))
    print('GROUP FLAGS:', groups)
    self.assertEqual(len(groups), 1,
                     f'two --group flags emitted: {groups}; the caller-supplied '
                     'one silently overrides the router choice')
    self.assertEqual(groups[0], '--group=5')

  def test_router_group_still_emitted_without_kwargs(self):
    # NEGATIVE CONTROL: dropping the passthrough must not drop the router's own
    # --group. A fix that emits none would be worse than the duplicate.
    e = R.QueueEntry(job_id='j', power='h100-8', allowed_archs=['h100'])
    e.tier = 'PROD'
    e.launch_kwargs = {'exp_name': 'x'}
    p = R.Placement(job_id='j', arch='h100', chips=8, cell='sh', price=1.0,
                    reason='r')
    argv = RC.build_tpu_queue_cmd(p, e, '3')
    self.assertIn('--group=3', argv)
    self.assertEqual([a for a in argv if a.startswith('--group')], ['--group=3'])

  def test_other_kwargs_still_pass_through(self):
    # NEGATIVE CONTROL: only `group` is filtered; everything else survives.
    e = R.QueueEntry(job_id='j', power='h100-8', allowed_archs=['h100'])
    e.launch_kwargs = {'group': '9', 'exp_name': 'keepme', 'load_from': '/p.pt'}
    p = R.Placement(job_id='j', arch='h100', chips=8, cell='sh', price=1.0,
                    reason='r')
    argv = RC.build_tpu_queue_cmd(p, e, '5')
    self.assertIn('--exp_name=keepme', argv)
    self.assertIn('--load_from=/p.pt', argv)

if __name__ == '__main__':
  absltest.main()
