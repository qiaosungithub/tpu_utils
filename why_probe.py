"""Dump the raw XManager work-unit status for given XIDs.

`tpu check` renders `Rejected by Allocator/Borg` for any PROD failure it cannot
classify (infra_check.derive_failure_reason, "Priority Rule 4"), which hides the
real reason. This prints the unfiltered status message so the actual allocator
verdict is visible.
"""

from absl import app
from absl import flags

from google3.learning.deepmind.xmanager2.client import xmanager_api

_XIDS = flags.DEFINE_list('xids', [], 'Experiment ids to inspect.')


def main(argv):
  del argv
  client = xmanager_api.XManagerApi()
  for raw in _XIDS.value:
    xid = int(raw)
    print(f'===== XID {xid} =====')
    try:
      experiment = client.get_experiment(xid)
    except Exception as e:  # pylint: disable=broad-except
      print(f'  get_experiment failed: {type(e).__name__}: {e}')
      continue
    print(f'  name={experiment.name!r}')
    try:
      # populate_detailed_executable_status=True is what fills in
      # `borg_job_states` (cell / user / job_name). Without it the field is an
      # empty list and there is no way to reach the job's logs.
      work_units = list(experiment.get_work_units(
          populate_detailed_executable_status=True, properties=['*']))
    except Exception as e:  # pylint: disable=broad-except
      print(f'  get_work_units failed: {type(e).__name__}: {e}')
      continue
    print(f'  #work_units={len(work_units)}')
    for wu in work_units:
      print(f'  --- WU {getattr(wu, "id", "?")} state={wu.status_name}')
      status = getattr(wu, 'status', None)
      if status is not None:
        print(f'      status.message = {getattr(status, "message", None)!r}')
      for attr in ('status_message', 'error_message', 'failure_reason',
                   'details', 'cell', 'borg_job_name', 'creation_time',
                   'start_time', 'stop_time'):
        value = getattr(wu, attr, None)
        if value:
          print(f'      {attr} = {value!r}')
      # BorgJobState is the bridge from an XID to actual logs: it carries the
      # cell, borg user and job name that `borg tasklog` / analog need. Without
      # it there is no way to reach a job's stderr once XManager's own
      # status.message comes back empty (which it routinely does).
      try:
        for bjs in (wu.borg_job_states or []):
          print(f'      [borg] cell={bjs.cell} user={bjs.user} '
                f'job_name={bjs.job_name} tasks={bjs.total_tasks}')
          print(f'      [borg] task_states={bjs.task_state_counts}')
          for line in (bjs.status_message_summary or []):
            print(f'      [borg] status: {line}')
          print(f'      [borg] logcmd: borg --borg={bjs.cell} tasklog '
                f'--user={bjs.user} --job={bjs.job_name} --task=0 --stderr')
      except Exception as e:  # pylint: disable=broad-except
        print(f'      [borg] unavailable: {type(e).__name__}: {e}')

      # Surface anything that could carry a restart / failure count.
      interesting = [a for a in dir(wu)
                     if not a.startswith('_')
                     and any(k in a.lower() for k in
                             ('fail', 'restart', 'retry', 'attempt', 'count', 'step'))]
      for attr in interesting:
        try:
          value = getattr(wu, attr)
        except Exception:  # pylint: disable=broad-except
          continue
        if callable(value):
          continue
        print(f'      [scan] {attr} = {value!r}')


if __name__ == '__main__':
  app.run(main)
