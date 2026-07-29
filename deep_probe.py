"""Forensic dump of everything XManager knows about a (possibly dead) XID.

Read-only. Prints the full work-unit proto including
`work_unit_executables[].borg` -- which carries persisted Borg locator
metadata (cell / borguser / job_name) that survives the job's death, unlike
`borg_job_states` (which needs populate_detailed_executable_status=True and
is empty once the job is gone).
"""

from absl import app
from absl import flags
from google.protobuf import text_format

from google3.learning.deepmind.xmanager2.client import xmanager_api

_XIDS = flags.DEFINE_list('xids', [], 'Experiment ids to inspect.')
_FULL = flags.DEFINE_bool('full', False, 'Dump the whole WU proto.')
_SERVICE = flags.DEFINE_bool('service', False, 'Dump evaluated BorgService.')


def _try(label, fn):
  try:
    v = fn()
  except Exception as e:  # pylint: disable=broad-except
    print(f'    {label}: <ERR {type(e).__name__}: {e}>')
    return None
  print(f'    {label} = {v!r}')
  return v


def main(argv):
  del argv
  client = xmanager_api.XManagerApi()
  for raw in _XIDS.value:
    xid = int(raw)
    print(f'\n{"="*78}\n===== XID {xid}\n{"="*78}')
    try:
      exp = client.get_experiment(xid)
    except Exception as e:  # pylint: disable=broad-except
      print(f'  get_experiment failed: {type(e).__name__}: {e}')
      continue

    print('--- EXPERIMENT ---')
    for a in ('name', 'display_name', 'author', 'status_name', 'notes',
              'tags', 'creation_time', 'completion_time', 'launch_time',
              'launch_script', 'launch_args', 'user_command', 'importance',
              'borgusers', 'attribution_urls', 'max_parallel_work_units'):
      _try(a, lambda a=a: getattr(exp, a))
    _try('status_class', lambda: exp.status_class)
    _try('resource_alloc_key', lambda: str(exp.resource_alloc_key).strip())
    _try('scheduling_settings', lambda: str(exp.scheduling_settings).strip())
    _try('citc_info', lambda: str(exp.citc_info).strip())
    _try('snapshot_info', lambda: str(exp.snapshot_info).strip()[:600])
    try:
      ep = exp._get_experiment_proto()  # pylint: disable=protected-access
      print('    --- experiment_proto (selected) ---')
      for f in ('description', 'status', 'plan', 'scheduling_settings'):
        if ep.HasField(f) if f in [x.name for x in ep.DESCRIPTOR.fields
                                   if x.message_type] else True:
          try:
            print(f'      {f}: {str(getattr(ep, f)).strip()[:900]}')
          except Exception:  # pylint: disable=broad-except
            pass
    except Exception as e:  # pylint: disable=broad-except
      print(f'    experiment_proto: <ERR {type(e).__name__}: {e}>')

    # -- work units, WITH detailed executable status --
    for detailed in (True,):
      print(f'--- WORK UNITS (populate_detailed_executable_status={detailed}) ---')
      try:
        wus = list(exp.get_work_units(
            populate_detailed_executable_status=detailed))
      except Exception as e:  # pylint: disable=broad-except
        print(f'  get_work_units failed: {type(e).__name__}: {e}')
        continue
      for wu in wus:
        print(f'  --- WU {wu.id} state={wu.status_name}')
        _try('status', lambda wu=wu: str(wu.status).strip())
        _try('creation_time', lambda wu=wu: wu.creation_time)
        _try('identity', lambda wu=wu: wu.identity)
        _try('executable_label', lambda wu=wu: wu.executable_label)
        _try('xborg_scu_id', lambda wu=wu: wu.xborg_scu_id)
        _try('stop_data', lambda wu=wu: str(wu.stop_data))
        _try('restart_data', lambda wu=wu: str(wu.restart_data))
        _try('scheduling_advice', lambda wu=wu: wu.get_scheduling_advice())
        _try('borgusers', lambda wu=wu: wu.borgusers)
        _try('borg_group_reservation_names',
             lambda wu=wu: wu.borg_group_reservation_names)
        _try('mpm_packages', lambda wu=wu: [str(p).strip()
                                            for p in wu.mpm_packages])
        _try('configuration', lambda wu=wu: wu.configuration)
        _try('parameters', lambda wu=wu: wu.parameters)

        print('    [borg_job_states]')
        try:
          bjs_list = wu.borg_job_states or []
          if not bjs_list:
            print('      <EMPTY>')
          for b in bjs_list:
            print(f'      cell={b.cell} user={b.user} job_name={b.job_name} '
                  f'uid={b.job_uid} tasks={b.total_tasks} '
                  f'exe={b.executable_name}')
            print(f'      task_state_counts={b.task_state_counts}')
            for line in (b.status_message_summary or []):
              print(f'      status: {line}')
        except Exception as e:  # pylint: disable=broad-except
          print(f'      <ERR {type(e).__name__}: {e}>')

        print('    [work_unit_executables -> borg]')
        try:
          p = wu.to_proto()
          if not p.work_unit_executables:
            print('      <no work_unit_executables>')
          for i, exe in enumerate(p.work_unit_executables):
            print('      exe[%d] fields=%s' % (
                i, [f.name for f, _ in exe.ListFields()]))
            if not exe.HasField('borg'):
              print(f'      exe[{i}] has no borg field; raw:\n'
                    f'{text_format.MessageToString(exe)[:2000]}')
              continue
            b = exe.borg
            print(f'      borg_job_name_substring = {b.borg_job_name_substring!r}')
            print(f'      initially_paused        = {b.initially_paused}')
            print(f'      xborg_scu_id            = {b.xborg_scu_id}')
            print(f'      xborg_scu_name          = {b.xborg_scu_name!r}')
            print(f'      xborg_scu_importance    = {b.xborg_scu_importance}')
            print(f'      has_custom_borg_job_names = {b.has_custom_borg_job_names}')
            print(f'      launch_config_mode      = {b.launch_config_mode}')
            print(f'      scu_affinity_group      = '
                  f'{text_format.MessageToString(b.scu_affinity_group).strip()}')
            for j in b.jobs:
              print(f'      >>> JOB service_name={j.names.service_name!r} '
                    f'job_name={j.names.job_name!r} borguser={j.borguser!r} '
                    f'cell={j.cell!r} priority={j.priority} '
                    f'task_replicas={j.task_replicas}')
            if b.borg_configuration:
              print('      --- borg_configuration (BCL) ---')
              print(b.borg_configuration[:8000])
            if _SERVICE.value and b.HasField('borg_service'):
              print('      --- borg_service (evaluated) ---')
              print(text_format.MessageToString(b.borg_service))
        except Exception as e:  # pylint: disable=broad-except
          import traceback
          print(f'      <ERR {type(e).__name__}: {e}>')
          traceback.print_exc()

        if _FULL.value:
          print('    [FULL WU PROTO]')
          try:
            print(text_format.MessageToString(wu.to_proto()))
          except Exception as e:  # pylint: disable=broad-except
            print(f'      <ERR {e}>')

        print('    [artifacts]')
        try:
          arts = list(wu.get_artifacts())
          if not arts:
            print('      <none>')
          for a in arts:
            print(f'      {a.type} {a.artifact!r} desc={a.description!r}')
        except Exception as e:  # pylint: disable=broad-except
          print(f'      <ERR {type(e).__name__}: {e}>')


if __name__ == '__main__':
  app.run(main)
