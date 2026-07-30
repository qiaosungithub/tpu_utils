import argparse
import datetime
import json
import os
import re
import sys
from absl import app
from absl import flags
from google3.experimental.users.qiaos.tpu_utils import group_utils
from google3.learning.deepmind.xmanager2.client import xmanager_api
from rich.console import Console
from rich.table import Table
from rich.align import Align

FLAGS = flags.FLAGS
flags.DEFINE_string('user', 'qiaos', 'User LDAP')


_JOBS_FILE = os.path.expanduser('~/.tpu_jobs.json')
_LEGACY_FILE = os.path.expanduser('~/.tpu_jobs_legacy.json')


def _load_json(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _clear_jobs(job_ids, mapping_dir):
    """Archive tracked jobs out of the status board.

    Entries are MOVED to ~/.tpu_jobs_legacy.json rather than deleted: the record
    carries the checkpoint bucket, staging dir and launch log, which is the only
    way back to a finished run's artefacts. `tpu check` reads only the live
    file, so archiving is enough to clean the board.

    `tpu clear all` archives every entry; otherwise pass explicit XIDs.
    """
    if not job_ids:
        print('Usage: tpu clear <xid> [xid...]   |   tpu clear all')
        return

    live = _load_json(_JOBS_FILE)
    legacy = _load_json(_LEGACY_FILE)

    if len(job_ids) == 1 and job_ids[0] == 'all':
        targets = sorted(live)
        # Legacy bucket-mapping dir predates ~/.tpu_jobs.json; sweep it too.
        if os.path.isdir(mapping_dir):
            targets += [f for f in os.listdir(mapping_dir) if f not in targets]
    else:
        targets = job_ids

    archived, missing = [], []
    for xid in targets:
        found = False
        if xid in live:
            entry = dict(live.pop(xid))
            entry['archived_at'] = datetime.datetime.now().isoformat(timespec='seconds')
            legacy[xid] = entry
            found = True
        target_file = os.path.join(mapping_dir, xid)
        if os.path.exists(target_file):
            legacy.setdefault(xid, {}).setdefault(
                'bucket_cp_path', open(target_file).read().strip())
            os.remove(target_file)
            found = True
        (archived if found else missing).append(xid)

    if archived:
        with open(_LEGACY_FILE, 'w') as f:
            json.dump(legacy, f, indent=2, sort_keys=True)
        with open(_JOBS_FILE, 'w') as f:
            json.dump(live, f, indent=2, sort_keys=True)
        print(f'Archived {len(archived)} job(s) to {_LEGACY_FILE}')
        print(f'  {len(live)} still tracked')
    if missing:
        print(f'Not tracked: {", ".join(missing)}')



def derive_failure_reason(exp_id, failed_wu, tpu_info):
    """Human-readable reason for a work unit that is not making progress.

    Ordering matters. The rules run most-specific first, because several xborg
    messages contain words that a looser rule would swallow -- e.g. the GQM
    price message says "costs exceed your configured limit order", which the
    old `'EXCEEDED' in msg -> Pool Capacity Limit` rule mislabelled as a
    capacity problem when it is really a pricing one.

    Reference for the message taxonomy: go/xborg-why and
    go/xborg-why-descheduled.
    """
    if not failed_wu:
        return 'No WorkUnits'

    msg = ''
    if hasattr(failed_wu, 'status') and hasattr(failed_wu.status, 'message'):
        msg = failed_wu.status.message or ''
    if not msg:
        msg = (getattr(failed_wu, 'status_message', '') or
               getattr(failed_wu, 'error_message', '') or
               getattr(failed_wu, 'failure_reason', '') or '')

    msg_upper = msg.upper()

    # Rule 1: explicit preemption wording.
    if 'PREEMPTED' in msg_upper or 'PREEMPT' in msg_upper:
        if 'HIGHER_PRIORITY' in msg_upper or 'HIGHER PRIORITY' in msg_upper:
            return 'Preempted (Higher Priority)'
        if 'DEFRAGMENTATION' in msg_upper or 'DEFRAG' in msg_upper:
            return 'Preempted (Defrag)'
        return 'Preempted'

    # Rule 2: DESCHEDULED == preempted. xborg does not use the word "preempt"
    # when it takes resources back; it says the workload was "descheduled".
    # Losing opportunistically-held capacity to an allotment with a guarantee
    # is exactly a preemption from the job's point of view, and reporting it as
    # anything softer hides real preemption pressure.
    # go/xborg-why-descheduled#resource-guarantee-reclaim
    if 'DESCHEDULED' in msg_upper:
        exhausted = _restart_budget_exhausted(failed_wu, tpu_info)
        if 'RECLAIM' in msg_upper or 'GUARANTEED CAPACITY' in msg_upper:
            base = 'Preempted (Guarantee Reclaim)'
        elif 'DEFRAGMENTATION' in msg_upper or 'DEFRAG' in msg_upper:
            base = 'Preempted (Defrag)'
        else:
            base = 'Preempted (Descheduled)'
        return f'{base} + resume exceeds limit' if exhausted else base

    # Rule 3: GQM pricing. Must precede the capacity rule below: the message
    # contains "exceed", but the workload is queued waiting for a lower price,
    # not short of capacity. It is still alive and may schedule later.
    if 'GQM_RESOURCE_DEFICIT' in msg_upper or 'LIMIT ORDER' in msg_upper:
        return 'Queued (GQM price over limit order)'

    # Rule 4: structural errors visible only in the launch log.
    launch_log = tpu_info.get('launch_log')
    if launch_log and os.path.exists(launch_log):
        try:
            with open(launch_log, 'r') as f:
                content = f.read().upper()
            if 'SLICE_DEFRAGMENTATION' in content:
                return 'Preempted (Defrag)'
            if 'PERMISSION_DENIED' in content or 'UNAUTHENTICATED' in content:
                return 'Permission Denied'
            if 'UNSUPPORTED_TOPOLOGY' in content or 'INVALID_TOPOLOGY' in content:
                return 'Unsupported Topology'
        except Exception:
            pass

    # Rule 5: remaining explicit message checks.
    if 'UNSUPPORTED' in msg_upper and 'TOPOLOGY' in msg_upper:
        return 'Unsupported Topology'

    if 'FLEX_CEILING_EXCEEDED' in msg_upper or 'DEFICIT_IN_PARENT_POOLS' in msg_upper:
        return 'PROD quota ceiling hit'

    if 'CAPACITY' in msg_upper or 'EXHAUSTED' in msg_upper or 'EXCEEDED' in msg_upper:
        return 'Pool Capacity Limit'

    # Rule 6: nothing recognised. Do NOT invent a cause: XManager genuinely
    # returns an empty status message for some allocator rejections, and the
    # old code turned that silence into a confident-sounding
    # "Rejected by Allocator/Borg" for every PROD failure. Say so instead, and
    # point at the tool that can dig further.
    if msg and msg.strip() and msg.strip() != 'Failed' and 'Rejected' not in msg:
        return msg.strip()

    state = str(getattr(failed_wu, 'status_name', '') or '').lower()
    if 'fail' in state:
        return 'Failed, no reason reported (try why_probe)'
    return 'Queued, no reason reported (try why_probe)'


def _restart_budget_exhausted(failed_wu, tpu_info):
    """True when this job has burned the Borg restart budget it was given.

    `xm_launcher.py` submits with `--borg_max_task_failures` (default 10) and
    records the value in ~/.tpu_jobs.json. Past that limit Borg stops
    restarting the job, so the preemption is terminal -- worth distinguishing
    from a preemption that will still be retried.

    The observed failure count is read from the work unit; XManager exposes it
    under several names across versions, so try each and give up quietly (the
    caller then reports a plain preemption rather than guessing).
    """
    try:
        budget = int(tpu_info.get('max_task_failures'))
    except (TypeError, ValueError):
        return False
    if budget <= 0:
        return False

    used = _restart_count(failed_wu)
    return used is not None and used >= budget


def _restart_count(wu):
    """How many times Borg has restarted this work unit, or None if unknown.

    unified_infra keeps an explicit `restarts` counter on the job row
    (infra/models.py:100) because it relaunches jobs itself. Under Borg the
    restarts happen inside the work unit, so the count has to be read back off
    the work unit -- `restart_data` is the field XManager exposes for it.
    """
    data = getattr(wu, 'restart_data', None)
    if data is not None:
        for attr in ('num_restarts', 'restart_count', 'count', 'restarts'):
            value = getattr(data, attr, None)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass
        if isinstance(data, (list, tuple)):
            return len(data)

    for attr in ('task_failures', 'num_task_failures', 'failure_count',
                 'num_failures', 'restarts'):
        value = getattr(wu, attr, None)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


_STEP_DIR_RE = re.compile(r'^step_(\d+)(?:_.*)?$')


def _progress_step(tpu_info):
    """Highest COMPLETE checkpoint step this job has written, else 0.

    unified_infra derives the same number by grepping each attempt's
    `output.log` for `saved to ... step_N` (infra/resume.py:34). We enumerate
    the checkpoint bucket instead: it is the same source of truth the job's own
    auto-resume consults, and unlike a log it cannot rotate away. A directory
    without `extra.json` was still being written when the task died and does
    not count -- the same completeness rule unified_infra applies when it
    ignores a bare `Saving` with no matching `saved to`.

    Returns 0 when nothing has been saved yet, matching the "no step grepped
    means 0" convention.
    """
    bucket = (tpu_info.get('bucket_cp_path') or '').strip()
    if not bucket:
        return 0
    root = bucket.rstrip('/') + '/checkpoints'
    try:
        from etils import epath
        path = epath.Path(root)
        if not path.is_dir():
            return 0
        best = 0
        for child in path.iterdir():
            match = _STEP_DIR_RE.match(child.name)
            if not match:
                continue
            step = int(match.group(1))
            if step > best and (child / 'extra.json').exists():
                best = step
        return best
    except Exception:  # pylint: disable=broad-except
        # Never let a storage hiccup break the status board.
        return 0


def main(argv):
    import os
    import sys
    mapping_dir = os.path.expanduser('~/xm_job_to_bucket')

    if len(argv) > 1 and argv[1] == 'clear':
        _clear_jobs(argv[2:], mapping_dir)
        sys.exit(0)

    args_user = FLAGS.user

    c = xmanager_api.XManagerApi()
    console = Console(force_terminal=True, color_system="standard")
    
    table_running = Table(title="━━ running", show_header=True, header_style="bold green")
    table_pending = Table(title="━━ pending", show_header=True, header_style="bold yellow")
    table_error = Table(title="━━ error / failed", show_header=True, header_style="bold red")
    table_completed = Table(title="━━ completed", show_header=True, header_style="bold magenta")
    table_unknown = Table(title="━━ unknown", show_header=True, header_style="bold dim")
    
    tables = [table_running, table_pending, table_error, table_completed, table_unknown]
    
    for table in tables:
        table.add_column("ID", style="dim")
        table.add_column("STATUS")
        table.add_column("NAME", overflow="fold")
        # RESUME = Borg task restarts observed; STEP = highest complete
        # checkpoint written to the bucket (0 when nothing saved yet).
        table.add_column("RESUME", justify="right")
        table.add_column("STEP", justify="right")
        if table is table_running:
             table.add_column("DETAILS")
        if table is table_pending or table is table_error or table is table_unknown:
             table.add_column("WHY", overflow="fold")

    active_ids = set()
    try:
        if os.path.exists(mapping_dir):
            active_ids.update(os.listdir(mapping_dir))
    except Exception:
        pass

    tpu_jobs_map = {}
    tpu_jobs_json = os.path.expanduser("~/.tpu_jobs.json")
    if os.path.exists(tpu_jobs_json):
        try:
            with open(tpu_jobs_json, "r") as f:
                tpu_jobs_map = json.load(f)
                active_ids.update(tpu_jobs_map.keys())
        except Exception:
            pass

    sorted_active_ids = sorted(list(active_ids), key=lambda x: int(x) if x.isdigit() else x, reverse=True)

    console.print(f"Fetching {len(sorted_active_ids)} tracked experiments for user {args_user}...", style="dim")
    experiments = []
    for xid in sorted_active_ids:
        try:
            experiments.append(c.get_experiment(int(xid)))
        except Exception:
            pass

    for exp in experiments:
        exp_id = exp.id
        try:
            work_units = list(exp.get_work_units())
        except:
            work_units = []
            
        name = exp.name if exp.name else str(exp_id)
        job_info = tpu_jobs_map.get(str(exp_id), {})
        step_str = str(_progress_step(job_info))
        if not work_units:
            table_unknown.add_row(str(exp_id), "[dim]unknown[/dim]", name[:50],
                                  "-", step_str, "No WorkUnits (config error?)")
            continue

        resumes = _restart_count(work_units[0])
        resume_str = '-' if resumes is None else str(resumes)
            
        is_running = False
        is_pending = False
        is_error = False
        is_preempted = False
        
        for wu in work_units:
            state_str = wu.status_name.lower()
            wu_msg = ''
            if hasattr(wu, 'status') and hasattr(wu.status, 'message'):
                wu_msg = (wu.status.message or '').lower()
            if 'preempt' in wu_msg:
                is_preempted = True
            
            if "running" in state_str:
                is_running = True
            elif "pending" in state_str or "queued" in state_str or "preparing" in state_str or "created" in state_str or "uninitialized" in state_str or "scheduling" in state_str:
                is_pending = True
            elif "fail" in state_str or "error" in state_str or "cancel" in state_str:
                is_error = True

        # NOTE on ordering: a terminal state MUST win over `is_preempted`.
        # A preempted TPU job is not waiting in a queue -- Borg counts the torn
        # down gang as a task FAILURE and, with the default
        # max_task_failures=0, declares the job dead. Nothing re-queues it.
        # This branch used to read `elif is_preempted or is_pending:`, so every
        # job whose status message merely contained "preempt" was rendered as
        # PENDING even after it had failed, which made dead experiments look
        # like they were still queued for hours.
        if is_running:
            details = f"{len([w for w in work_units if 'running' in w.status_name.lower()])} active"
            table_running.add_row(str(exp_id), "[green]running[/green]", name[:50],
                                  resume_str, step_str, details)
        elif is_error:
            failed_wu = next((w for w in work_units if "fail" in w.status_name.lower() or "error" in w.status_name.lower() or "cancel" in w.status_name.lower()), work_units[0])
            state_str = failed_wu.status_name.lower()
            color = "red" if "cancel" not in state_str else "yellow"
            message = derive_failure_reason(exp_id, failed_wu, job_info)
            table_error.add_row(str(exp_id), f"[{color}]{state_str}[/{color}]", name[:50],
                                resume_str, step_str, message[:60])
        elif is_pending:
            failed_wu = next((w for w in work_units if "fail" in w.status_name.lower() or "error" in w.status_name.lower() or "cancel" in w.status_name.lower()), work_units[0])
            reason = derive_failure_reason(exp_id, failed_wu, job_info)
            if not reason or reason == 'No WorkUnits':
                reason = 'Pending'
            if is_preempted and 'preempt' not in reason.lower():
                # Genuinely still queued, but it has been preempted at least
                # once before -- worth surfacing.
                reason = f"{reason} (was preempted)"
            table_pending.add_row(str(exp_id), "[yellow]PENDING[/yellow]", name[:50],
                                  resume_str, step_str, reason)
        else:
            state_str = work_units[0].status_name.lower() if hasattr(work_units[0], 'status_name') else "completed"
            if "unknown" in state_str:
                message = getattr(work_units[0], 'status_message', '') or 'unknown backend state'
                table_unknown.add_row(str(exp_id), "[dim]unknown[/dim]", name[:50],
                                      resume_str, step_str, message[:60])
            else:
                table_completed.add_row(str(exp_id), f"[magenta]{state_str}[/magenta]", name[:50],
                                        resume_str, step_str)

    for table in tables:
        if table.row_count > 0:
            console.print(table)
            console.print()
    
if __name__ == '__main__':
    app.run(main)
