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

    # Rule 6: APPLICATION errors -- the job's own code broke, not the infra.
    #
    # This whole table exists to answer "is it me or is it Borg", and until now
    # it could only say the latter. Every rule above is an infra verdict, so an
    # application crash fell through to Rule 7, which printed the raw status
    # message -- and Borg prefixes those with the job name, so the WHY column
    # read `qiaos_group_275707651`, i.e. the job's own name as its cause. Four
    # consecutive EqR-jax code bugs (a missing wandb attribute, os.makedirs on
    # /cns, and two segfaults) were reported that way and each one needed
    # why_probe to find out what the table already knew.
    #
    # These are deliberately checked AFTER the infra rules: a preemption during
    # a crash loop is still a preemption, and infra causes are actionable in a
    # different way (retry, move cell, raise a limit order) from code bugs
    # (read the traceback, fix, resubmit).
    #
    # Marked "CODE BUG" so the verdict is unambiguous, with the concrete signal
    # in parentheses, because the next action differs per signal: OOM means
    # shrink the batch or ask for more RAM, SIGSEGV means read the stack.
    app_error = _application_error(msg_upper)
    if app_error:
        return app_error

    # Rule 7: nothing recognised. Do NOT invent a cause: XManager genuinely
    # returns an empty status message for some allocator rejections, and the
    # old code turned that silence into a confident-sounding
    # "Rejected by Allocator/Borg" for every PROD failure. Say so instead, and
    # point at the tool that can dig further.
    if msg and msg.strip() and msg.strip() != 'Failed' and 'Rejected' not in msg:
        return _strip_job_prefix(msg.strip())

    state = str(getattr(failed_wu, 'status_name', '') or '').lower()
    if 'fail' in state:
        return 'Failed, no reason reported (try why_probe)'
    return 'Queued, no reason reported (try why_probe)'


# Application-level failure signatures -> the verdict shown in the WHY column.
# Ordered most-specific first; the first substring hit wins.
#
# Sources: Borg surfaces the signal name and its own wording in
# `status.message` (go/xborg-why lists the task-level terminations), and the
# Python exception itself arrives there for an unhandled crash because the
# runtime writes it to stderr before the task exits.
_APPLICATION_ERROR_SIGNATURES = (
    # Fatal signals. 'SIGNAL 11' is how Borg words it ('Killed by signal 11!');
    # 'SEGMENTATION FAULT' is the human sentence it puts alongside.
    ('SEGMENTATION FAULT', 'CODE BUG: segfault (SIGSEGV)'),
    ('SIGNAL 11', 'CODE BUG: segfault (SIGSEGV)'),
    ('SIGSEGV', 'CODE BUG: segfault (SIGSEGV)'),
    ('SIGNAL 6', 'CODE BUG: abort (SIGABRT)'),
    ('SIGABRT', 'CODE BUG: abort (SIGABRT)'),
    ('SIGNAL 8', 'CODE BUG: arithmetic fault (SIGFPE)'),
    ('SIGNAL 4', 'CODE BUG: illegal instruction (SIGILL)'),
    ('SIGNAL 7', 'CODE BUG: bus error (SIGBUS)'),
    # Memory. Distinguished from a plain crash because the fix is different:
    # smaller batch / more RAM, not a code read.
    ('OUT OF MEMORY', 'CODE BUG: out of memory (raise RAM or shrink batch)'),
    ('OUT-OF-MEMORY', 'CODE BUG: out of memory (raise RAM or shrink batch)'),
    ('RESOURCE_EXHAUSTED: OOM', 'CODE BUG: out of memory (HBM)'),
    ('OOM_KILLED', 'CODE BUG: out of memory (OOM-killed)'),
    ('OOMKILLED', 'CODE BUG: out of memory (OOM-killed)'),
    ('MEMORY LIMIT', 'CODE BUG: exceeded memory limit'),
    # Python exceptions that reach the status message. PermissionError on /cns
    # gets its own verdict: it is nearly always stdlib file I/O against a path
    # that needs the epath helpers, not a real ACL problem.
    ("PERMISSION DENIED: '/CNS", 'CODE BUG: stdlib I/O on /cns (use epath helpers)'),
    ('PERMISSIONERROR', 'CODE BUG: PermissionError'),
    ('MODULENOTFOUNDERROR', 'CODE BUG: missing module (packaging)'),
    ('IMPORTERROR', 'CODE BUG: ImportError (packaging)'),
    ('ATTRIBUTEERROR', 'CODE BUG: AttributeError'),
    ('TYPEERROR', 'CODE BUG: TypeError'),
    ('VALUEERROR', 'CODE BUG: ValueError'),
    ('KEYERROR', 'CODE BUG: KeyError'),
    ('FILENOTFOUNDERROR', 'CODE BUG: FileNotFoundError'),
    ('ASSERTIONERROR', 'CODE BUG: AssertionError'),
    ('RUNTIMEERROR', 'CODE BUG: RuntimeError'),
    ('NOT_FOUND: COULD NOT FIND', 'CODE BUG: missing input file'),
    ('XLARUNTIMEERROR', 'CODE BUG: XLA runtime error'),
    ('JAXRUNTIMEERROR', 'CODE BUG: JAX runtime error'),
    ('TRACEBACK (MOST RECENT CALL LAST)', 'CODE BUG: unhandled Python exception'),
    # Borg's own phrasing for "your binary died on its own".
    ('APPLICATION LEVEL ERROR', 'CODE BUG: application-level failure'),
    ('UNRECOVERABLE FAILURE', 'CODE BUG: unrecoverable application failure'),
    ('EXITED WITH NON-ZERO', 'CODE BUG: non-zero exit'),
    ('NON-ZERO EXIT', 'CODE BUG: non-zero exit'),
)


# How long after submission a missing work unit is still considered normal.
# XManager creates the experiment record first and the work unit a moment later,
# so a just-submitted job legitimately has none.
_WORK_UNIT_GRACE_MINUTES = 15


def _experiment_age_minutes(exp, job_info):
  """Minutes since the experiment was created, or None if unknown.

  Prefers XManager's own creation timestamp and falls back to the submission
  time tpu_wrapper recorded in ~/.tpu_jobs.json, since the two are written by
  different systems and either may be absent.
  """
  import datetime
  created = getattr(exp, 'creation_time', None) or getattr(exp, 'create_time', None)
  if created is not None:
    try:
      now = datetime.datetime.now(datetime.timezone.utc)
      if created.tzinfo is None:
        created = created.replace(tzinfo=datetime.timezone.utc)
      return (now - created).total_seconds() / 60.0
    except Exception:  # noqa: BLE001
      pass
  # ~/.tpu_jobs.json keys the log dir by timestamp: eqr_run_YYMMDD_HHMMSS.
  logdir = str(job_info.get('logdir') or '')
  match = re.search(r'_(\d{6})_(\d{6})$', logdir)
  if match:
    try:
      stamp = datetime.datetime.strptime(match.group(1) + match.group(2), '%y%m%d%H%M%S')
      return (datetime.datetime.now() - stamp).total_seconds() / 60.0
    except ValueError:
      pass
  return None


# Placeholder verdicts that carry no information for a QUEUED job. Rendering
# them wastes the reader's attention on "the tool has nothing to say", which is
# the default state of a healthy queued job.
_UNINFORMATIVE_PENDING_REASONS = frozenset({
    'No WorkUnits',
    'Pending',
    'Queued, no reason reported (try why_probe)',
    'Failed, no reason reported (try why_probe)',
})


def _application_error(msg_upper):
    """Classify an application (not infra) failure, or return None.

    Kept separate from `classify_failure_reason` so the signature table stays
    readable and can be unit-tested directly.
    """
    for needle, verdict in _APPLICATION_ERROR_SIGNATURES:
        if needle in msg_upper:
            return verdict
    return None


def _strip_job_prefix(msg):
    """Drop Borg's leading `<job_name>/<user>: ` from a status message.

    Without this the WHY column shows the job's own name as its cause, which is
    both useless and actively misleading -- `qiaos_group_275707651` reads like
    an infra identifier rather than "we do not know".
    """
    head, sep, tail = msg.partition(': ')
    if sep and '/' in head and ' ' not in head and tail.strip():
        return tail.strip()
    return msg


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
        fetch_error = ''
        try:
            work_units = list(exp.get_work_units())
        except Exception as exc:  # noqa: BLE001 - one bad experiment must not
            # abort the whole table, but the reason must not be silently
            # rewritten into 'config error' either.
            work_units = []
            fetch_error = type(exc).__name__

        name = exp.name if exp.name else str(exp_id)
        job_info = tpu_jobs_map.get(str(exp_id), {})
        step_str = str(_progress_step(job_info))
        if not work_units:
            # Do NOT call this a config error. A freshly-submitted experiment has
            # no work units for the first minute or so -- XManager creates the
            # experiment record before the work unit exists -- and the fetch
            # above can also fail transiently on an RPC. Both looked identical
            # to a broken config, so two healthy PENDING jobs were reported as
            # 'No WorkUnits (config error?)' while they were simply queuing.
            # Age tells the two apart: minutes-old is normal, hours-old is not.
            age = _experiment_age_minutes(exp, job_info)
            if fetch_error:
                detail = f"work units unreadable ({fetch_error}); retry"
            elif age is not None and age < _WORK_UNIT_GRACE_MINUTES:
                detail = f"just submitted {int(age)}m ago, work unit not created yet"
            else:
                detail = "No WorkUnits (config error?)"
            table_unknown.add_row(str(exp_id), "[dim]unknown[/dim]", name[:50],
                                  "-", step_str, detail)
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
            # A queued job usually has nothing to explain: XManager leaves
            # status.message empty until something actually blocks it, and the
            # fallbacks then invent filler like 'Queued, no reason reported'
            # that occupies the eye without informing. Blank means "waiting,
            # nothing wrong"; only a REAL blocker (a GQM price cap, a quota
            # deficit, a prior preemption) earns text here.
            if not reason or reason in _UNINFORMATIVE_PENDING_REASONS:
                reason = ''
            if is_preempted and 'preempt' not in reason.lower():
                # Genuinely still queued, but it has been preempted at least
                # once before -- worth surfacing.
                reason = f"{reason} (was preempted)".strip()
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
