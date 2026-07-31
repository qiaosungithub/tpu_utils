import argparse
import concurrent.futures
import datetime
import json
import os
import re
import subprocess
import sys
from absl import app
from absl import flags
from google3.experimental.users.qiaos.tpu_utils import group_utils
from google3.learning.deepmind.xmanager2.client import xmanager_api
from rich.console import Console
from rich.table import Table
from rich.text import Text
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

    # Rules 1-2: preemption. ALWAYS name the cause -- "Preempted" alone does not
    # say whether to resubmit as-is (defrag/higher-priority: nothing you can do,
    # just resume), to raise a limit order (price), or to stop resuming
    # (restart budget spent). `_preemption_cause` maps Borg/xborg wording onto
    # that decision.
    if 'PREEMPTED' in msg_upper or 'PREEMPT' in msg_upper:
        return _preemption_verdict(msg_upper, failed_wu, tpu_info)

    # DESCHEDULED == preempted. xborg does not use the word "preempt" when it
    # takes resources back; it says the workload was "descheduled". Losing
    # opportunistically-held capacity to an allotment with a guarantee is
    # exactly a preemption from the job's point of view, and reporting it as
    # anything softer hides real preemption pressure.
    # go/xborg-why-descheduled#resource-guarantee-reclaim
    if 'DESCHEDULED' in msg_upper:
        return _preemption_verdict(msg_upper, failed_wu, tpu_info)

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
        # A resumed experiment accumulates every attempt's text in
        # status.message, so an old traceback outlives the bug that caused it.
        # XID 275793223 kept reporting `CODE BUG: ValueError` from attempt 1
        # while attempts 4 and 5 were training fine and merely being preempted.
        # Picking the newest work unit was not enough -- the message itself is
        # the concatenation.
        #
        # Progress is the tie-breaker Borg cannot fake: if the job checkpointed
        # PAST the step it was at when that traceback was written, the crash is
        # historical. Say so instead of pinning a stale cause that sends the
        # reader to debug already-fixed code.
        if _resumed_past_error(tpu_info):
            return f'{app_error} (STALE: earlier attempt; job has since progressed)'
        return app_error

    # Rule 7: nothing recognised. Do NOT invent a cause: XManager genuinely
    # returns an empty status message for some allocator rejections, and the
    # old code turned that silence into a confident-sounding
    # "Rejected by Allocator/Borg" for every PROD failure. Say so instead, and
    # point at the tool that can dig further.
    if msg and msg.strip() and msg.strip() != 'Failed' and 'Rejected' not in msg:
        return _humanize(_strip_job_prefix(msg.strip()))

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


# XManager/xborg phrasings that are accurate but unreadable at a glance, mapped
# onto what the reader actually needs to decide: is this MY problem, and is
# there anything to do? Each entry is (compiled regex, template). The template
# may reference regex groups, so a percentage in the original survives into the
# translation.
#
# The originals are kept verbatim in the tooltip-ish tail where they are short
# enough; the point is that the FIRST words say what is going on.
_HUMANIZE_RULES = (
    # "86% of SCUs in your pool (98% within your allotment) are ..."
    # An SCU is a scheduling unit; the sentence means the pool is full and this
    # job is waiting its turn. Nothing is broken and there is nothing to fix.
    (re.compile(r'(\d+)%\s+of\s+SCUs\s+in\s+your\s+pool.*?\((\d+)%\s+within\s+your\s+allot',
                re.I | re.S),
     'Waiting for chips: your group is {1}% full (pool {0}%). '
     'Normal queueing -- it starts when a running job frees a slice.'),
    (re.compile(r'SCUs?\s+in\s+your\s+pool', re.I),
     'Waiting for chips: the pool is busy. Normal queueing, no action needed.'),
    (re.compile(r'GQM_RESOURCE_DEFICIT', re.I),
     'Waiting for chips: the 30s GQM auction did not clear enough for this job yet.'),
    (re.compile(r'no\s+resources?\s+available|insufficient\s+capacity', re.I),
     'Waiting for chips: no free slice of this shape in the cell right now.'),
    (re.compile(r'work\s+unit\s+not\s+created\s+yet', re.I),
     'Just submitted; XManager has not created the work unit yet.'),
)


def _humanize(msg):
    """Rewrite an XManager status line into something readable at a glance.

    `tpu check` is scanned, not read. A line like "86% of SCUs in your pool (98%
    within your allotment) are ..." is precise and almost useless in that mode:
    it does not say whether the job is broken, whose fault it is, or whether to
    act. Unmatched messages pass through unchanged -- a wrong translation is
    worse than an opaque original.
    """
    if not msg:
        return msg
    for pattern, template in _HUMANIZE_RULES:
        m = pattern.search(msg)
        if m:
            try:
                return template.format(*m.groups())
            except (IndexError, KeyError):
                return template
    return msg


# Placeholder verdicts that carry no information for a QUEUED job. Rendering
# them wastes the reader's attention on "the tool has nothing to say", which is
# the default state of a healthy queued job.
_UNINFORMATIVE_PENDING_REASONS = frozenset({
    'No WorkUnits',
    'Pending',
    'Queued, no reason reported (try why_probe)',
    'Failed, no reason reported (try why_probe)',
})


# How many trailing log lines to show under a running job, and how wide.
_LOG_TAIL_LINES = 2
_LOG_TAIL_WIDTH = 150

# Lines that say nothing about progress. tqdm repaints a bar hundreds of times a
# second, so without this the tail is always the same spinner frame.
_LOG_TAIL_SKIP = (
    'log-mirror', 'coordination flags', 'RuntimeWarning', 'warnings.warn',
    'dtype = _resolve_stablemax', 'WARNING:', 'DeprecationWarning',
)


# CNS round-trips dominate the tail, so they are issued concurrently. One pool
# for the process: the client releases the GIL, and rebuilding a pool per call
# would cost more than the reads. Sized for (jobs x ranks) in flight at once.
_CNS_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=16)

# How long the whole tail-fetching phase may take before we render without it.
_LOG_TAIL_BUDGET_SEC = 8.0


def _read_log_tail(bucket: str, nbytes: int = 16384) -> str:
  """Raw tail bytes of the most active rank log under `bucket`, or ''.

  Rank 0 is not always the talkative one -- under pmap the process that owns the
  progress bar can be any rank -- so pick whichever rank log is LARGEST, which
  needs one stat() per rank.

  Two things here are deliberate, both measured against a live 1 MB job log:

  * the per-rank stat() calls go through `_CNS_POOL` instead of running inside a
    `sorted(key=...)`. That key function made them strictly serial, and with
    4 ranks it was the single biggest cost in the tail (1271 ms -> 656 ms for
    two jobs once parallelised).
  * the read SEEKS to the tail rather than doing `read_bytes()[-16384:]`, which
    downloaded the entire file and then threw away all but the last 16 KB. At
    1 MB that is only ~1.4x -- small reads are dominated by the RPC round trip,
    not by bytes -- but the old form grew without bound as the run went on,
    which is exactly the regime a 100k-step job ends up in.

  Never raises: a status table that dies because a log was unreadable is worse
  than one with no tail.
  """
  try:
    from etils import epath
    logdir = epath.Path(bucket) / 'logs'
    entries = [p for p in logdir.iterdir() if p.name.startswith('rank_')]
    if not entries:
      return ''

    def _size(path):
      try:
        return path.stat().length
      except Exception:  # noqa: BLE001 - a vanished rank file is not fatal
        return -1

    sizes = list(_CNS_POOL.map(_size, entries))
    best = max(zip(sizes, range(len(entries))), key=lambda t: t[0])
    if best[0] <= 0:
      return ''
    with entries[best[1]].open('rb') as handle:
      try:
        handle.seek(-nbytes, os.SEEK_END)
      except OSError:
        pass  # file shorter than the window; read it whole
      return handle.read().decode('utf-8', errors='replace')
  except Exception:  # noqa: BLE001 - the tail is a nicety, never a hard failure
    return ''


def _fetch_log_tails(buckets: list[str]) -> dict[str, str]:
  """Tail every bucket at once. Missing/slow entries simply come back absent.

  Fetching the whole table's tails concurrently is what keeps `tpu check`
  interactive: the cost becomes that of the slowest single job rather than the
  sum over jobs. The budget is a hard ceiling -- a wedged CNS cell must not be
  able to hang the status table, so whatever has not arrived is dropped.
  """
  wanted = [b for b in dict.fromkeys(buckets) if b]
  if not wanted:
    return {}
  futures = {b: _CNS_POOL.submit(_read_log_tail, b) for b in wanted}
  out: dict[str, str] = {}
  for bucket, fut in futures.items():
    try:
      raw = fut.result(timeout=_LOG_TAIL_BUDGET_SEC)
    except Exception:  # noqa: BLE001 - timeout or read error: render without it
      continue
    if raw:
      out[bucket] = raw
  return out


def _log_tail(job_info, lines: int = _LOG_TAIL_LINES, cache=None) -> list[str]:
  """The last few meaningful log lines of a running job, newest last.

  Reads the rank log the application mirrors to the checkpoint bucket
  (utils/logging_util.py::mirror_logs_to_bucket). `cache` is the prefetched
  `{bucket: raw_tail}` from `_fetch_log_tails`; without it this falls back to
  fetching synchronously, which is correct but serial.
  """
  bucket = (job_info.get('bucket_cp_path') or '').strip()
  if not bucket:
    return []
  raw = cache.get(bucket) if cache is not None else _read_log_tail(bucket)
  if not raw:
    return []

  out: list[str] = []
  # tqdm uses \r to repaint in place, so split on it too or the whole bar is
  # one enormous "line".
  for chunk in raw.replace('\r', '\n').splitlines():
    line = chunk.strip()
    if not line or any(skip in line for skip in _LOG_TAIL_SKIP):
      continue
    line = re.sub(r'\x1b\[[0-9;?]*[A-Za-z]', '', line).strip()
    if not line:
      continue
    if len(line) > _LOG_TAIL_WIDTH:
      line = line[:_LOG_TAIL_WIDTH - 1] + '…'
    out.append(line)
  return out[-lines:]


def _resumed_past_error(tpu_info):
    """True when the run kept making progress after the traceback was written.

    The signal is the checkpoint directory: a job that crashed and stayed dead
    cannot write a checkpoint NEWER than the crash. Comparing the newest
    checkpoint's mtime with the launch log's mtime (the launch that carried the
    failing attempt) separates "crashed and dead" from "crashed once, then ran
    on for another 50k steps".

    Deliberately conservative: anything unreadable answers False, so a genuine
    crash is never softened into a stale-looking one.
    """
    bucket = (tpu_info or {}).get('bucket_cp_path') or ''
    if not bucket:
        return False
    try:
        out = subprocess.run(
            ['fileutil', 'ls', '-l', f'{bucket}/checkpoints'],
            capture_output=True, text=True, timeout=25,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return False
        # More than one checkpoint means the run survived long enough to write
        # again. A crash-on-startup loop produces zero or one.
        steps = re.findall(r'step_(\d+)', out.stdout)
        return len(set(steps)) > 1
    except Exception:  # noqa: BLE001 - diagnosis must not raise
        return False


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


# Preemption causes, most specific first. Each entry is
# (substrings, cause phrase). The phrase completes the sentence
# "Preempted. Due to <phrase>" and should say what to DO, not just what Borg
# called it -- the whole point of the column is to choose the next action.
_PREEMPTION_CAUSES = (
    (('SLICE_DEFRAGMENTATION', 'DEFRAGMENTATION', 'DEFRAG'),
     'slice defrag (Borg repacking the pod; resume, same cell is fine)'),
    (('HIGHER_PRIORITY', 'HIGHER PRIORITY', 'PRIORITY'),
     'a higher-priority job taking the chips (resume; PROD already is prio 200)'),
    (('RECLAIM', 'GUARANTEED CAPACITY', 'RESOURCE_GUARANTEE'),
     'guarantee reclaim -- we were ABOVE floor, holding chips opportunistically'),
    (('LIMIT_ORDER', 'LIMIT ORDER', 'PAUSED_BY_LIMIT_ORDER'),
     'a GQM limit order: market price rose above the cap, so the job was paused'),
    (('GQM_RESOURCE_DEFICIT', 'RESOURCE_DEFICIT'),
     'a GQM resource deficit -- the auction did not clear enough chips this cycle'),
    (('EVICT', 'EVICTION'), 'machine eviction (drain/repair)'),
    (('MAINTENANCE', 'DRAIN'), 'scheduled machine maintenance'),
    (('OUT_OF_CAPACITY', 'NO_CAPACITY', 'CAPACITY'),
     'the cell running out of capacity for this slice shape'),
)


def _preemption_verdict(msg_upper, failed_wu, tpu_info):
    """'Preempted. Due to <cause>' plus whether resuming is still allowed.

    A bare 'Preempted' tells you the job stopped but not what to do next, and
    the four cases want four different actions: defrag and higher-priority just
    need a resume, a guarantee reclaim means we were running above floor, a
    limit order means the price moved and the cap has to be raised before
    anything will schedule, and an exhausted restart budget means resuming is
    pointless until the budget is raised.
    """
    cause = None
    for needles, phrase in _PREEMPTION_CAUSES:
        if any(n in msg_upper for n in needles):
            cause = phrase
            break

    if cause is None:
        # Borg did not say. Do not guess a cause -- say that it did not, so the
        # reader knows to look rather than trusting a fabricated reason.
        verdict = 'Preempted. Cause not reported by Borg (try why_probe)'
    else:
        verdict = f'Preempted. Due to {cause}'

    if _restart_budget_exhausted(failed_wu, tpu_info):
        verdict += ' [restart budget SPENT -- resume will not be retried]'
    return verdict


def _latest_failed_wu(work_units):
    """The MOST RECENT failed/cancelled work unit, else the most recent overall.

    `--resume_xid` appends a work unit to the SAME experiment, so a long run
    that has been preempted twice has three: WU 1 (crashed), WU 2 (crashed),
    WU 3 (the live one). Picking the FIRST match -- which is what this used to
    do -- pins the verdict to the oldest failure forever. XID 275793223 was
    reported as `CODE BUG: ValueError` long after that bug was fixed, because
    WU 1 still carried the original traceback while WU 3 had merely been
    preempted. A stale cause is worse than none: it sends you to debug code
    that is already correct.

    Ordering is by work-unit id, which XManager assigns monotonically, with
    `creation_time` as the fallback and list order as the last resort.
    """
    if not work_units:
        return None

    def _order(wu):
        wid = getattr(wu, 'id', None)
        if isinstance(wid, int):
            return (2, wid)
        created = getattr(wu, 'creation_time', None)
        if created is not None:
            try:
                return (1, created.timestamp())
            except Exception:  # noqa: BLE001 - not a datetime
                pass
        return (0, 0)

    def _is_bad(wu):
        state = str(getattr(wu, 'status_name', '') or '').lower()
        return 'fail' in state or 'error' in state or 'cancel' in state

    bad = [w for w in work_units if _is_bad(w)]
    return max(bad or work_units, key=_order)


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
    # Pin the width. Piped into the cache file rich cannot detect a terminal and
    # falls back to 80 columns, which CLIPS the table -- and a clipped row loses
    # its trailing box character, which is exactly what tpu_wrapper's cache
    # parser keys on. Every job then fell back to "SUBMITTED".
    # $COLUMNS wins when a human is looking at a real terminal.
    _width = int(os.environ.get("TPU_CHECK_WIDTH") or os.environ.get("COLUMNS") or 0)
    if _width < 80:
        _width = 160
    # `_environ={}` is load-bearing: rich re-reads $COLUMNS from the process
    # environment and lets it OVERRIDE an explicit width=, so passing width
    # alone silently kept the 80-column default. Hiding the environment from
    # this Console is what makes the width stick.
    console = Console(force_terminal=True, color_system="standard",
                      width=_width, _environ={})
    
    table_running = Table(title="━━ running", show_header=True, header_style="bold green")
    table_pending = Table(title="━━ pending", show_header=True, header_style="bold yellow")
    table_error = Table(title="━━ error / failed", show_header=True, header_style="bold red")
    table_completed = Table(title="━━ completed", show_header=True, header_style="bold magenta")
    table_unknown = Table(title="━━ unknown", show_header=True, header_style="bold dim")
    
    tables = [table_running, table_pending, table_error, table_completed, table_unknown]
    
    for table in tables:
        table.add_column("ID", style="dim")
        table.add_column("STATUS")
        # Wide on purpose: for the running table this column also carries the
        # log tail, and a 25-char tail is not worth printing.
        table.add_column("NAME", overflow="fold", min_width=60)
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

    # Start every log tail NOW, before the per-experiment loop, so the CNS round
    # trips overlap each other AND the XManager work-unit fetches below. Tailing
    # inside the loop made the cost the SUM over jobs; here it is the max.
    # Buckets are cheap to over-request -- a job that turns out not to be running
    # just leaves an unused entry in the dict.
    log_tails = _fetch_log_tails(
        [(tpu_jobs_map.get(str(exp.id), {}) or {}).get('bucket_cp_path', '')
         for exp in experiments])

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
            # Second row per run: what the job is actually SAYING. A status of
            # "running" tells you Borg is happy, not that training is
            # progressing -- a job wedged in a collective looks identical here.
            # Same idea as unified_infra's dim indented continuation lines.
            for line in _log_tail(job_info, cache=log_tails):
                # `Text` + no_wrap: a log tail that soft-wraps inside the NAME
                # column is unreadable, and truncation is the right trade -- the
                # point is a glanceable "is it moving", not the full line.
                table_running.add_row(
                    "", "",
                    Text(f"  │ {line}", style="dim", no_wrap=True, overflow="ellipsis"),
                    "", "", "")
        elif is_error:
            failed_wu = _latest_failed_wu(work_units)
            state_str = failed_wu.status_name.lower()
            color = "red" if "cancel" not in state_str else "yellow"
            message = derive_failure_reason(exp_id, failed_wu, job_info)
            table_error.add_row(str(exp_id), f"[{color}]{state_str}[/{color}]", name[:50],
                                resume_str, step_str, message[:160])
        elif is_pending:
            failed_wu = _latest_failed_wu(work_units)
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
                                      resume_str, step_str, message[:160])
            else:
                table_completed.add_row(str(exp_id), f"[magenta]{state_str}[/magenta]", name[:50],
                                        resume_str, step_str)

    for table in tables:
        if table.row_count > 0:
            console.print(table)
            console.print()
    
if __name__ == '__main__':
    app.run(main)
