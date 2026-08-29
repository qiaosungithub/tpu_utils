"""The real builder: turn a placed job into a `tpu queue` invocation, and read its outcome.

This is the boundary between the new structure and the existing submission path. It keeps the
old argv shape (that part works) and fixes three things the old call site got wrong:

  1. ★THE OUTPUT IS CAPTURED AND RETURNED. The old daemon lane piped the builder through
     `sed`, so the exit status came from `sed` and the text went to a log nobody could find.
     One line spent twelve hours and nine failed builds unable to obtain a single sentence of
     error text. Here the tail is returned to the caller, which persists it on the record.
  2. ★EVERY INVOCATION IS BOUNDED. The rule was written in the daemon's own comments -- SIGTERM
     at 120s, SIGKILL 10s later -- and applied to one lane but not to the two that mattered;
     an unbounded one-shot then lived 2.45 hours holding a stale snapshot.
  3. ★A BUDGET REFUSAL IS NOT A BUILD FAILURE, and it is recognised by TWO markers, not one.
     The machine-readable `[[BUDGET_DEFERRED]]` line is authoritative, but rows predating it
     carry only the human sentence, and any future refusal path that forgets the marker would
     be silently misclassified as a defect. Recognising both is cheap; the cost of missing one
     is a counter that climbs forever.
"""

from __future__ import annotations

import os
import json
import re
import time
import shlex
import subprocess
from typing import Any, Optional

try:
  # Under blaze these resolve as package modules; a bare `python3 file.py` run
  # (used by the tests and by ad-hoc inspection) falls back to the flat names.
  from google3.experimental.users.qiaos.tpu_utils import jobchain as jc
  from google3.experimental.users.qiaos.tpu_utils import jobplace as jp
except ImportError:  # pragma: no cover - flat-layout fallback
  import jobchain as jc  # type: ignore
  import jobplace as jp  # type: ignore


_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')
_XID_RE = re.compile(
    r'(?:Created experiment|Launched experiment|registered XID'
    r'|work unit\(s\) in experiment)\D*(\d{6,})')
"""Phrasings that mean "a job was launched".

★This list is necessarily incomplete, and that is the point of `confirm_xid_from_registry`
below. The canary launched XID 284525515 and the scheduler recorded "build produced no XID",
because `tpu queue` says "Launched experiment" while the regex only knew "Created
experiment". The failure is silent AND expensive: the dispatcher would have retried, built
again, launched a SECOND real job, and failed to recognise that one too.
★A regex describes what this kind of output usually looks like; it cannot describe what
counts as a launch. So it is the fast path, never the only one.
"""

_REGISTRY_PATH = os.path.expanduser('~/.tpu_jobs.json')
_BUDGET_MARKER = '[[BUDGET_DEFERRED]]'
_BUDGET_SENTENCE = re.compile(r'ERROR: Budget exceeded', re.I)

DRY_RUN_MARKER = '[[DRY_RUN]]'
"""Emitted instead of an XID when nothing was executed. The dispatcher must treat it as
"nothing happened", never as a failure: a dry run is an observation, and an observation that
mutates the thing observed is not one."""

WRAPPER_PATH = os.path.expanduser('~/work/tpu_cmd/tpu_wrapper.sh')
"""Where the `tpu` shell function lives. Sourced per build rather than kept warm: a
long-lived shell would hold whatever the file said when it started, and this file is edited
by several lines."""

BUILD_TIMEOUT_S = 900
"""Bounded, always. 900s rather than the daemon's 120s because this call includes a real
package + submit, but the principle is the daemon's own: an unbounded checker does not fail,
it accumulates."""


def build_argv(job: jc.Job, placement: jp.Placement, group: str = '9') -> list[str]:
  """The `tpu queue` argv. Pure and inspectable -- print it before running it.

  launch_kwargs pass through VERBATIM: None/True become bare flags, False drops the flag,
  everything else becomes --k=v. Rebuilding this dict from scratch instead of carrying it is
  what cost 17 jobs their checkpoints and let them hold budget at a tenth of the real price.
  """
  argv = ['tpu', 'queue',
          f'--tpu_type={job.power}',
          f'--group={group}',
          f'--cell={placement.cell}']
  if job.tier:
    argv.append(f'--tier={job.tier}')
  for k, v in (job.launch_kwargs or {}).items():
    flag = k if k.startswith('--') else f'--{k}'
    if v is None or v is True:
      argv.append(flag)
    elif v is False:
      continue
    else:
      argv.append(f'{flag}={v}')
  return argv


def extract_xid(output: str) -> Optional[str]:
  """The XID from `tpu queue` output, ANSI-stripped."""
  m = _XID_RE.search(_ANSI_RE.sub('', output or ''))
  return m.group(1) if m else None


def is_budget_refusal(output: str) -> bool:
  """True if this no-XID outcome was the budget gate, not a defect of the job.

  ★Two markers. The bare marker line is what budget_check prints today; the English sentence
  is what older records carry and what any refusal path that forgot the marker would emit.
  Classifying a refusal as a defect is how rows reached att=65 -- and, worse, how a job three
  rounds from success (income moved 25x in a night) was parked where only a human could free
  it.
  """
  clean = _ANSI_RE.sub('', output or '')
  if any(line.strip() == _BUDGET_MARKER for line in clean.splitlines()):
    return True
  return bool(_BUDGET_SENTENCE.search(clean))


def confirm_xid_from_registry(exp_name: str, since: float) -> Optional[str]:
  """Ask the ARTEFACT, not the transcript: did a launch register itself just now?

  ★The launcher writes every successful launch into ~/.tpu_jobs.json. Reading that back is
  immune to phrasing changes in `tpu queue`'s output, so it catches the case the regex misses
  -- which matters because missing a launch does not merely lose information, it makes the
  dispatcher launch again.

  Matches on exp_name and requires the entry to be newer than `since` (the moment this build
  started), so an older run of the same experiment cannot be mistaken for this one.
  """
  try:
    with open(_REGISTRY_PATH) as fh:
      reg = json.load(fh)
  except (OSError, ValueError):
    return None
  best, best_t = None, since
  for xid, rec in (reg.items() if isinstance(reg, dict) else []):
    if not isinstance(rec, dict) or rec.get('exp_name') != exp_name:
      continue
    t = rec.get('submitted_at') or rec.get('created_at') or 0
    try:
      t = float(t)
    except (TypeError, ValueError):
      continue
    if t >= best_t:
      best, best_t = str(xid), t
  return best


def tail(s: str, n: int = 2000) -> str:
  """Keep the END of the output: the error is at the bottom, the banner is at the top."""
  return (s or '').strip()[-n:]


class TpuQueueBuilder:
  """Runs `tpu queue` for one job. Callable, so it drops straight into Dispatcher."""

  def __init__(self, group: str = '9', timeout_s: int = BUILD_TIMEOUT_S,
               dry_run: bool = False, log=None,
               runner=None):
    self.group = group
    self.timeout_s = timeout_s
    self.dry_run = dry_run
    self.log = log or (lambda s: None)
    self._runner = runner or self._run

  def __call__(self, job: jc.Job, placement: jp.Placement,
               env: dict[str, str]) -> tuple[Optional[str], str]:
    argv = build_argv(job, placement, self.group)
    cwd = job.workdir or None
    self.log(f'[build] {job.job_id}: {" ".join(argv)} (cwd={cwd}) env={sorted(env)}')
    if self.dry_run:
      # ★DRY_RUN_MARKER, not a bare "no XID". A dry run produces no XID for the same reason a
      # crashed build does, and the dispatcher classified that as a job-intrinsic failure:
      # three observation rounds pushed a job that had never been built to HELD with
      # attempts=3. Observing a system must not change it -- and the damage here outlived the
      # observation, since HELD needs a human to clear.
      return None, DRY_RUN_MARKER

    started_at = time.time()
    full_env = dict(os.environ)
    full_env.update(env)
    out, rc, timed_out = self._runner(argv, cwd, full_env, self.timeout_s)

    if timed_out:
      # ★A timeout is the ENVIRONMENT, not a defect: the host was too slow or wedged. Reported
      # as a no-XID with an explicit tail so the caller can classify it, rather than being
      # swallowed into a generic failure.
      return None, tail(f'TIMED OUT after {self.timeout_s}s (SIGTERM+SIGKILL). {out}')

    xid = extract_xid(out)
    if not xid:
      # ★The regex missed it, or the phrasing changed. Before calling this a failed build --
      # which leads to a retry and a second real job -- ask the registry whether a launch
      # actually happened.
      exp = str((job.launch_kwargs or {}).get('exp_name') or '')
      if exp:
        xid = confirm_xid_from_registry(exp, started_at)
        if xid:
          self.log(f'[build] {job.job_id}: output did not name an XID, but the registry '
                   f'records {xid} for {exp} -- treating as LAUNCHED. The phrasing list in '
                   f'_XID_RE is missing a case; add it.')
    if xid:
      return xid, tail(out)
    if is_budget_refusal(out):
      return None, tail(f'{_BUDGET_MARKER} {out}')
    # ★rc is reported but NOT used to decide: the submit path returns 1 for a budget refusal
    # too, so an exit code alone cannot separate "not allowed now" from "this job is broken".
    return None, tail(f'no XID (rc={rc}). {out}')

  @staticmethod
  def _run(argv, cwd, env, timeout_s) -> tuple[str, int, bool]:
    # ★`tpu` is a SHELL FUNCTION defined in tpu_wrapper.sh, not an executable -- running
    # argv directly gives `[Errno 2] No such file or directory: 'tpu'`. It has to be sourced
    # into the shell that then calls it. Substituting an absolute path would not work either:
    # there is no binary to point at.
    #
    # ★cwd is what `tpu queue`'s rsync packages, so it must be the job's own checkout. An
    # absent cwd is refused up front rather than silently packaging whatever directory the
    # process happened to inherit -- that is how a job ships the wrong source.
    inner = ' '.join(shlex.quote(a) for a in argv)
    script = f'source {shlex.quote(WRAPPER_PATH)} >/dev/null 2>&1; {inner}'
    if cwd is not None and not os.path.isdir(cwd):
      return f'refusing to build: workdir does not exist: {cwd}', -1, False
    try:
      p = subprocess.run(['bash', '-c', script], cwd=cwd, env=env, capture_output=True,
                         text=True, timeout=timeout_s)
      return (p.stdout or '') + (p.stderr or ''), p.returncode, False
    except subprocess.TimeoutExpired as e:
      partial = (e.stdout or b'').decode(errors='replace') if isinstance(e.stdout, bytes) \
          else (e.stdout or '')
      return partial, -1, True


def staged_identity(job: jc.Job) -> tuple[str, str]:
  """Read (target_label, project_name) from the job's staged config.sh, for the identity gate.

  ★This reads the STAGEDIR copy on purpose -- the one that gets backfilled from a global
  default when a ghost write drops it, and which therefore may belong to whoever staged last.
  Comparing it against the fingerprint recorded at enqueue is the only thing that catches a
  job about to run another line's target with every structural check green.
  """
  stagedir = (job.launch_kwargs or {}).get('stagedir') or job.workdir
  cfg = os.path.join(stagedir or '', 'config.sh')
  if not os.path.isfile(cfg):
    raise OSError(f'no config.sh at {cfg}')
  text = open(cfg, errors='ignore').read()
  tl = re.search(r'^export TARGET_LABEL="([^"]*)"', text, re.M)
  pn = re.search(r'^export PROJECT_NAME="([^"]*)"', text, re.M)
  return (tl.group(1) if tl else jc.UNKNOWN, pn.group(1) if pn else jc.UNKNOWN)
