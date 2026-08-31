"""One job, one chain: the scheduler's job record.

Operator-authorised full rewrite (2026-08-28). No legacy compatibility layer.

WHY THIS EXISTS. The previous record was a flat row that every pass rewrote wholesale, so
a writer holding a 20-minute-old snapshot could legally un-do anything: a RUNNING job lost
its xid and was rebuilt (5x 8-card B200 on one row), a FAILED terminal flipped back to
claimable within 30s, a deleted row was resurrected from a 9-minute-old snapshot, and an
attempts counter went *down*. None of that was a race on the lock -- the lock was correct
throughout. It was a race on the SNAPSHOT.

THE SHAPE. A job is an append-only chain: immutable identity, monotonic counters, and a list
of attempt nodes. A resume is not a new row; it is the same chain continuing. Nothing that
happened can be un-happened by a later writer, because:
  - every write is a field-level patch to named jobs, never a whole-file replace (I2);
  - every write carries the base `version` it read and is REFUSED if the record moved (I1);
  - the invariants below are checked on the way in, so a stale or buggy caller is rejected
    rather than silently believed.

READ THE INVARIANT TABLE BEFORE CHANGING ANYTHING HERE. Each one is a fix for a measured
incident, not a hypothetical; `INVARIANTS` carries the evidence inline.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import time
from typing import Any, Iterable, Optional

try:
  from google3.experimental.users.qiaos.tpu_utils import cell_locality as _cell_locality
except ImportError:  # pragma: no cover - flat-layout fallback for ad-hoc runs
  try:
    import cell_locality as _cell_locality  # type: ignore
  except ImportError:
    _cell_locality = None  # type: ignore


# --- states ----------------------------------------------------------------
class JobState(str, enum.Enum):
  """Lifecycle states. TERMINAL states are one-way (I4).

  There is no DONE-that-nothing-writes here: the previous enum had a `DONE` member with no
  write site anywhere in the codebase, so "succeeded" and "failed" were merged into one
  boolean at the probe layer and the distinction was destroyed before any decision saw it.
  """

  QUEUED = 'QUEUED'                    # eligible for dispatch
  BUILD_REQUESTED = 'BUILD_REQUESTED'  # passed the gates, awaiting the serial builder
  BUILDING = 'BUILDING'                # a builder holds it
  SUBMITTED = 'SUBMITTED'              # has an xid, not yet observed running
  RUNNING = 'RUNNING'                  # observed running
  DEFERRED = 'DEFERRED'                # environment said "not now" -- auto-retried, NO strike
  HELD = 'HELD'                        # job-intrinsic defect; needs a human
  COMPLETED = 'COMPLETED'              # terminal, success
  FAILED = 'FAILED'                    # terminal, failure
  CANCELLED = 'CANCELLED'              # terminal, withdrawn


TERMINAL = frozenset({JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED})
CLAIMABLE = frozenset({JobState.QUEUED, JobState.BUILD_REQUESTED})
LIVE_XID_STATES = frozenset({JobState.SUBMITTED, JobState.RUNNING})


class Outcome(str, enum.Enum):
  """How one attempt ended. Kept distinct from JobState so that a probe can never collapse
  success and failure into a single 'terminal' boolean (that collapse mislabelled a real
  COMPLETED run as FAILED)."""

  IN_FLIGHT = 'IN_FLIGHT'   # ★still running; the attempt has no outcome YET
  COMPLETED = 'COMPLETED'
  FAILED = 'FAILED'
  PREEMPTED = 'PREEMPTED'
  CANCELLED = 'CANCELLED'
  ABANDONED = 'ABANDONED'   # we stopped tracking it (purged from XM, unknowable)


class FailureClass(str, enum.Enum):
  """★The single most important distinction in this file (I9).

  JOB_DEFECT is the only class that may increment `build_attempts`. Everything else is the
  fleet telling us "not now", and counting it as a strike is what drove entries to att=65
  and parked jobs in a state only a human could leave -- while income swung 25x in one night,
  i.e. while the refusal was demonstrably transient.
  """

  JOB_DEFECT = 'JOB_DEFECT'      # bad BUILD, missing entry point, package error
  ENVIRONMENT = 'ENVIRONMENT'    # budget, quota, capacity, cluster fault, host lock, srcfs


UNKNOWN_TIME = -1.0
"""Timestamp sentinel: "this happened, but we do not know when".

★NOT 0.0, which is what these fields used to default to. Zero is the worst possible choice:
`is None` says it was set, a truthiness test says it was not, and any date formatter prints
**1970-01-01** -- a perfectly ordinary-looking timestamp that nobody re-reads. Two readers
scanning the same store for inconsistencies got 1 hit and 0 hits, and the monitor's own
formatter rendered the 0.0 as the string "None", so the evidence in the report was wrong
while its conclusion was right.

An ABANDONED default at least uses a startling word. A zero timestamp turns into a date.
"""

UNKNOWN = '<UNKNOWN>'
"""Explicit sentinel. Never '' and never a plausible real value: tonight `headroom=0` meant
both 'genuinely full' and 'could not read', and a cell name doubling as its own metro made
26 of 57 lookups silently wrong."""

PERSONAL_ONLY_METROS = frozenset({'phx', 'ske'})
"""Metros where the GROUP has no storage registration; mirrors
`xm_launcher.py:_PERSONAL_ONLY_METROS`. A literal copy ON PURPOSE: importing the
launcher would drag xmanager into every enqueue. Safe to duplicate because this
is a REFUSAL list -- drifting by GAINING a metro only refuses more, and drifting
by LOSING one is still caught by the launcher's identical gate downstream.
Verify with: `grep -A4 _PERSONAL_ONLY_METROS ~/work/tpu_cmd/xm_launcher.py`"""


# --- the chain -------------------------------------------------------------
@dataclasses.dataclass
class Node:
  """One past attempt. Appended, never rewritten (I8).

  ★`ckpt_path` is stored VERBATIM as the job reported it, and is never parsed, normalised,
  or completed by the scheduler. There are four incompatible checkpoint shapes in this fleet
  -- `step_<N>/state/` (EqR-jax), flat `step_<N>/` (codi, coconut), `checkpoint_<N>` files
  (paligemma, jax_llava), and `step_<N>.pt` -- a single FILE, not a directory (torch ports).
  Any clever path arithmetic breaks at least one family; appending `/state` once produced a
  FileNotFoundError *after* printing a reassuring metadata warning.

  An attempt that saved nothing still becomes a node, with ckpt_path=None. unified_infra
  discards those; we keep them, because the recurring failure tonight was precisely that the
  absence of evidence left no trace and so could not be reasoned about afterwards.
  """

  xid: str
  cell: str = UNKNOWN
  metro: str = UNKNOWN
  continent: str = UNKNOWN
  bucket: str = UNKNOWN
  ckpt_path: Optional[str] = None     # verbatim; None = this attempt saved nothing new
  ckpt_bytes: Optional[int] = None    # for the duty-cycle gate
  save_cadence_s: Optional[int] = None
  step: Optional[int] = None
  outcome: Outcome = Outcome.IN_FLIGHT
  """★Defaults to IN_FLIGHT, not ABANDONED.

  It used to default to ABANDONED, so a node for a job that was RUNNING right now -- verified
  in XM, writing to its bucket -- carried the word ABANDONED. Two fields on the same record
  said opposite things, and anyone reading `outcome` to decide whether a job was alive would
  have concluded it was dead. Caught by monitor-v48 on the canary's own node.

  ★The rule is the one that keeps recurring tonight: a placeholder must not be a value that
  reads as an answer. ABANDONED means "we stopped tracking this and cannot know"; that is a
  real finding, and it must never be produced by simply not having reached the end yet.
  """
  started_at: float = UNKNOWN_TIME
  ended_at: Optional[float] = None
  entry_beacon_at: Optional[float] = None
  """★Set when the job's own code STARTED, not when its first stage finished. A sanity job
  whose first beacon fired only after device_count() succeeded sat at XM 'RUNNING' for 3h42m
  on 8 H100s with zero bytes written -- externally identical to a healthy slow start. Only an
  entry beacon separates 'starting up' from 'died before main()'."""

  def to_dict(self) -> dict[str, Any]:
    d = dataclasses.asdict(self)
    d['outcome'] = self.outcome.value
    return d

  @classmethod
  def from_dict(cls, d: dict[str, Any]) -> 'Node':
    d = dict(d)
    d['outcome'] = Outcome(d.get('outcome', Outcome.ABANDONED.value))
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in known})


@dataclasses.dataclass
class Job:
  """One job, one chain, one row -- for its whole life, resumes included.

  The identity block is immutable after enqueue: changing it means it is a DIFFERENT job.
  `launch_kwargs` in particular is carried verbatim across every re-dispatch. Tonight's
  requeue path rebuilt it from scratch with a single key, which cost twice: the jobs
  cold-started (all 17 lost their checkpoint), and with no config the cost estimator had no
  spec, priced them at a tenth of reality, and let them hold fleet budget at a fake price.
  """

  job_id: str
  # ---- identity (immutable after enqueue) ----
  workdir: str = ''
  target_label: str = UNKNOWN
  project_name: str = UNKNOWN
  """★Submit-time fingerprints, compared against the staged config.sh before launch. When a
  ghost write dropped config.sh from a stagedir, the wrapper backfilled a GLOBAL default
  owned by whoever wrote it last, and two lines' jobs silently ran a third line's target --
  with every structural check green (config.sh present, BUILD present, 415 files)."""
  launch_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
  placement_policy: str = 'RESELECT'          # 'PIN' | 'RESELECT'
  allowed_metros: Optional[list[str]] = None  # ★never dropped (I7)
  allowed_archs: Optional[list[str]] = None
  power: str = ''
  tier: str = 'PROD'
  is_eval: bool = False
  """★Declared by the submitter, never inferred. Gates BATCH eligibility.

  A boolean the caller must state is the whole point: the previous check guessed from a
  substring of `job_id`, a machine-generated `<power>-<6 hex>` string that cannot contain the
  word -- so it refused every BATCH job while still admitting a training job whose id happened
  to spell it. Semantics live in fields, not in the spelling of an identifier.
  """
  topology_locked: bool = False
  priority: int = 0

  # ---- monotonic counters ----
  version: int = 0
  build_attempts: int = 0
  dispatch_count: int = 0

  # ---- mutable current state ----
  state: JobState = JobState.QUEUED
  cur_xid: Optional[str] = None
  cur_cell: Optional[str] = None
  cur_bucket: Optional[str] = None
  last_reason: str = ''
  failure_tail: Optional[str] = None
  """★Persisted stderr tail. One line spent 12 hours and 9 failed builds unable to obtain a
  single error message, because the real output went only to a daemon lane's stdout, through
  an unbounded pipe whose exit status came from `sed`."""
  ever_had_xid: bool = False          # basis of I5
  enqueued_at: float = UNKNOWN_TIME
  updated_at: float = UNKNOWN_TIME
  nodes: list[Node] = dataclasses.field(default_factory=list)

  # ---- derived ----
  @property
  def is_terminal(self) -> bool:
    return self.state in TERMINAL

  @property
  def resume_source(self) -> Optional[Node]:
    """The newest node that actually holds a checkpoint, or None for a cold start."""
    for n in reversed(self.nodes):
      if n.ckpt_path:
        return n
    return None

  def to_dict(self) -> dict[str, Any]:
    d = dataclasses.asdict(self)
    d['state'] = self.state.value
    d['nodes'] = [n.to_dict() for n in self.nodes]
    return d

  @classmethod
  def from_dict(cls, d: dict[str, Any]) -> 'Job':
    d = dict(d)
    d['state'] = JobState(d.get('state', JobState.QUEUED.value))
    d['nodes'] = [Node.from_dict(n) for n in d.get('nodes', [])]
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in known})


# --- invariants ------------------------------------------------------------
class InvariantError(Exception):
  """A transition that would corrupt the chain. Raised on the way IN, so a stale or buggy
  caller is refused rather than silently believed."""


INVARIANTS = """
I1  version strictly increases; a write whose base version != current is REFUSED.
I2  a write touches only genuinely-mutated FIELDS of genuinely-mutated jobs.
I3  build_attempts never decreases          <- codi-torch measured att 1 -> 0
I4  terminal states are one-way             <- gpu-v3: FAILED -> BUILD_REQUESTED in 30s, rebuilt
I5  a job that ever held an xid never returns to cur_xid=None
                                            <- trm: RUNNING xid=284366800 -> xid=None, double-spend
I6  at most one non-terminal xid per job    <- gpu-v3: 5x 8-card B200 from one row
I7  allowed_metros survives every re-dispatch  <- 7 lines; pruner deletes cross-metro jobs
I8  nodes is append-only                    <- codi-v6: deleted rows resurrected from a 9-min snapshot
I9  build_attempts increments ONLY for FailureClass.JOB_DEFECT
                                            <- budget refusals drove attempts to 65
I10 every attempts++ path has a convergence exit
                                            <- route_check.py:804 had none: the real root cause
"""


def attempt_has_ended(node: 'Node') -> bool:
  """★The ONE definition of "this attempt is over". Everything asks this function.

  Not `ended_at is not None` (a 0.0 passes), not `bool(ended_at)` (a 0.0 fails), not
  `>= 0` (a 0.0 passes). Those three spellings of the same idea disagreed on exactly one
  value, and that value was in the store: two readers scanning for inconsistencies got 1 hit
  and 0 hits, and an invariant refused to let the bad value be corrected because it judged
  the node "already closed".

  A credible end time is a POSITIVE one. Zero is not a time here, it is the absence of one
  wearing a plausible date (1970-01-01).
  """
  return node.ended_at is not None and node.ended_at > 0


def fmt_time(t: Optional[float]) -> str:
  """The ONLY sanctioned way to render one of these timestamps for a human.

  ★UNKNOWN_TIME (-1) formats as 1969-12-31 under strftime, and 0.0 as 1970-01-01 -- both are
  perfectly ordinary-looking dates that land in a report and are never re-read. The sentinel
  is only safe while every reader knows it; this function is how a reader stops having to.

  Call this instead of strftime. A bare strftime on `ended_at` is now the thing you have to
  go out of your way to do, rather than the obvious default -- which is the only kind of
  guard that survives, since "remember to check" has failed for every one of us tonight.
  """
  import datetime
  if t is None:
    return '<not set>'
  if t <= 0:
    return '<unknown>'
  return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def fmt_end_time(node: 'Node') -> str:
  """Human-readable end time for an attempt, or why there isn't one."""
  if node.outcome is Outcome.IN_FLIGHT:
    return '<still running>'
  if not attempt_has_ended(node):
    return '<ended, time unknown>'
  return fmt_time(node.ended_at)


def check_node_consistency(job: Job) -> list[str]:
  """Problems with a job's chain that any reader would trip over. Empty list = clean.

  ★This exists because the transition tests only cover records this process just built, and
  the store is full of records written by earlier code. monitor-v48 and I scanned the same
  file for the same inconsistency and got 1 hit versus 0 -- the node carried `ended_at = 0.0`,
  which passes `is None` and fails a truthiness test, so our two readers disagreed about
  whether the attempt had ended. Neither of us was wrong about the data; we were asking
  slightly different questions of a value that should never have existed.

  Run it over the whole store, not just over what you are writing.
  """
  problems = []
  for i, n in enumerate(job.nodes):
    # ★"Ended at an unknown time" is a legitimate state, not an inconsistency. A seed node
    # records an attempt that finished before this scheduler existed: we know THAT it ended
    # (its checkpoint is right there) and not WHEN. Demanding a timestamp would force the
    # migrator to invent one -- which is how the 0.0 got written in the first place.
    # So the contradiction to catch is between the OUTCOME and the CLAIM of being open,
    # measured with UNKNOWN_TIME excluded from both sides.
    ended = attempt_has_ended(n) or n.ended_at == UNKNOWN_TIME
    in_flight = n.outcome is Outcome.IN_FLIGHT
    if ended == in_flight:
      problems.append(
          f'node[{i}] xid={n.xid}: outcome={n.outcome.value} but ended_at={n.ended_at!r} -- '
          f'"is this attempt over" answers differently depending on which field you ask')
    if n.ended_at == 0:
      problems.append(
          f'node[{i}] xid={n.xid}: ended_at is exactly 0, which reads as both "never" and '
          f'"1970". Use -1 for an unknown time, or a real timestamp.')
  open_nodes = [n for n in job.nodes if n.outcome is Outcome.IN_FLIGHT]
  if len(open_nodes) > 1:
    problems.append(f'{len(open_nodes)} nodes are IN_FLIGHT at once (I6)')
  if job.cur_xid and not any(n.xid == job.cur_xid and n.outcome is Outcome.IN_FLIGHT
                             for n in job.nodes):
    problems.append(
        f'cur_xid={job.cur_xid} but no IN_FLIGHT node carries it -- the row claims a live '
        f'job that its own chain does not record')
  return problems


def check_transition(old: Optional[Job], new: Job) -> None:
  """Raise InvariantError if `old -> new` would corrupt the chain.

  Called by the store on every write. Deliberately dumb and total: it re-derives everything
  from the two records rather than trusting a caller's claim about what it changed. A checker
  that only knows how to recognise success is indistinguishable from a broken checker.
  """
  if old is None:                       # enqueue
    if new.version != 0:
      raise InvariantError(f'{new.job_id}: new job must start at version 0, got {new.version}')
    if new.nodes:
      raise InvariantError(f'{new.job_id}: new job cannot start with nodes')
    return

  if new.job_id != old.job_id:
    raise InvariantError(f'job_id is immutable: {old.job_id} -> {new.job_id}')

  # I1 -- monotonic version
  if new.version <= old.version:
    raise InvariantError(
        f'{old.job_id}: version must increase (I1): {old.version} -> {new.version}')

  # I3 -- monotonic counters
  if new.build_attempts < old.build_attempts:
    raise InvariantError(
        f'{old.job_id}: build_attempts decreased (I3): '
        f'{old.build_attempts} -> {new.build_attempts}. A decrease is by definition a '
        f'stale-snapshot overwrite; the counter cannot go down.')
  if new.dispatch_count < old.dispatch_count:
    raise InvariantError(
        f'{old.job_id}: dispatch_count decreased (I3): '
        f'{old.dispatch_count} -> {new.dispatch_count}')

  # I4 -- terminal is one-way
  if old.state in TERMINAL and new.state != old.state:
    raise InvariantError(
        f'{old.job_id}: {old.state.value} is terminal (I4); refusing {new.state.value}. '
        f'A terminal state that can be flipped back to claimable is how one row built five '
        f'8-card jobs.')

  # I5 -- an xid, once held, is never forgotten
  if old.ever_had_xid and not new.ever_had_xid:
    raise InvariantError(f'{old.job_id}: ever_had_xid cannot be unset (I5)')
  if old.ever_had_xid and new.cur_xid is None and new.state in CLAIMABLE:
    # ★A launched job MAY become claimable again -- that is exactly what a resume is. What it
    # may not do is become claimable while an attempt is still OPEN, because then the real
    # job keeps running while the row looks never-launched, and the dispatcher builds a
    # second one (measured: five 8-card B200s from one row). So the discriminator is not
    # "did it ever launch", it is "is its last attempt accounted for".
    #
    # This distinction cost a test failure to find, and the failing test was right: an
    # invariant that also forbids the legitimate transition would have made resume impossible
    # while looking like extra safety.
    #
    # ★Judge the PROPOSED record, not the previous one: `old` still shows the attempt open
    # (that is what this transition is closing). Reading `old.nodes` here made the invariant
    # reject every legitimate resume -- a second bug the tests caught, in the fix for the
    # first. Both times the guard was "right" about a state nobody was proposing.
    unaccounted = [n for n in new.nodes if n.ended_at is None]
    if unaccounted or (new.cur_xid and not new.nodes):
      raise InvariantError(
          f'{old.job_id}: refusing to make a job claimable again while attempt '
          f'{(unaccounted[0].xid if unaccounted else old.cur_xid)} is still open (I5). Close '
          f'it into a node first -- otherwise the dispatcher builds a second job on the same '
          f'config while the first is still running.')

  # I6 -- one live xid at a time
  if new.cur_xid and new.state in LIVE_XID_STATES:
    live_nodes = [n for n in new.nodes
                  if n.xid == new.cur_xid and n.ended_at is None]
    if len(live_nodes) > 1:
      raise InvariantError(f'{new.job_id}: more than one open node for xid {new.cur_xid} (I6)')
  open_nodes = [n for n in new.nodes if n.ended_at is None]
  if len(open_nodes) > 1:
    raise InvariantError(
        f'{new.job_id}: {len(open_nodes)} open (un-ended) nodes (I6); a new dispatch must '
        f'close the previous attempt first.')

  # I7 -- placement constraints are never silently widened
  if old.allowed_metros and new.allowed_metros != old.allowed_metros:
    raise InvariantError(
        f'{old.job_id}: allowed_metros is immutable (I7): '
        f'{old.allowed_metros} -> {new.allowed_metros}. Losing it does not fail loudly -- '
        f'the job builds, runs, and is then silently deleted by the pruner.')

  # I8 -- append-only chain
  if len(new.nodes) < len(old.nodes):
    raise InvariantError(
        f'{old.job_id}: nodes shrank (I8): {len(old.nodes)} -> {len(new.nodes)}')
  for i, on in enumerate(old.nodes):
    nn = new.nodes[i]
    # ★Identity is the xid, and only the xid. This used to compare started_at too, on the
    # theory that a node is "the attempt that began at time T" -- but that makes a timestamp
    # an identity field, so correcting a bad one (the 0.0 seed values) reads as replacing the
    # node. An attempt is identified by the experiment it launched; when it started is a fact
    # ABOUT it, and facts get corrected.
    if nn.xid != on.xid:
      raise InvariantError(
          f'{old.job_id}: node[{i}] was rewritten (I8): {on.xid} -> {nn.xid}')
    # ★"Closed" means a real end time, not merely a non-None one. A seed node written by the
    # migrator carried ended_at=0 -- not None, so this read it as closed and refused to let
    # the value be corrected to the -1 "unknown time" sentinel. The invariant was right to
    # protect closed nodes and wrong about which nodes were closed: it used a different
    # definition of "ended" than check_node_consistency does, and two definitions of the same
    # word is how the 0.0 got through in the first place.
    if attempt_has_ended(on) and nn.ended_at != on.ended_at:
      raise InvariantError(f'{old.job_id}: node[{i}] was already closed; cannot re-open (I8)')

  # identity block is immutable
  for f in ('workdir', 'target_label', 'project_name'):
    if getattr(old, f) != getattr(new, f) and getattr(old, f) not in ('', UNKNOWN):
      raise InvariantError(
          f'{old.job_id}: {f} is immutable after enqueue ({getattr(old, f)!r} -> '
          f'{getattr(new, f)!r}); a different {f} is a DIFFERENT job.')


# --- transitions -----------------------------------------------------------
# These are the ONLY sanctioned mutations. Each returns a modified copy with version bumped,
# so a caller cannot accidentally half-apply one. `check_transition` still runs at the store.

def _bump(job: Job, now: float) -> Job:
  new = Job.from_dict(job.to_dict())     # deep copy
  new.version = job.version + 1
  new.updated_at = now
  return new


def open_attempt(job: Job, xid: str, cell: str, metro: str, continent: str,
                 bucket: str, now: Optional[float] = None) -> Job:
  """Record that a dispatch produced `xid`. Opens a node and takes the single live-xid slot."""
  now = time.time() if now is None else now
  if job.cur_xid and job.state in LIVE_XID_STATES:
    raise InvariantError(
        f'{job.job_id}: already holds live xid {job.cur_xid} (I6); close it before opening '
        f'{xid}. Two live xids on one row is the double-spend shape.')
  new = _bump(job, now)
  new.nodes.append(Node(xid=xid, cell=cell, metro=metro, continent=continent,
                        bucket=bucket, started_at=now))
  new.cur_xid, new.cur_cell, new.cur_bucket = xid, cell, bucket
  new.ever_had_xid = True
  new.state = JobState.SUBMITTED
  new.dispatch_count += 1
  new.last_reason = f'submitted xid={xid} cell={cell} ({metro}/{continent})'
  return new


def close_attempt(job: Job, outcome: Outcome, ckpt_path: Optional[str] = None,
                  ckpt_bytes: Optional[int] = None, step: Optional[int] = None,
                  reason: str = '', now: Optional[float] = None) -> Job:
  """Close the open node. ★`ckpt_path` is stored verbatim -- see Node's docstring."""
  now = time.time() if now is None else now
  new = _bump(job, now)
  open_nodes = [n for n in new.nodes if n.ended_at is None]
  if not open_nodes:
    raise InvariantError(f'{job.job_id}: no open attempt to close')
  n = open_nodes[-1]
  n.ended_at, n.outcome = now, outcome
  if ckpt_path is not None:
    n.ckpt_path, n.ckpt_bytes, n.step = ckpt_path, ckpt_bytes, step
  new.cur_xid = None
  new.last_reason = reason or f'attempt {n.xid} ended: {outcome.value}'
  if outcome is Outcome.COMPLETED:
    new.state = JobState.COMPLETED
  elif outcome is Outcome.CANCELLED:
    new.state = JobState.CANCELLED
  else:
    new.state = JobState.QUEUED        # preempted / failed-but-retriable: chain continues
  return new


def record_failure(job: Job, cls: FailureClass, reason: str, tail: Optional[str] = None,
                   max_attempts: int = 3, now: Optional[float] = None) -> Job:
  """★The attempts rule, in one place (I9/I10).

  Only FailureClass.JOB_DEFECT increments `build_attempts`. An ENVIRONMENT refusal -- budget,
  quota, capacity, cluster fault -- is the fleet saying "not now"; it returns the job to the
  queue with its strike count untouched, because those refusals are transient by nature
  (income swung 25x in one night) and counting them sent jobs three rounds from success into
  a state only a human could leave.

  ★Every path through this function CONVERGES: JOB_DEFECT either retries or lands in HELD;
  ENVIRONMENT lands in DEFERRED and is re-judged next round. The old code had a third path
  that only did `attempts += 1` with no exit at all -- that, not the budget mislabelling,
  is why entries reached att=65 and kept occupying the build channel forever.

  `tail` is persisted on the record. Without it a failure is a state with no explanation:
  one line spent 12 hours and 9 attempts unable to obtain a single line of error text.
  """
  now = time.time() if now is None else now
  new = _bump(job, now)
  new.failure_tail = tail or new.failure_tail
  if cls is FailureClass.ENVIRONMENT:
    new.state = JobState.DEFERRED
    new.last_reason = f'DEFERRED ({reason}); attempts unchanged at {new.build_attempts}'
    return new
  new.build_attempts += 1
  if new.build_attempts >= max_attempts:
    new.state = JobState.HELD
    new.last_reason = (f'HELD after {new.build_attempts} job-intrinsic build failures: '
                       f'{reason}')
  else:
    new.state = JobState.QUEUED
    new.last_reason = (f'build failed ({new.build_attempts}/{max_attempts}): {reason}')
  return new


def promote_deferred(job: Job, now: Optional[float] = None) -> Job:
  """DEFERRED -> QUEUED for the next round. Never touches build_attempts."""
  now = time.time() if now is None else now
  if job.state is not JobState.DEFERRED:
    raise InvariantError(f'{job.job_id}: promote_deferred on state {job.state.value}')
  new = _bump(job, now)
  new.state = JobState.QUEUED
  new.last_reason = 'promoted from DEFERRED for re-test'
  return new


def hold(job: Job, reason: str, now: Optional[float] = None) -> Job:
  """Park a malformed row for a human. ★Not a strike: the job did not fail, it cannot be
  expressed -- and a counter is the wrong instrument for "someone must fix this".

  ★A sanctioned transition rather than a hand-built record. The dispatcher used to construct
  one inline and forgot `updated_at`, so a row changed state while claiming it had not been
  touched since an earlier time. That is a small lie with a specific cost: monitor-v48's
  sentinel saw the file rewritten with no entry newer than the baseline and had to ask
  whether a third-party writer existed. Going through _bump() makes the timestamp impossible
  to omit.
  """
  now = time.time() if now is None else now
  new = _bump(job, now)
  new.state = JobState.HELD
  new.last_reason = f'HELD: {reason}'
  return new


def cancel(job: Job, reason: str, now: Optional[float] = None) -> Job:
  """Withdraw a job. Terminal (I4)."""
  now = time.time() if now is None else now
  new = _bump(job, now)
  for n in new.nodes:
    if n.ended_at is None:
      n.ended_at, n.outcome = now, Outcome.CANCELLED
  new.cur_xid = None
  new.state = JobState.CANCELLED
  new.last_reason = f'CANCELLED: {reason}'
  return new


# --- enqueue-time validation ------------------------------------------------
# The single gate every job passes through. Checks live HERE, not in the wrapper, because
# this is the one path no submission can bypass: a shell-side guard was skipped entirely
# tonight when the binary it was nested under went missing, and `--metro` silently became a
# no-op. Nothing here is nested under an `if <tool> exists`.

FORBIDDEN_ARCHS = frozenset({'gb200', 'gb300'})
"""★Operator directive (2026-08-28, from their manager): GB200/GB300 must not be used.

Enforced by REFUSING at enqueue, deliberately NOT by deleting the price-cap rows in
tpu_wrapper.sh / cap_policy.py. Those rows read `gb200) echo "20"`, and the caller treats an
empty cap as "no policy for this family: leave the job UNCAPPED" -- so deleting them would
RELAX the constraint while looking like a removal. The caps stay; the family is refused here.
"""


class RejectedAtEnqueue(Exception):
  """A job that must not enter the queue. Distinct from InvariantError: this is about the
  submission being wrong, not about a transition corrupting an existing chain."""


LAUNCHER_SWALLOWED_KEYS = frozenset({
    'load_from', 'config.load_from',
    'wandb_resume_id', 'config.wandb_resume_id',
    'cell',
})
"""Keys `xm_launcher.py` intercepts and forwards via the environment rather than to the
binary. Anything here that appears in launch_kwargs is silently dropped.

★NOT a ban on dotted keys: `--config.*` is a normal ml_collections override and some entry
points require it (`--config.eval_ckpt=`), so a blanket dotted-key refusal makes legitimate
eval jobs unsubmittable. Only these specific names are swallowed.
"""


POISONED_BUCKET_PREFIXES = ('/cns/yutulpz-d/',)
"""Buckets a job must not be pointed at. Currently: the PERSONAL Colossus quota.

【实测 2026-08-28 21:53Z】`qiaos` is at 468.49G of a 500G personal ceiling and the handle is
poisoned: a write returns `resource_exhausted: ... over Colossus bytes HDD quota` with
`file_poisoned: true` -- ★and still leaves a 0-byte file behind. So the failure presents as
"the file exists", which is why a run lost 5000 steps before anyone noticed: step_70000
landed, the quota alarm fired, step_75000 left only a tmp dir and 0-byte rank logs.

★The group-billed bucket in the same metro (/cns/oi-d/, 1.11T of 100T) is unaffected -- I
wrote a probe there and re-read it after 10s: intact. So this is a quota boundary, not a
storage outage, and the fix is to stop defaulting jobs onto a personal ceiling.
"""


def _bucket_of(job: 'Job') -> Optional[str]:
  """The bucket this job will actually write to, or None if it never says.

  ★None is NOT the same as "fine": the launcher's own default is the personal bucket, so a
  job that declares nothing inherits the poisoned one. Silence has to be refused explicitly.
  """
  for k in ('bucket', 'CHECKPOINT_BUCKET', '--bucket'):
    v = (job.launch_kwargs or {}).get(k)
    if v:
      return str(v)
  return None


def _looks_like_depot_root(workdir: str, top_level_count: Optional[int] = None) -> bool:
  """A google3 depot root, by SHAPE or by MARKERS -- never by matching a known-bad string.

  Both a monitor and I independently searched for the one bad workdir we already knew about,
  and both of us missed rows pointing at a DIFFERENT depot root. Scanning by predicate found
  20 offending rows where string-matching found 4.
  """
  import os
  import re
  if not workdir:
    return False
  w = workdir.rstrip('/')
  if re.match(r'^/google(_src)?/src/cloud/[^/]+/[^/]+/google3$', w):
    return True
  if re.match(r'^/google/src/(head/depot|files/[^/]+/depot)/google3$', w):
    return True
  if (os.path.isfile(os.path.join(w, 'WORKSPACE'))
      and os.path.isdir(os.path.join(w, 'devtools'))
      and os.path.isdir(os.path.join(w, 'third_party'))):
    return True
  if top_level_count is not None and top_level_count > 400:
    return True
  return False


def validate_enqueue(job: Job, *, top_level_count: Optional[int] = None,
                     stagedir_root: Optional[str] = None) -> None:
  """Raise RejectedAtEnqueue if this job must not enter the queue.

  Every check below is a fix for something that reached production tonight.
  """
  import os

  # --- forbidden accelerator families -------------------------------------
  arch = (job.power or '').split('-')[0].lower()
  if arch in FORBIDDEN_ARCHS:
    raise RejectedAtEnqueue(
        f'{job.job_id}: {arch} is not permitted (operator directive). Requesting it '
        f'explicitly is refused here, not merely omitted from candidate lists.')
  for a in (job.allowed_archs or []):
    if a.lower() in FORBIDDEN_ARCHS:
      raise RejectedAtEnqueue(
          f'{job.job_id}: allowed_archs contains {a}, which is not permitted '
          f'(operator directive).')

  # --- workdir ------------------------------------------------------------
  if not job.workdir:
    raise RejectedAtEnqueue(f'{job.job_id}: workdir is required')
  if _looks_like_depot_root(job.workdir, top_level_count):
    raise RejectedAtEnqueue(
        f'{job.job_id}: workdir {job.workdir!r} is a google3 depot root. The stage rsync '
        f'would copy the depot into a subdirectory of itself: one such row produced 76.1% '
        f'of a day\'s 91,437 CreateSnapshot failures. Point workdir at the project checkout.')
  if os.path.realpath(job.workdir) == os.path.realpath('/tmp'):
    raise RejectedAtEnqueue(
        f'{job.job_id}: workdir is /tmp. 15 rows had this (from a requeue path that lost the '
        f'original), and each would package ~9,400 top-level entries and then die as a '
        f'launcher FATAL for want of a config -- a failure shaped exactly like an infra fault.')
  if not os.path.isdir(job.workdir):
    raise RejectedAtEnqueue(f'{job.job_id}: workdir does not exist: {job.workdir}')

  # --- rsync containment, both directions ---------------------------------
  if stagedir_root:
    src = os.path.realpath(job.workdir).rstrip('/') + '/'
    dst = os.path.realpath(stagedir_root).rstrip('/') + '/'
    if dst.startswith(src):
      raise RejectedAtEnqueue(
          f'{job.job_id}: the stagedir is INSIDE workdir -- rsync would copy the source into '
          f'itself (the 300s-timeout / rm -rf / retry loop).')
    if src.startswith(dst):
      raise RejectedAtEnqueue(
          f'{job.job_id}: workdir is INSIDE the stagedir root, and staging opens with '
          f'`rm -rf <stagedir>` -- it would DELETE the source.')

  # --- identity fingerprints ----------------------------------------------
  if job.target_label in ('', UNKNOWN) or job.project_name in ('', UNKNOWN):
    raise RejectedAtEnqueue(
        f'{job.job_id}: target_label and project_name must be recorded at enqueue. They are '
        f'compared against the staged config.sh before launch: when a ghost write dropped '
        f'config.sh, the wrapper backfilled a GLOBAL default owned by whoever wrote it last, '
        f'and two lines silently ran a third line\'s target with every structural check green.')

  # --- storage destination -------------------------------------------------
  bucket = _bucket_of(job)
  if bucket is None:
    raise RejectedAtEnqueue(
        f'{job.job_id}: no bucket declared. The launcher would default to the PERSONAL '
        f'Colossus quota (/cns/yutulpz-d/...), which is at 468G of a 500G ceiling and whose '
        f'handle is poisoned -- writes fail with resource_exhausted AND leave a 0-byte file, '
        f'so the loss looks like a file that exists. Declare a group-billed bucket in your '
        f'data metro, e.g. /cns/oi-d/home/qiaos/eqr_data for tul.')
  for bad in POISONED_BUCKET_PREFIXES:
    if bucket.startswith(bad):
      raise RejectedAtEnqueue(
          f'{job.job_id}: bucket {bucket!r} is on the personal Colossus quota, which is full '
          f'and poisoned (see POISONED_BUCKET_PREFIXES). A run already lost 5000 steps this '
          f'way. Use the group-billed bucket in the same metro.')

  # --- launch_kwargs shape -------------------------------------------------
  lk = job.launch_kwargs or {}
  # ★Only the keys the launcher SWALLOWS are pseudo-keys. `--config.*` is otherwise a real
  # ml_collections override that the launcher forwards and the binary parses -- some entry
  # points REQUIRE the dotted form (a bare `--eval_ckpt=` is rejected upstream), so refusing
  # every dotted key would make a legitimate eval unsubmittable.
  #
  # The distinction is not "does it contain a dot". It is "does anything downstream consume
  # it": the launcher `continue`s past this exact list, routing those values through the
  # environment instead, so a job that puts them in launch_kwargs has them silently dropped
  # and cold-starts. Reported by parcae-v6 against its own eval recipe, with the launcher
  # line numbers -- my first version of this gate was wrong in the widening direction.
  #
  # ★Keep in sync with xm_launcher.py's skip-list (search: `--config.load_from=`). A copy
  # here is a second source of truth; when it drifts, the failure is silent in both
  # directions, so re-check it rather than trusting this comment.
  swallowed = sorted(k for k in lk if str(k).lstrip('-') in LAUNCHER_SWALLOWED_KEYS)
  if swallowed:
    raise RejectedAtEnqueue(
        f'{job.job_id}: launch_kwargs contains {swallowed}, which the launcher consumes and '
        f'routes through the environment instead of passing to the binary -- putting them '
        f'here means the value is silently dropped and the job cold-starts. Use the '
        f'scheduler field instead (a checkpoint belongs in the chain, not in launch_kwargs).')
  # ★An exp_name with nothing else is the residue of several `--launch=` arguments
  # overwriting each other: config and load_from were lost. Such a row builds and then dies
  # in the launcher, or worse, runs the wrong thing -- and it looks well-formed either way.
  # ★What makes a row runnable is that it says WHAT TO RUN -- not specifically that it has
  # a `config` key. A torch sanity job carries `app.sanity_only=true` and no config file at
  # all, and it is perfectly well-formed; my first version of this check demanded `config`
  # and refused it (reported by trm-torch-v3 against its only row).
  #
  # The shape actually being caught is the residue of repeated `--launch=` arguments
  # overwriting each other, which leaves a row with nothing but its NAME and the plumbing
  # every row has. So the test is: strip the keys that are pure plumbing, and see whether
  # anything is left.
  _PLUMBING = {'exp_name', 'bucket', 'CHECKPOINT_BUCKET', '--bucket', 'group', 'workdir',
               'tier', 'borg_max_task_failures', 'borg_max_task_evictions',
               'borg_max_per_task_failures', 'skip-preflight'}
  if lk and not (set(lk) - _PLUMBING):
    raise RejectedAtEnqueue(
        f'{job.job_id}: launch_kwargs says nothing about what to RUN (present: {sorted(lk)} '
        f'-- all of it plumbing). Rows in this shape are residue of repeated --launch= '
        f'arguments overwriting each other: the config, the checkpoint, or the entry-point '
        f'flag was lost, and the row reaches the launcher with nothing to execute. '
        f'Re-enqueue from the original definition rather than repairing this row.')

  # --- personal-only metros are refused outright ---------------------------
  # ★phx / ske are metros the GROUP has no storage registration in. They are
  # the worst of the three metro outcomes, because they do not fail like the
  # other two: a metro in neither of the launcher's dicts makes xm_launcher
  # SystemExit into an inert zero-work-unit shell (visibly broken), while
  # phx/ske RESOLVE, launch, bill, and write to the personal 500 GiB quota --
  # ~468G used and its handle poisoned -- where the write fails with
  # resource_exhausted AND LEAVES A 0-BYTE FILE. The loss looks like a file
  # that exists and `tpu check` still says SUBMITTED.
  # The launcher (xm_launcher.py `_local_bucket`) refuses these too; this is
  # the second, earlier gate, so the refusal costs zero credits instead of
  # arriving after an XID exists. Two gates on purpose: the launcher is the
  # one nothing can bypass, this one is the one that gives a readable error
  # at the moment the human typed the command.
  if job.allowed_metros:
    personal = sorted({m.strip().lower() for m in job.allowed_metros}
                      & PERSONAL_ONLY_METROS)
    if personal:
      raise RejectedAtEnqueue(
          f'{job.job_id}: metro(s) {personal} have NO group storage '
          f'registration, so every write lands on the personal 500 GiB '
          f'per-cell quota (~468G used, handle poisoned): the write fails '
          f'with resource_exhausted and still leaves a 0-byte file, so the '
          f'job looks like it produced output. Use a metro with group '
          f'storage instead, or pass an explicit group-billed --bucket if '
          f'you have chosen this location on purpose.')

  # --- bucket must be reachable from where the job will land ---------------
  # ★A bucket alone is not enough: it has to be in a metro the job can actually be placed
  # in. A row with a tul bucket and no metro constraint can land in cbf, and then every
  # checkpoint write crosses a metro -- ~94x slower, duty cycle under the 0.20 floor, and
  # the pruner deletes the job mid-run with no preemption notice and no crash.
  # Caught by monitor-v47, whose scan included this check while mine did not.
  if bucket.startswith('/cns/') and job.allowed_metros:
    parts = bucket.split('/')
    storage_cell = parts[2][:-2] if len(parts) > 2 and parts[2].endswith('-d') else None
    if storage_cell:
      if _cell_locality is not None:
        bucket_metro = str(_cell_locality.metro_of(storage_cell))
        allowed = {m.lower() for m in job.allowed_metros}
        if bucket_metro != UNKNOWN and bucket_metro.lower() not in allowed:
          raise RejectedAtEnqueue(
              f'{job.job_id}: bucket {bucket} lives in metro {bucket_metro}, which is not in '
              f'allowed_metros={job.allowed_metros}. Every checkpoint write would cross a '
              f'metro: ~94x slower, duty cycle under the 0.20 floor, and the pruner deletes '
              f'the job mid-run. Either declare {bucket_metro} as an allowed metro, or use a '
              f'bucket co-located with the metros you did declare.')
  if bucket.startswith('/cns/') and not job.allowed_metros:
    raise RejectedAtEnqueue(
        f'{job.job_id}: a CNS bucket ({bucket}) is declared but allowed_metros is empty, so '
        f'nothing keeps the job in the bucket\'s metro. Declare the metros you can run in.')

  # --- placement ----------------------------------------------------------
  if job.placement_policy not in ('PIN', 'RESELECT'):
    raise RejectedAtEnqueue(
        f'{job.job_id}: placement_policy must be PIN or RESELECT, got '
        f'{job.placement_policy!r}')
  if job.tier == 'BATCH' and not job.is_eval:
    raise RejectedAtEnqueue(
        f'{job.job_id}: BATCH is eval-only, and this job does not declare is_eval=True. '
        f'BATCH is a PAYING best-effort tier that any PROD demand preempts instantly, so a '
        f'training run on it is silently starved AND still billed.\n'
        f'★Declare it explicitly (is_eval=True); do not rename the job to get past this. An '
        f'earlier version of this check inferred "eval" from a substring of job_id, and was '
        f'wrong in BOTH directions: job_id is machine-generated as "<power>-<6 hex>", so it '
        f'can never contain the word (every BATCH job was refused, including real evals), '
        f'while a hand-written id containing "eval" let a TRAINING job onto BATCH -- the one '
        f'thing the rule exists to prevent.')


def verify_identity(job: Job, staged_target_label: str, staged_project_name: str) -> None:
  """Pre-launch: does the staged package belong to THIS job? (C5)

  Asks whether config.sh is MINE, not whether config.sh EXISTS. The completeness check that
  ran tonight verified existence -- BUILD present, config.sh present, 415 files -- and passed
  a package belonging to an entirely different line.
  """
  if staged_target_label != job.target_label or staged_project_name != job.project_name:
    raise RejectedAtEnqueue(
        f'{job.job_id}: STAGED PACKAGE IDENTITY MISMATCH. Expected '
        f'{job.project_name}/{job.target_label}, staged package declares '
        f'{staged_project_name}/{staged_target_label}. Refusing to launch: this is how a job '
        f'runs another line\'s experiment while every structural check reports green.')
