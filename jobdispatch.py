"""The dispatch loop: the one place a job moves from QUEUED to launched.

This is deliberately the ONLY drainer. The previous generation had three entry points
(a reroute loop, a build worker, and a one-shot CLI the daemon invoked without a timeout),
each holding its own snapshot and writing it back; the resulting overwrites are why this
rewrite exists. Here, every mutation goes through `JobStore.commit`, which refuses a write
computed from a version that has moved.

ORDER OF THE GATES, AND WHY. Each step is placed so that the cheapest and most irreversible
refusals happen before anything is spent:

  1. identity        -- is the staged package MINE?         (a wrong package runs someone
                                                              else's experiment, all green)
  2. placement       -- a cell inside my declared metros    (outside them, the pruner deletes
                                                              the job mid-run)
  3. co-location     -- checkpoint local to that cell       (a remote write is ~94x and gets
                                                              the job pruned)
  4. price           -- affordable, on the shared path      (a stale price cancelled a healthy
                                                              six-hour job)
  5. build + submit  -- the only step that spends anything

★A failure at 1 is a JOB_DEFECT; at 2, 3 and 4 it is ENVIRONMENT. That distinction is the
whole point of `FailureClass`: only a defect of this job may accumulate strikes, because the
fleet saying "not now" is transient (income swung 25x in one night) and counting it sent jobs
three rounds from success into a state only a human could leave.
"""

from __future__ import annotations

import dataclasses
import os
import time
from typing import Any, Callable, Optional, Protocol

try:
  # Under blaze these resolve as package modules; a bare `python3 file.py` run
  # (used by the tests and by ad-hoc inspection) falls back to the flat names.
  from google3.experimental.users.qiaos.tpu_utils import jobbuild as jb
  from google3.experimental.users.qiaos.tpu_utils import jobchain as jc
  from google3.experimental.users.qiaos.tpu_utils import jobcost
  from google3.experimental.users.qiaos.tpu_utils import jobplace as jp
  from google3.experimental.users.qiaos.tpu_utils import jobstore as js
except ImportError:  # pragma: no cover - flat-layout fallback
  import jobbuild as jb  # type: ignore
  import jobchain as jc  # type: ignore
  import jobcost  # type: ignore
  import jobplace as jp  # type: ignore
  import jobstore as js  # type: ignore



@dataclasses.dataclass
class DispatchResult:
  job_id: str
  action: str            # 'submitted' | 'deferred' | 'held' | 'failed' | 'skipped'
  detail: str


class Builder(Protocol):
  """Builds and submits one job. Returns (xid, output_tail); xid None on failure."""
  def __call__(self, job: jc.Job, placement: jp.Placement,
               env: dict[str, str]) -> tuple[Optional[str], str]: ...


class Dispatcher:
  """One round = one pass over the queue. Stateless between rounds, on purpose.

  ★No in-memory cache of the queue survives a round. A long-lived reader holding a private
  copy of shared state is the same bug in a different carrier -- it appeared tonight as a
  queue rolled back by a stale snapshot, as bash not re-reading an already-parsed function,
  and as a Python process serving a day-old import while the fix sat on disk. Re-reading is
  cheap; being wrong about the world is not.
  """

  def __init__(self, store: js.JobStore, builder: Builder,
               available_cells_fn: Callable[[], list[str]],
               headroom_fn: Callable[[], Optional[float]],
               copy_fn: jp.CopyFn,
               exists_fn: Callable[[str], bool],
               verify_fn: Optional[Callable[[str], bool]] = None,
               identity_fn: Optional[Callable[[jc.Job], tuple[str, str]]] = None,
               max_attempts: int = 3,
               log: Optional[Callable[[str], None]] = None):
    self.store = store
    self.builder = builder
    self.available_cells_fn = available_cells_fn
    self.headroom_fn = headroom_fn
    self.copy_fn = copy_fn
    self.exists_fn = exists_fn
    self.verify_fn = verify_fn
    self.identity_fn = identity_fn
    self.max_attempts = max_attempts
    self.log = log or (lambda s: None)

  # --- one round ----------------------------------------------------------
  def round(self, limit: int = 1) -> list[DispatchResult]:
    """Promote deferred jobs, then dispatch up to `limit` queued ones.

    `limit=1` by default: builds are serial on this host, and a concurrent build was the
    original cause of zombie XIDs.
    """
    out: list[DispatchResult] = []
    self._promote_deferred()
    jobs = self.store.load()
    queued = sorted((j for j in jobs.values() if j.state is jc.JobState.QUEUED),
                    key=lambda j: (-j.priority, j.enqueued_at))
    for job in queued[:limit]:
      out.append(self._dispatch_one(job))
    return out

  def _promote_deferred(self) -> None:
    """DEFERRED -> QUEUED. Never touches build_attempts: the refusal was not this job's fault.

    ★Promotion is unconditional rather than backoff-based. The refusals are fleet-wide and
    transient, so a job re-tests cheaply; a backoff here would reproduce the old behaviour of
    punishing a job for the fleet's state.
    """
    for job in list(self.store.load().values()):
      if job.state is jc.JobState.DEFERRED:
        try:
          self.store.commit(jc.promote_deferred(job))
        except js.StaleWriteError:
          pass          # someone else moved it; the next round re-reads

  # --- one job ------------------------------------------------------------
  def _dispatch_one(self, job: jc.Job) -> DispatchResult:
    jid = job.job_id

    # --- gate 0: the enqueue contract, re-checked at dispatch -------------
    # ★A row can enter the store and only later become undispatchable: the store predates a
    # gate, a migration carried an old shape across, or the world moved (the personal bucket
    # was fine this morning and is poisoned now). Validating only at enqueue means the gate
    # protects the door and not the road.
    #
    # Found while picking a canary: h100-8-c27e8f sits in the store with no bucket at all,
    # `validate_enqueue` refuses it, and the dispatcher would have submitted it anyway --
    # straight onto the poisoned personal quota, whose writes fail while leaving a 0-byte
    # file behind. An entry check is not an invariant; this makes it one.
    try:
      jc.validate_enqueue(job)
    except jc.RejectedAtEnqueue as e:
      return self._hold(job, f'no longer satisfies the enqueue contract: {e}')

    # --- gate 1: identity (JOB_DEFECT if wrong) ---------------------------
    if self.identity_fn is not None:
      try:
        staged_target, staged_project = self.identity_fn(job)
        jc.verify_identity(job, staged_target, staged_project)
      except jc.RejectedAtEnqueue as e:
        return self._fail(job, jc.FailureClass.JOB_DEFECT, str(e))
      except Exception as e:                              # noqa: BLE001
        # Could not READ the staged identity: that is the environment, not the job.
        return self._defer(job, f'could not verify staged identity: {e}')

    # --- gate 2: placement (ENVIRONMENT) ----------------------------------
    try:
      placement = jp.choose(job, self.available_cells_fn())
    except jp.PlacementError as e:
      return self._defer(job, str(e))

    # --- gate 3: checkpoint co-location (ENVIRONMENT) ---------------------
    try:
      load_from = jp.colocate_checkpoint(
          job, placement, copy_fn=self.copy_fn, exists_fn=self.exists_fn,
          verify_fn=self.verify_fn)
    except jp.CheckpointCopyError as e:
      # ★Deferred, not failed: the checkpoint is fine, the copy could not be made. Launching
      # anyway against the remote path is the fallback that gets a job pruned an hour later.
      return self._defer(job, f'checkpoint not co-located: {e}')

    # --- gate 4: price (ENVIRONMENT) --------------------------------------
    try:
      cost, basis = jobcost.price_job(job.power, job.tier)
    except jobcost.PricingUnavailable as e:
      return self._defer(job, f'price unavailable: {e}')   # never price as 0 and proceed
    headroom = self.headroom_fn()
    if headroom is not None and cost > headroom:
      return self._defer(job, f'over the bar: {cost:.4g} > headroom {headroom:.4g} '
                              f'(basis={basis}); attempts unchanged')

    # --- gate 5: build + submit (the only step that spends) ---------------
    env = jp.launch_env(job, placement, load_from)
    self.log(f'[dispatch] {jid} -> {placement.cell} ({placement.metro}/'
             f'{placement.continent}) cost={cost:.4g} basis={basis} '
             f'load_from={load_from or "<cold start>"}')
    try:
      xid, tail = self.builder(job, placement, env)
    except Exception as e:                                # noqa: BLE001
      return self._fail(job, jc.FailureClass.JOB_DEFECT, f'builder raised: {e}')

    if not xid and jb.DRY_RUN_MARKER in (tail or ''):
      # ★Nothing was executed, so nothing is recorded. Not a failure, not a deferral, not a
      # version bump -- a dry run that leaves footprints cannot be used to answer "what would
      # happen", because by the second round it is describing the marks it made itself.
      self.log(f'[dispatch] {jid}: DRY RUN -- would submit to {placement.cell} '
               f'({placement.metro}) cost={cost:.4g} load_from={load_from or "<cold>"}')
      return DispatchResult(jid, 'dry_run', f'would submit to {placement.cell}')

    if not xid:
      # ★No XID with no budget marker is a real build failure -- and it MUST converge. The
      # old code's equivalent path did `attempts += 1` and nothing else, with no exit at all,
      # which is how rows reached att=65 while occupying the build channel forever.
      return self._fail(job, jc.FailureClass.JOB_DEFECT,
                        'build produced no XID', tail=tail)

    try:
      launched = jc.open_attempt(job, xid, placement.cell, placement.metro,
                                 placement.continent, placement.bucket)
      self.store.commit(launched)
    except (js.StaleWriteError, jc.InvariantError) as e:
      # ★The job IS running -- say so loudly rather than losing the xid. A row that forgets a
      # live xid is how one config produced five concurrent 8-card jobs.
      self.log(f'[dispatch] ★{jid} LAUNCHED as {xid} but the record could not be updated: '
               f'{e}. RECONCILE MANUALLY -- the job is running.')
      return DispatchResult(jid, 'submitted', f'xid={xid} (record update failed: {e})')
    return DispatchResult(jid, 'submitted', f'xid={xid} cell={placement.cell}')

  # --- outcomes -----------------------------------------------------------
  def _hold(self, job: jc.Job, reason: str) -> DispatchResult:
    """Park a row that cannot legally be dispatched. ★Not a strike: the job did not fail, it
    is malformed -- and a counter is the wrong instrument for "a human must fix this"."""
    try:
      self.store.commit(jc.hold(job, reason))
    except (js.StaleWriteError, jc.InvariantError):
      pass
    self.log(f'[dispatch] {job.job_id} -> HELD ({reason})')
    return DispatchResult(job.job_id, 'held', reason)

  def _defer(self, job: jc.Job, reason: str) -> DispatchResult:
    try:
      self.store.commit(jc.record_failure(job, jc.FailureClass.ENVIRONMENT, reason,
                                          max_attempts=self.max_attempts))
    except js.StaleWriteError:
      pass
    self.log(f'[dispatch] {job.job_id} -> DEFERRED ({reason})')
    return DispatchResult(job.job_id, 'deferred', reason)

  def _fail(self, job: jc.Job, cls: jc.FailureClass, reason: str,
            tail: Optional[str] = None) -> DispatchResult:
    try:
      updated = self.store.commit(jc.record_failure(job, cls, reason, tail=tail,
                                                    max_attempts=self.max_attempts))
    except js.StaleWriteError:
      return DispatchResult(job.job_id, 'skipped', 'record moved; will re-read next round')
    action = 'held' if updated.state is jc.JobState.HELD else 'failed'
    self.log(f'[dispatch] {job.job_id} -> {updated.state.value} ({reason})')
    return DispatchResult(job.job_id, action, reason)

  # --- C11 ----------------------------------------------------------------
  def belief_line(self) -> str:
    """What this process believes right now. Log it every round.

    ★A long-lived process must be externally falsifiable. Tonight three different carriers of
    "edited but not in effect" -- environ, import, symlink -- all looked healthy from outside;
    the only thing that would have exposed any of them is the process stating its own view.
    """
    jobs = self.store.load()
    by_state: dict[str, int] = {}
    for j in jobs.values():
      by_state[j.state.value] = by_state.get(j.state.value, 0) + 1
    cells = self.available_cells_fn()
    return (f'[belief {time.strftime("%H:%M:%SZ", time.gmtime())} pid={os.getpid()}] '
            f'store={self.store.path} jobs={len(jobs)} {by_state} '
            f'cells_available={len(cells)} headroom={self.headroom_fn()}')
