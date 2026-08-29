"""Persistence for the job chain: field-level writes under optimistic concurrency.

WHY THIS IS NOT THE OLD STORE. The previous store's write path took the lock correctly and
was still the fleet's biggest source of corruption, because correctness of the LOCK is not
correctness of the WRITE. A pass would:

    t0   acquire lock, read all 125 rows, release lock
    t0   go and build for twenty minutes
    t20  acquire lock, write back the 125 rows it read at t0, release lock

Every step is lock-legal. The damage is that the t20 write carries a t0 world-view, so any
edit made in between -- by a human, by another pass, by the job itself -- is silently undone.
Measured: 110 of 125 rows overwritten per pass, a RUNNING job's xid erased (then rebuilt: five
8-card B200s from one row), deleted rows resurrected nine minutes later, an attempts counter
going backwards.

Holding the lock across the twenty minutes is not the fix -- that freezes the queue for
everyone. The fix is two changes to what a write MEANS:

  1. FIELD-LEVEL PATCHES. A writer declares `job_id -> {field: value}`. It cannot express
     "and everything else stays as I remember it", so it cannot assert something it did not
     observe. Fields nobody named are not touched, by construction rather than by care.

  2. OPTIMISTIC CONCURRENCY (CAS). Every record carries a `version`. A patch states the
     version it was computed from; if the record has moved on, the write is REFUSED and the
     caller must re-read and re-decide. A twenty-minute-old decision is exactly the decision
     that should not land unexamined.

Both are needed. (1) alone still lets a stale writer clobber the specific fields it names;
(2) alone still forces whole-record rewrites. Together, a stale writer can only fail loudly.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import tempfile
import time
from typing import Any, Callable, Iterable, Optional

try:
  # Under blaze these resolve as package modules; a bare `python3 file.py` run
  # (used by the tests and by ad-hoc inspection) falls back to the flat names.
  from google3.experimental.users.qiaos.tpu_utils import jobchain as jc
except ImportError:  # pragma: no cover - flat-layout fallback
  import jobchain as jc  # type: ignore



SCHEMA_VERSION = 1


class StaleWriteError(Exception):
  """A patch was computed from a version that is no longer current (I1).

  This is a NORMAL outcome, not a fault: it means the world moved while you were thinking.
  The caller re-reads and re-decides. It is deliberately loud -- the old failure mode was
  precisely that this situation was silent and won.
  """

  def __init__(self, job_id: str, expected: int, actual: int):
    super().__init__(
        f'{job_id}: stale write refused (I1): patch computed from version {expected}, '
        f'record is now at {actual}. Re-read and re-decide -- do not retry blindly.')
    self.job_id, self.expected, self.actual = job_id, expected, actual


@contextlib.contextmanager
def _flock(path: str):
  """Cross-process exclusive lock on a SIDECAR file.

  Never on the store itself: the store is swapped in via os.replace, so a lock taken on its
  inode is silently lost at the rename. Hold this only across a read-modify-write, never
  across build/RPC work.
  """
  lock_path = path + '.lock'
  with open(lock_path, 'a+') as fh:
    fcntl.flock(fh, fcntl.LOCK_EX)
    try:
      yield
    finally:
      fcntl.flock(fh, fcntl.LOCK_UN)


class JobStore:
  """The queue. All mutation goes through `patch`; there is no whole-file write API.

  The absence of a `save_all()` is deliberate and is the main structural defence: the shape
  of the old bug was not reachable through a narrow API, it was reachable because a wide one
  existed and looked reasonable at the call site.
  """

  def __init__(self, path: str):
    self.path = path

  # --- reads --------------------------------------------------------------
  def load(self) -> dict[str, jc.Job]:
    """Read all jobs. Callers get plain objects; the store keeps no in-memory cache.

    No cache, on purpose: a long-lived reader holding a private copy of shared state is the
    same bug in a different carrier -- tonight it appeared as a queue rolled back by a stale
    snapshot, as bash not re-reading an already-parsed function, and as a Python process
    serving a 24-hour-old import while the fix sat on disk for 63 minutes.
    """
    if not os.path.exists(self.path):
      return {}
    with open(self.path) as fh:
      raw = json.load(fh)
    return {j['job_id']: jc.Job.from_dict(j) for j in raw.get('jobs', [])}

  def get(self, job_id: str) -> Optional[jc.Job]:
    return self.load().get(job_id)

  # --- writes -------------------------------------------------------------
  def create(self, job: jc.Job, *, top_level_count: Optional[int] = None,
             stagedir_root: Optional[str] = None) -> jc.Job:
    """Enqueue a new job. Runs the full enqueue gate (forbidden archs, workdir, identity)."""
    jc.validate_enqueue(job, top_level_count=top_level_count, stagedir_root=stagedir_root)
    now = time.time()
    with _flock(self.path):
      jobs = self.load()
      if job.job_id in jobs:
        raise jc.RejectedAtEnqueue(f'{job.job_id}: already exists')
      job.version, job.enqueued_at, job.updated_at = 0, now, now
      jc.check_transition(None, job)
      jobs[job.job_id] = job
      self._write_all(jobs)
    return job

  def patch(self, job_id: str, base_version: int, fields: dict[str, Any]) -> jc.Job:
    """★The only mutation path. Apply named FIELDS to one job, if it is still at
    `base_version`.

    Raises StaleWriteError if the record moved (I1), InvariantError if the result would
    corrupt the chain. Fields not named are untouched -- the caller has no way to express an
    opinion about them, which is the point (I2).
    """
    with _flock(self.path):
      jobs = self.load()
      cur = jobs.get(job_id)
      if cur is None:
        raise KeyError(f'{job_id}: no such job')
      if cur.version != base_version:
        raise StaleWriteError(job_id, base_version, cur.version)
      new = jc.Job.from_dict(cur.to_dict())
      for k, v in fields.items():
        if not hasattr(new, k):
          raise AttributeError(f'{job_id}: no such field {k!r}')
        setattr(new, k, v)
      new.version = cur.version + 1
      new.updated_at = time.time()
      jc.check_transition(cur, new)
      jobs[job_id] = new
      self._write_all(jobs)
      return new

  def commit(self, updated: jc.Job) -> jc.Job:
    """Persist a job produced by a jobchain transition (open_attempt, record_failure, ...).

    Those helpers already bumped `version`, so the CAS base is `updated.version - 1`.
    """
    with _flock(self.path):
      jobs = self.load()
      cur = jobs.get(updated.job_id)
      if cur is None:
        raise KeyError(f'{updated.job_id}: no such job')
      if cur.version != updated.version - 1:
        raise StaleWriteError(updated.job_id, updated.version - 1, cur.version)
      jc.check_transition(cur, updated)
      jobs[updated.job_id] = updated
      self._write_all(jobs)
      return updated

  def remove(self, job_id: str, base_version: int) -> None:
    """Delete a job. Requires the CAS base, so a stale deleter is refused like any other.

    ★Deletion needs this more than modification does. A modification that loses a race is
    re-applied on the next round because the writer re-reads and sees the new value; a
    deletion removes the very channel that would tell the other writer it happened, so it
    never converges on its own. Rows deleted under a lock with triple assertions came back
    nine minutes later, from a snapshot taken before the delete.
    """
    with _flock(self.path):
      jobs = self.load()
      cur = jobs.get(job_id)
      if cur is None:
        return
      if cur.version != base_version:
        raise StaleWriteError(job_id, base_version, cur.version)
      del jobs[job_id]
      self._write_all(jobs)

  # --- internals ----------------------------------------------------------
  def _write_all(self, jobs: dict[str, jc.Job]) -> None:
    """Atomic replace. MUST be called with the lock held."""
    payload = {'schema_version': SCHEMA_VERSION,
               'jobs': [j.to_dict() for j in jobs.values()]}
    d = os.path.dirname(self.path) or '.'
    fd, tmp = tempfile.mkstemp(dir=d, prefix=os.path.basename(self.path) + '.tmp.')
    try:
      with os.fdopen(fd, 'w') as fh:
        json.dump(payload, fh, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
      os.replace(tmp, self.path)
    except BaseException:
      with contextlib.suppress(OSError):
        os.unlink(tmp)
      raise
