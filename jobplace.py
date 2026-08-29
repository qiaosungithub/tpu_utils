"""Placement and checkpoint co-location for a resuming job.

TWO POLICIES, BOTH HONOURING allowed_metros:
  PIN       reuse the exact cell of the last attempt.
  RESELECT  re-pick from allowed_archs x allowed_metros (the default).

★RESELECT does not mean "anywhere". It means "anywhere I declared". Seven lines independently
asked for the same guarantee, because losing a metro constraint does not fail loudly: the job
builds, runs, and is then deleted mid-run by the pruner.

THE CO-LOCATION RULE (operator, 2026-08-28). Before launch, if the resume checkpoint sits in
a different metro than the chosen cell, COPY it to the co-located CNS prefix and point the
job at the copy. Never let the job itself read or write across a metro.

Measured asymmetry behind that rule:
    restore READ  cross-metro  ~6x slower;  cross-continent ~2.5x (6.0 GiB / ~14 s) -- survivable
    training WRITE cross-metro ~94x slower -> blocking saves -> duty cycle under the 0.20
                   floor -> the WIM pruner DELETES the job mid-run (no preemption, no crash)
A one-off copy costs seconds; getting it wrong costs the run, hours later, far from any
evidence. So the copy is cheap insurance rather than an optimisation.
"""

from __future__ import annotations

import dataclasses
import os
import posixpath
from typing import Any, Callable, Optional, Protocol

try:
  # Under blaze these resolve as package modules; a bare `python3 file.py` run
  # (used by the tests and by ad-hoc inspection) falls back to the flat names.
  from google3.experimental.users.qiaos.tpu_utils import jobchain as jc
except ImportError:  # pragma: no cover - flat-layout fallback
  import jobchain as jc  # type: ignore


try:
  from google3.experimental.users.qiaos.tpu_utils import cell_locality
except ImportError:                                    # pragma: no cover - test shim
  import cell_locality                                 # type: ignore


class PlacementError(Exception):
  """No cell can be chosen under this job's declared constraints."""


class CheckpointCopyError(Exception):
  """★Fail-closed. A resume whose checkpoint could not be co-located does NOT launch.

  The tempting fallback -- launch anyway, pointing at the remote path -- is how a job gets
  pruned an hour later with nothing linking the death back to this decision.
  """


@dataclasses.dataclass(frozen=True)
class Placement:
  cell: str
  metro: str
  continent: str
  bucket: str
  reason: str


class CopyFn(Protocol):
  def __call__(self, src: str, dst: str) -> None: ...


def choose(job: jc.Job, available_cells: list[str],
           now_reason: str = '') -> Placement:
  """Pick a cell. `available_cells` is the caller's live-capacity answer, already filtered
  for oversold/cooldown/price -- this function only applies the JOB's own constraints."""
  if job.placement_policy == 'PIN':
    last = job.nodes[-1] if job.nodes else None
    if last is None or not last.cell or last.cell == jc.UNKNOWN:
      raise PlacementError(
          f'{job.job_id}: placement_policy=PIN but no previous cell is recorded. The old '
          f'registry had no cell field at all, so a migrated job cannot be pinned until it '
          f'has run once under the new structure.')
    if last.cell not in available_cells:
      raise PlacementError(
          f'{job.job_id}: pinned to {last.cell}, which is not currently available. PIN means '
          f'wait for that cell, not silently go elsewhere.')
    return _describe(last.cell, f'PIN to {last.cell} (previous attempt)')

  allowed = {m.lower() for m in (job.allowed_metros or [])}
  cands = []
  for c in available_cells:
    metro = cell_locality.metro_of(c)
    if metro is cell_locality.UNKNOWN or metro == jc.UNKNOWN:
      continue          # ★unknown locality is not a candidate: fail-closed, never guess
    if allowed and metro.lower() not in allowed:
      continue
    try:
      cell_locality.storage_cell_of(c)
    except Exception:
      continue          # a cell with no co-located storage cannot host a checkpoint
    cands.append(c)

  if not cands:
    raise PlacementError(
        f'{job.job_id}: no available cell satisfies allowed_metros={job.allowed_metros}. '
        f'(Refusing rather than widening: a job that lands outside its declared metros is '
        f'deleted by the pruner mid-run, which is far more expensive than not starting.)')
  cell = cands[0]
  return _describe(cell, now_reason or f'RESELECT from {len(cands)} candidate(s)')


DEFAULT_BUCKET_SUFFIX = 'home/qiaos/eqr_data'
"""The project's own directory under a cell root.

★The shared locality layer deliberately owns WHICH CELL and never which directory, so the
suffix belongs here, with the scheduler. Keeping the split means adding a cell never touches
a project's layout, and changing a layout never risks re-pointing storage at another metro.
"""


def _describe(cell: str, reason: str,
              suffix: str = DEFAULT_BUCKET_SUFFIX) -> Placement:
  try:
    bucket = cell_locality.bucket_for(cell, suffix)
  except Exception as e:
    raise PlacementError(
        f'{cell}: no co-located CNS root is registered for its metro ({e}). Refusing rather '
        f'than falling back to a default prefix -- that fallback put a job\'s writes a '
        f'continent away and the pruner deleted it mid-run.') from e
  return Placement(cell=cell,
                   metro=str(cell_locality.metro_of(cell)),
                   continent=str(cell_locality.continent_of(cell)),
                   bucket=str(bucket),
                   reason=reason)


# --- checkpoint co-location -------------------------------------------------
def rewrite_prefix(ckpt_path: str, src_bucket: str, dst_bucket: str) -> str:
  """Swap the CNS prefix, keep the tail VERBATIM.

  ★The tail is never parsed. Four incompatible shapes exist -- `step_<N>/state/`,
  flat `step_<N>/`, `checkpoint_<N>`, and a torch `step_<N>.pt` that is a FILE -- so any
  attempt to understand the path breaks a family. We only replace a known prefix.
  """
  src_bucket = src_bucket.rstrip('/')
  dst_bucket = dst_bucket.rstrip('/')
  if not ckpt_path.startswith(src_bucket + '/'):
    raise CheckpointCopyError(
        f'{ckpt_path!r} does not live under {src_bucket!r}; refusing to guess how to '
        f'relocate it.')
  return dst_bucket + ckpt_path[len(src_bucket):]


def colocate_checkpoint(job: jc.Job, placement: Placement, *,
                        copy_fn: CopyFn,
                        exists_fn: Callable[[str], bool],
                        verify_fn: Optional[Callable[[str], bool]] = None) -> Optional[str]:
  """Ensure the resume checkpoint is in the chosen cell's metro. Returns the LOAD_FROM value.

  Returns None for a cold start (no checkpoint in the chain). Raises CheckpointCopyError if
  the checkpoint cannot be made local -- the job then does not launch.

  `verify_fn` is a DELAYED re-read, and it is not optional in production: a workspace in a
  dropped-write state returns rc=0, reads back correctly, and loses the file seconds later.
  Only a later re-read distinguishes a real write from a ghost one.
  """
  src_node = job.resume_source
  if src_node is None or not src_node.ckpt_path:
    return None                                     # cold start, nothing to co-locate

  src_path = src_node.ckpt_path
  src_metro = src_node.metro
  if src_metro == placement.metro:
    return src_path                                 # already local; no copy, no risk

  src_bucket = src_node.bucket
  if not src_bucket or src_bucket == jc.UNKNOWN:
    raise CheckpointCopyError(
        f'{job.job_id}: checkpoint {src_path!r} has no recorded bucket, so its prefix cannot '
        f'be swapped safely.')
  dst_path = rewrite_prefix(src_path, src_bucket, placement.bucket)

  if exists_fn(dst_path):
    return dst_path                                 # idempotent: a previous resume copied it

  copy_fn(src_path, dst_path)
  if not exists_fn(dst_path):
    raise CheckpointCopyError(
        f'{job.job_id}: copy of {src_path} -> {dst_path} reported success but the '
        f'destination does not exist.')
  if verify_fn is not None and not verify_fn(dst_path):
    raise CheckpointCopyError(
        f'{job.job_id}: {dst_path} passed an immediate read-back but failed the DELAYED '
        f're-read -- the destination workspace is dropping writes. Refusing to launch: '
        f'the job would start, find nothing, and silently train from scratch.')
  return dst_path


def launch_env(job: jc.Job, placement: Placement, load_from: Optional[str]) -> dict[str, str]:
  """The environment a dispatch hands to the job.

  ★LOAD_FROM is set ONLY when this dispatch is actually resuming from a checkpoint. Leaving
  it pinned across later restarts overrides the job's own auto-resume unconditionally, so
  every preemption reloads the same old checkpoint -- one run was measured falling back from
  step 380k to 298k, which reads as training instability rather than as an infra fault.

  ★CHECKPOINT_BUCKET is where the job WRITES, and it follows the compute cell. The torch
  ports derive their whole working directory from it, so it must never be repointed at some
  other job's prefix.
  """
  # ★A job's OWN declared bucket wins over the placement default. `placement.bucket` is
  # built from a project-agnostic suffix (DEFAULT_BUCKET_SUFFIX), which is right for a job
  # that never said where it writes -- and wrong for one that did. parcae declares
  # `/cns/is-d/home/qiaos/lyy_parcae_runs`; handing it `/cns/is-d/home/qiaos/eqr_data`
  # would be the same metro but ANOTHER LINE'S DIRECTORY, so its outputs would land where
  # nobody looks for them and its owner would see an empty result dir.
  #
  # The enqueue gate already guarantees the declared bucket is in an allowed metro, so
  # honouring it cannot reintroduce a cross-metro write. Found while verifying the canary:
  # the placement was correct and the destination silently was not.
  declared = None
  for k in ('bucket', 'CHECKPOINT_BUCKET', '--bucket'):
    v = (job.launch_kwargs or {}).get(k)
    if v:
      declared = str(v)
      break
  env = {'CHECKPOINT_BUCKET': declared or placement.bucket}
  if load_from:
    env['LOAD_FROM'] = load_from
  return env
