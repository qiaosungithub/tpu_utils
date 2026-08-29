"""One-shot migration: the old flat queue -> the job chain.

Operator decision (2026-08-28): no legacy compatibility, and the old records are ARCHIVED
rather than converted wholesale. So this is not a format translator that must preserve
everything -- it is a filter whose default answer is "archive, do not migrate".

WHY THE DEFAULT IS "DROP". Of the 126 rows in the shutdown snapshot, most were artefacts of
the bugs being fixed: 15 shells with workdir=/tmp and a single launch_kwarg (a requeue path
that rebuilt context from nothing), rows whose workdir was a whole depot root, five-day-old
SUBMITTEDs whose experiments no longer exist in XM, and 60+ terminal FAILEDs. Carrying those
forward would import the bugs' output into the structure meant to prevent them. A row is
migrated only if it is both LIVE and INTACT; everything else is written to an archive file
with the reason, so nothing is lost, it is merely not resurrected.

★A migration is also the last moment where a bad row is cheap to stop. Rows are put through
the same `validate_enqueue` gate as a fresh submission, so a workdir that would flood srcfs
cannot re-enter the queue by having been there before.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

try:
  # Under blaze these resolve as package modules; a bare `python3 file.py` run
  # (used by the tests and by ad-hoc inspection) falls back to the flat names.
  from google3.experimental.users.qiaos.tpu_utils import jobchain as jc
  from google3.experimental.users.qiaos.tpu_utils import jobstore as js
except ImportError:  # pragma: no cover - flat-layout fallback
  import jobchain as jc  # type: ignore
  import jobstore as js  # type: ignore


Node_T = jc.Node


class Verdict:
  MIGRATE = 'MIGRATE'
  ARCHIVE = 'ARCHIVE'


# States in the old queue that mean "this row is finished with".
_OLD_TERMINAL = {'FAILED', 'CANCELLED', 'COMPLETED', 'DONE'}


def classify(row: dict[str, Any], *, live_xids: Optional[set[str]] = None) -> tuple[str, str]:
  """Decide MIGRATE / ARCHIVE for one old row, with the reason.

  `live_xids` is XM's answer, not ours. When it is None we do not guess that a SUBMITTED row
  is alive -- 12 of them were five days dead with their experiments purged, and counting
  'has an xid' as 'ran successfully' is exactly the error that made them look healthy.
  """
  jid = row.get('job_id', '?')
  state = str(row.get('state', ''))
  workdir = row.get('workdir') or ''
  lk = row.get('launch_kwargs') or {}
  xid = row.get('xid')

  if state in _OLD_TERMINAL:
    return Verdict.ARCHIVE, f'terminal in the old queue ({state})'

  # An enforcer shell: context was rebuilt from nothing, so there is nothing to carry.
  if list(lk.keys()) == ['resume_xid']:
    return Verdict.ARCHIVE, (
        'enforcer shell: launch_kwargs holds only resume_xid, so config/bucket/exp_name are '
        'gone. It would cold-start AND be priced at a tenth of reality. Re-enqueue from the '
        'original definition instead.')
  if workdir == '/tmp':
    return Verdict.ARCHIVE, 'workdir=/tmp: no config.sh, would die as a launcher FATAL'

  if xid:
    if live_xids is None:
      return Verdict.ARCHIVE, (
          f'holds xid {xid} but XM was not consulted; refusing to guess whether it is alive. '
          f'Re-enqueue or re-attach explicitly.')
    if xid not in live_xids:
      return Verdict.ARCHIVE, f'xid {xid} is not live in XM (purged or finished)'
    return Verdict.MIGRATE, f'live in XM as {xid}'

  if not workdir:
    return Verdict.ARCHIVE, 'no workdir recorded'
  return Verdict.MIGRATE, f'live, no xid ({state})'


_CKPT_KEYS = ('load_from', 'load_model_path', 'config.load_from', 'CONFIG_LOAD_FROM')
"""The keys different families use to name a checkpoint. ★All of them mean the same thing to
the scheduler: an opaque string to replay. We recognise the KEY; we never read the value."""

_UNKNOWN_TIME = jc.UNKNOWN_TIME
"""Alias for jobchain.UNKNOWN_TIME -- one definition, see there."""

_XID_IN_PATH = __import__('re').compile(r'xid[_-](\d{6,})')


def declared_checkpoint(launch_kwargs: dict[str, Any]) -> Optional[str]:
  """The checkpoint a legacy row declared, VERBATIM, or None."""
  for k in _CKPT_KEYS:
    v = (launch_kwargs or {}).get(k)
    if v:
      return str(v)
  return None


def seed_node_for(row: dict[str, Any], ckpt: str) -> Node_T:
  """★A node recording the attempt that PRODUCED this checkpoint -- not a fabricated attempt
  of this job.

  WHY THIS EXISTS. The new resume path reads `job.resume_source`, which walks `nodes` looking
  for a ckpt_path; it never reads launch_kwargs. So a migrated row that carried its checkpoint
  only in launch_kwargs would have been faithfully preserved AND completely ignored -- every
  such job restarting from step 0. Measured on the live queue: 29 rows would have cold-started
  that way, including runs at step 518070 and step 598000.

  ★The lesson is narrower than "carry the data": the string WAS carried, byte for byte, into a
  field nobody reads. Preserving a value and preserving its meaning are different jobs, and
  only reading the CONSUMER tells you which one you did.

  The producing xid is recoverable from the path itself (`.../xid_284061100_.../step_20000`),
  so the node is evidence, not invention. When it is not recoverable we say UNKNOWN rather
  than inventing one -- the checkpoint still works, only PIN placement is unavailable.
  """
  m = _XID_IN_PATH.search(ckpt)
  return Node_T(
      xid=m.group(1) if m else jc.UNKNOWN,
      cell=jc.UNKNOWN, metro=jc.UNKNOWN, continent=jc.UNKNOWN,
      bucket=_bucket_prefix_of(ckpt),
      ckpt_path=ckpt,                      # ★verbatim, never parsed or normalised
      outcome=jc.Outcome.COMPLETED,
      # ★NOT 0.0. A zero timestamp passes `is not None` and fails a truthiness test, so two
      # readers disagree about whether this attempt ever ended -- monitor-v48 and I scanned
      # the same file for exactly this inconsistency and got 1 vs 0. Epoch zero is also a
      # real-looking date (1970), which is worse than an obvious absence.
      # A seed node describes an attempt that finished before this scheduler existed: we know
      # THAT it ended, not WHEN. `_UNKNOWN_TIME` says so without lying about a date.
      started_at=_UNKNOWN_TIME, ended_at=_UNKNOWN_TIME,
  )


def _bucket_prefix_of(ckpt: str) -> str:
  """`/cns/<cell>-d/<user>/<project>` -- the prefix `colocate_checkpoint` swaps on a
  cross-metro resume.

  ★A leading empty segment: `'/cns/oi-d/home/qiaos/eqr_data/...'.split('/')` starts with `''`,
  so the four path components end at index 5, not 4. An earlier version took `[:5]` and
  returned `/cns/oi-d/home/qiaos`, dropping the project segment. Nothing failed loudly --
  `colocate_checkpoint` only asserts `ckpt.startswith(bucket + '/')`, which a shorter prefix
  satisfies -- so the swap spliced two different depths together and produced
  `/cns/is-d/home/qiaos/eqr_data/eqr_data/logs/...`. The job would then find nothing at that
  path and silently train from step 0: the very failure the seed node exists to prevent,
  relocated into the cross-metro branch.

  ★And it was invisible locally: `colocate_checkpoint` returns early when the checkpoint is
  already in the placement's metro, so only a cross-metro resume ever evaluates this.

  Returns UNKNOWN for anything that is not a /cns/ path with a project segment -- an UNKNOWN
  bucket makes a cross-metro resume refuse loudly (CheckpointCopyError), which is the correct
  direction: same-metro resume still works, and nobody silently loses their progress.
  """
  parts = (ckpt or '').split('/')
  # ['', 'cns', '<cell>-d', '<user>', '<project>', ...] -- 5 leading entries, 4 components.
  if len(parts) >= 6 and parts[0] == '' and parts[1] == 'cns':
    return '/'.join(parts[:5 + 1])
  return jc.UNKNOWN


def _looks_like_eval(launch_kwargs: dict[str, Any]) -> bool:
  """MIGRATION ONLY: recover the eval flag from what the submitter wrote.

  ★Not a general rule -- new submissions must state `is_eval` explicitly. This exists
  because legacy rows predate the field, and reading `config`/`exp_name` (which carry
  meaning) is strictly better than reading `job_id` (which is a random identifier).
  A row this misses is refused at the gate and lands in the manual list, which is the
  safe direction: a missed eval costs one human decision, a missed TRAINING job on
  BATCH is silently starved and still billed.
  """
  hay = ' '.join(str(v) for v in (launch_kwargs or {}).values()).lower()
  return 'eval' in hay


def read_identity(workdir: str) -> tuple[Optional[str], Optional[str]]:
  """Recover (project_name, target_label) from the job's OWN checkout.

  ★Note which config.sh this is. The cross-wiring incident came from the config.sh inside a
  shared STAGEDIR, which a wrapper backfills from a global default when a ghost write drops
  it -- so that copy can belong to whoever staged last. This one lives in the job's own
  workdir, is the file its owner edits, and is precisely the thing the stagedir copy is
  supposed to match. Reading it here is what later makes a mismatch detectable: without a
  recorded expectation there is nothing to compare the staged copy against.

  Returns (None, None) when the file is absent or has no fingerprint -- never a guess.
  """
  import re
  cfg = os.path.join(workdir or '', 'config.sh')
  if not os.path.isfile(cfg):
    return None, None
  try:
    text = open(cfg, errors='ignore').read()
  except OSError:
    return None, None
  pn = re.search(r'^export PROJECT_NAME="([^"]*)"', text, re.M)
  tl = re.search(r'^export TARGET_LABEL="([^"]*)"', text, re.M)
  return (pn.group(1) if pn and pn.group(1) else None,
          tl.group(1) if tl and tl.group(1) else None)


def to_job(row: dict[str, Any], now: Optional[float] = None) -> jc.Job:
  """Build a Job from an old row. Everything unknown becomes UNKNOWN, never a plausible guess.

  The old queue never recorded the identity fingerprints, so they are recovered from the
  job's own `workdir/config.sh` (see `read_identity`). A row whose checkout cannot supply
  them is refused by `validate_enqueue` -- the right outcome, since the whole point of the
  fingerprint is to be an expectation stated in advance, and one the scheduler invented
  would compare a staged package against itself.
  """
  now = time.time() if now is None else now
  lk = dict(row.get('launch_kwargs') or {})
  pn, tl = read_identity(row.get('workdir') or '')
  j = jc.Job(
      job_id=row['job_id'],
      workdir=row.get('workdir') or '',
      target_label=row.get('target_label') or tl or jc.UNKNOWN,
      project_name=row.get('project_name') or pn or jc.UNKNOWN,
      launch_kwargs=lk,
      placement_policy=row.get('placement_policy') or 'RESELECT',
      allowed_metros=row.get('allowed_metros'),
      allowed_archs=row.get('allowed_archs'),
      power=row.get('power') or '',
      tier=row.get('tier') or 'PROD',
      # ★is_eval is DECLARED, never inferred from a name. Legacy rows have no such
      # field, so it is recovered from the config/exp_name the submitter actually
      # wrote -- and a BATCH row that cannot evidence it is refused by the gate and
      # surfaces for a human, rather than being quietly promoted.
      is_eval=bool(row.get('is_eval')) or _looks_like_eval(lk),
      topology_locked=bool(row.get('topology_locked')),
      priority=int(row.get('priority') or 0),
      enqueued_at=float(row.get('submitted_at') or now),
      updated_at=now,
  )
  # ★attempts is NOT carried across. The old counter conflated environment refusals with
  # build defects (that is the bug), so an att=65 row would arrive pre-condemned under a
  # taxonomy it was never scored by. Its history is in the archive; its strike count restarts.
  j.build_attempts = 0
  ckpt = declared_checkpoint(lk)
  if ckpt:
    # ★Put the checkpoint where the RESUME PATH looks for it, not merely where the old row
    # happened to keep it. See seed_node_for().
    j.nodes.append(seed_node_for(row, ckpt))
  j.last_reason = f'migrated from the legacy queue (was {row.get("state")}, ' \
                  f'att={row.get("attempts")}); strike count reset under the new taxonomy'
  return j


def migrate(old_path: str, store: js.JobStore, archive_path: str, *,
            live_xids: Optional[set[str]] = None, dry_run: bool = True) -> dict[str, Any]:
  """Run the migration. Returns a report; writes nothing when dry_run.

  Every row lands in exactly one of three lists -- migrated, archived, or rejected -- and the
  three always sum to the input count. A row that silently vanishes between formats is the
  failure mode this whole rewrite exists to stop, so the accounting is asserted, not assumed.
  """
  with open(old_path) as fh:
    raw = json.load(fh)
  rows = raw['entries'] if isinstance(raw, dict) and 'entries' in raw else raw

  report: dict[str, Any] = {'total': len(rows), 'migrated': [], 'archived': [], 'rejected': []}
  archive_rows = []

  for row in rows:
    jid = row.get('job_id', '?')
    verdict, why = classify(row, live_xids=live_xids)
    if verdict is Verdict.ARCHIVE or verdict == Verdict.ARCHIVE:
      report['archived'].append((jid, why))
      archive_rows.append({'row': row, 'reason': why})
      continue
    job = to_job(row)
    try:
      if not dry_run:
        store.create(job)
      else:
        jc.validate_enqueue(job)
      report['migrated'].append((jid, why))
    except (jc.RejectedAtEnqueue, jc.InvariantError) as e:
      # ★Rejected is distinct from archived: the row LOOKED live, and the gate stopped it.
      # Surfacing that separately is the point -- it is the list a human must act on.
      report['rejected'].append((jid, str(e)))
      archive_rows.append({'row': row, 'reason': f'REJECTED BY GATE: {e}'})

  assert (len(report['migrated']) + len(report['archived']) + len(report['rejected'])
          == report['total']), 'row accounting does not balance'

  if not dry_run:
    with open(archive_path, 'w') as fh:
      json.dump({'archived_at': time.time(), 'source': old_path, 'rows': archive_rows},
                fh, indent=1)
  return report


def format_report(r: dict[str, Any]) -> str:
  out = [f'total={r["total"]}  migrate={len(r["migrated"])}  '
         f'archive={len(r["archived"])}  rejected={len(r["rejected"])}', '']
  if r['rejected']:
    out.append('REJECTED BY THE ENQUEUE GATE (a human must re-state these):')
    out += [f'  {jid}: {why}' for jid, why in r['rejected']] + ['']
  if r['migrated']:
    out.append('MIGRATED:')
    out += [f'  {jid}: {why}' for jid, why in r['migrated']] + ['']
  if r['archived']:
    from collections import Counter
    out.append('ARCHIVED (kept in the archive file, not resurrected):')
    for why, n in Counter(w for _, w in r['archived']).most_common():
      out.append(f'  {n:>3}x {why[:100]}')
  return '\n'.join(out)
