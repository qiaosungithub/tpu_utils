"""Entry point for the new dispatcher: `python3 jobd.py --once` or `--loop`.

Deliberately a thin wiring layer -- policy lives in the modules it imports. What it DOES own
is the startup contract:

  * it prints what it BELIEVES before doing anything (C11), so a stale view is visible from
    outside rather than only in its effects;
  * it refuses to run against the OLD queue file, because the old scheduler's daemon reads
    that one and two dispatchers on two files is the most expensive shape available;
  * `--once --dry-run` is the default, so an accidental invocation plans and explains instead
    of spending.
"""

from __future__ import annotations

import os
import sys
import time

from absl import app
from absl import flags

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/google/src/cloud/qiaos/run_amply_workspace')

try:
  from google3.experimental.users.qiaos.tpu_utils import jobbuild as jb
  from google3.experimental.users.qiaos.tpu_utils import jobchain as jc
  from google3.experimental.users.qiaos.tpu_utils import jobcost
  from google3.experimental.users.qiaos.tpu_utils import jobdispatch as jd
  from google3.experimental.users.qiaos.tpu_utils import jobplace as jp
  from google3.experimental.users.qiaos.tpu_utils import jobstore as js
except ImportError:  # pragma: no cover - flat-layout fallback for ad-hoc runs
  import jobbuild as jb          # type: ignore
  import jobchain as jc          # type: ignore
  import jobcost                 # type: ignore
  import jobdispatch as jd       # type: ignore
  import jobplace as jp          # type: ignore
  import jobstore as js          # type: ignore

DEFAULT_STORE = os.path.expanduser('~/.tpu_jobs_v2.json')
LEGACY_QUEUE = os.path.expanduser('~/.tpu_local_queue.json')


def _available_cells() -> list[str]:
  """Live capacity, via the router's own AvailabilityProvider (one RPC per call).

  ★Fail-closed: any error yields an EMPTY list, so the round dispatches nothing. The
  tempting alternative -- treat "could not ask" as "everything is available" -- is the same
  shape as pricing a job at zero when the table is unreadable, and it fails in the expensive
  direction.
  """
  try:
    from google3.experimental.users.qiaos.tpu_utils import avail_provider
    prov = avail_provider.AvailabilityProvider(group='9')
    avail, _arch_price, _cell_price = prov.fetch()
    # ★The provider keys availability by "<cell>|<arch>" (e.g. "dl|v4"), not by cell: the
    # same cell offers different families independently. Passing the composite key straight
    # through made every metro lookup return UNKNOWN and every job refuse -- a total outage
    # that presented as "no capacity anywhere", which is exactly what a real shortage looks
    # like. Split it, and de-duplicate.
    out = set()
    for key, ca in avail.items():
      if not getattr(ca, 'placeable', True):
        continue
      out.add(key.split('|', 1)[0])
    return sorted(out)
  except Exception as e:                                  # noqa: BLE001
    print(f'[jobd] availability unavailable ({e}); dispatching nothing this round')
    return []


def _headroom() -> float | None:
  """Credit headroom from budget_check --query (XM-truth; the router never recomputes cost).

  ★None means UNKNOWN and the dispatcher spends nothing. Never 0 (which also means "genuinely
  full") and never a large default -- both are real-looking numbers that a reader cannot tell
  from a measurement.
  """
  import json
  import subprocess
  script = os.path.expanduser('~/work/wiki_agents/tools/budget_check.py')
  if not os.path.isfile(script):
    return None
  # ★NOT sys.executable: inside a PAR that is the PAR ITSELF, not an
  # interpreter, so this line re-invoked jobd with budget_check.py as a
  # positional arg. absl then parsed --query against JOBD's flag set, failed
  # with "Unknown command line flag 'query'", and exited 1 before main() --
  # every single time. _headroom saw no '{' line and returned None, so the
  # budget gate was permanently blind while looking permanently safe.
  # ★The tell for this whole family: an absl flag error naming a flag that
  # belongs to the HELPER, not to the caller.
  interp = '/usr/bin/python3'
  if not os.path.isfile(interp):
    return None                      # fail closed; never a plausible default
  try:
    p = subprocess.run([interp, script, '--query', 'v6p-32', 'PROD'],
                       capture_output=True, text=True, timeout=90)
    for line in (p.stdout or '').splitlines():
      line = line.strip()
      if line.startswith('{'):
        return float(json.loads(line).get('headroom'))
  except Exception:                                       # noqa: BLE001
    return None
  return None


_STORE = flags.DEFINE_string('store', DEFAULT_STORE, 'Path to the v2 job store.')
_LOOP = flags.DEFINE_bool('loop', False, 'Run forever instead of a single round.')
_POLL_S = flags.DEFINE_integer('poll_s', 120, 'Seconds between rounds under --loop.')
_LIMIT = flags.DEFINE_integer(
    'limit', 1, 'Dispatches per round. Builds are serial on this host, and a concurrent '
    'build was the original source of zombie XIDs -- raise this only deliberately.')
_DRY_RUN = flags.DEFINE_bool(
    'dry_run', True, 'Plan and explain without submitting. ★Defaults TRUE so an accidental '
    'invocation costs nothing; pass --nodry_run to actually dispatch.')


_EXPLAIN = flags.DEFINE_bool(
    'explain', False,
    'Read-only: print what each queued job would do and why, and dispatch NOTHING. This is '
    'the answer to "why is my job not going out" -- a question that previously required '
    'reading a daemon lane\'s stdout through an unbounded pipe.')


def _explain(store, cells):
  """Why each job is or is not dispatchable, right now."""
  from collections import Counter
  print(f'[explain] {len(cells)} placeable cells')
  by_metro = Counter()
  for c in cells:
    try:
      by_metro[str(jp.cell_locality.metro_of(c))] += 1
    except Exception:                                     # noqa: BLE001
      by_metro['<unknown>'] += 1
  print('[explain] cells by metro: ' +
        ', '.join(f'{m}={n}' for m, n in sorted(by_metro.items(), key=lambda x: -x[1])[:12]))
  print('[explain] sample cell keys: ' + ', '.join(sorted(cells)[:12]))
  for job in sorted(store.load().values(), key=lambda j: j.job_id):
    if job.state not in (jc.JobState.QUEUED, jc.JobState.DEFERRED):
      continue
    # ★Check the SAME gates the dispatcher checks, in the same order. An --explain that
    # only reports placement would tell someone their job is fine while the dispatcher
    # holds it -- a diagnostic that disagrees with the thing it describes is worse than
    # none, because it is believed.
    try:
      jc.validate_enqueue(job)
      pl = jp.choose(job, cells)
      # ★Show the bucket the job will ACTUALLY write to, not the placement default. A
      # declared bucket overrides it (jobplace.launch_env), and --explain exists precisely
      # so someone can see the destination before the job runs -- printing the wrong one
      # here would hide the very mistake this flag is for.
      env = jp.launch_env(job, pl, None)
      verdict = (f'OK -> {pl.cell} ({pl.metro}/{pl.continent}) '
                 f'writes={env["CHECKPOINT_BUCKET"]}')
    except jc.RejectedAtEnqueue as e:
      verdict = f'HELD-AT-DISPATCH: {e}'
    except Exception as e:                                # noqa: BLE001
      verdict = f'BLOCKED: {e}'
    print(f'  {job.job_id:<24} metros={job.allowed_metros} archs={job.allowed_archs}')
    print(f'      {verdict}')


def main(argv):
  del argv

  if os.path.realpath(_STORE.value) == os.path.realpath(LEGACY_QUEUE):
    print('[jobd] REFUSING to run against the legacy queue file. The old daemon reads that '
          'file; two dispatchers over two formats is how one config becomes five jobs.')
    return 2

  store = js.JobStore(_STORE.value)
  builder = jb.TpuQueueBuilder(dry_run=_DRY_RUN.value, log=print)
  d = jd.Dispatcher(
      store=store, builder=builder,
      available_cells_fn=_available_cells,
      headroom_fn=_headroom,
      copy_fn=lambda s, t: None,      # wired in once a real cross-metro resume is due
      exists_fn=lambda p: False,
      identity_fn=jb.staged_identity,
      log=print)

  print(f'[jobd] start pid={os.getpid()} store={_STORE.value} '
        f'dry_run={_DRY_RUN.value} limit={_LIMIT.value}')
  print('[jobd] ' + d.belief_line())
  print('[jobd] ' + jobcost.belief_report('v6p-32', 'PROD'))

  if _EXPLAIN.value:
    _explain(store, _available_cells())
    return 0

  while True:
    for r in d.round(limit=_LIMIT.value):
      print(f'[jobd] {r.job_id}: {r.action} -- {r.detail}')
    if not _LOOP.value:
      return 0
    time.sleep(_POLL_S.value)
    print('[jobd] ' + d.belief_line())   # ★every round, so disagreement is detectable


if __name__ == '__main__':
  # ★app.run, not a bare main(): it performs InitGoogle(), without which the RPC stack
  # CHECK-fails the moment availability is queried.
  app.run(main)
