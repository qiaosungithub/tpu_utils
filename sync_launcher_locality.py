#!/usr/bin/env python3
"""Push the measured cell-locality rows into every embedded copy that needs one.

WHY COPIES EXIST AT ALL. `xm_launcher.py` is rsynced into a per-run stagedir and
built there, so it cannot import a module from `~/work/tpu_cmd`. Those files are
the documented exception to "one owner": they keep a copy, and the copy carries
`_assert_locality_matches_source()`, which diffs itself against
`cell_locality.py` at launch and REFUSES to launch on a mismatch. So a drift is
loud rather than silent -- but it still has to be repaired, and this is the
repair.

    python3 sync_launcher_locality.py            # report drift, change nothing
    python3 sync_launcher_locality.py --write    # rewrite the rows in place

WHAT IT WILL NOT DO. It only ever replaces the generated region between the
`_CELL_LOCALITY = {` / `_PERSONAL_ONLY_METROS = {...}` markers, by hunk. It
never copies a whole file: the launcher copies have diverged deliberately (a
per-project bucket suffix, project-specific bootstrap), and a wholesale copy
silently reverts those -- a mistake that has already cost this workspace a
launcher's log-mirror bootstrap.
"""
from __future__ import annotations

import argparse
import difflib
import glob
import importlib.util
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, 'cell_locality.py')

# Every file carrying an embedded copy. Globbed rather than listed so a new
# checkout is picked up automatically -- a hardcoded list is the thing that goes
# stale here.
TARGET_GLOBS = (
    os.path.expanduser('~/work/*/xm_launcher.py'),
    os.path.expanduser('~/work/tpu_cmd/xm_launcher.py'),
)
# Checkouts that are NOT ours to edit.
EXCLUDE = re.compile(r'/(lyy-work|mesh_diffusion|migrate_arms|src_snapshot|'
                     r'eqr_ss20_launchlogs|backups|archive)/')

BEGIN = '_CELL_LOCALITY = {'
END_MARKER = '_PERSONAL_ONLY_METROS = {'

# A file carrying its own `_CELL_BUCKETS` is a FORK, owned by
# sync_fork_locality.py -- that tool keeps the fork's chosen storage cells and
# writes its own wording around the same rows. Two generators editing one file
# fight forever over comment text while the DATA is identical, so ownership is
# decided here, once, by a marker in the file itself.
FORK_MARKER = '_CELL_BUCKETS = {'


def load_source():
  spec = importlib.util.spec_from_file_location('cell_locality', SOURCE)
  m = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(m)
  return m


def render_rows(cl) -> str:
  """The generated region: cell rows, metro->storage, personal-only."""
  rows = []
  for cell, row in sorted(cl._MEASURED.items(),  # pylint: disable=protected-access
                          key=lambda kv: (kv[1][1], kv[1][0], kv[0])):
    if cell.endswith('-d'):
      continue
    rows.append("    %-14s ('%s', '%s')," % (f"'{cell}':", row[0], row[1]))
  storage = ['    %-8s %r,' % (f"'{m}':", sc)
             for m, sc in sorted(cl._METRO_STORAGE_CELL.items())]  # pylint: disable=protected-access
  personal = ['    %-8s %r,' % (f"'{m}':", sc)
              for m, sc in sorted(cl._PERSONAL_ONLY_METROS.items())]  # pylint: disable=protected-access
  return (BEGIN + '\n' + '\n'.join(rows) + '\n}\n\n'
          '# metro -> the CNS cell the GROUP is registered in. One row per metro\n'
          '# because every cell in a metro shares its storage.\n'
          '_METRO_STORAGE_CELL = {\n' + '\n'.join(storage) + '\n}\n\n'
          '# Metros with NO group registration: a write here lands on the PERSONAL\n'
          '# 500 GiB per-cell ceiling. Named rather than omitted so the error can\n'
          '# say WHICH kind of "no" it is.\n'
          + END_MARKER + '\n' + '\n'.join(personal) + '\n}\n')


def rows_of(text: str) -> dict:
  """The DATA in a file's generated region: the three dicts, parsed.

  Drift is judged on parsed values, never on the surrounding prose. Two
  generators legitimately word their comments differently; only a difference in
  the rows themselves is a fault worth reporting.
  """
  ns: dict = {}
  span = region_of(text)
  if span is None:
    return {}
  exec(compile(text[span[0]:span[1]], '<region>', 'exec'), ns)  # pylint: disable=exec-used
  return {k: ns[k] for k in
          ('_CELL_LOCALITY', '_METRO_STORAGE_CELL', '_PERSONAL_ONLY_METROS')
          if k in ns}


def region_of(text: str):
  """(start, end) of the generated region, or None if this file has no copy."""
  try:
    start = text.index(BEGIN)
  except ValueError:
    return None
  try:
    tail = text.index(END_MARKER, start)
  except ValueError:
    return None
  close = text.index('\n}\n', tail) + len('\n}\n')
  return start, close


def targets():
  out = []
  for pattern in TARGET_GLOBS:
    for path in glob.glob(pattern):
      real = os.path.realpath(path)
      if EXCLUDE.search(real + '/') or real in out:
        continue
      out.append(real)
  return sorted(out)


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument('--write', action='store_true')
  args = ap.parse_args()

  cl = load_source()
  problems = cl.self_check()
  if problems:
    print('REFUSING: the source of truth fails its own self_check():', file=sys.stderr)
    for p in problems:
      print('  ' + p, file=sys.stderr)
    return 2
  want = render_rows(cl)
  want_rows = rows_of(want + '\n')

  drifted, synced, skipped, owned_elsewhere = [], [], [], []
  for path in targets():
    text = open(path).read()
    span = region_of(text)
    if span is None:
      skipped.append(path)
      continue
    if FORK_MARKER in text:
      owned_elsewhere.append(path)
      continue
    have = text[span[0]:span[1]]
    if rows_of(text) == want_rows:
      synced.append(path)
      continue
    drifted.append(path)
    print(f'--- DRIFT: {path}')
    diff = list(difflib.unified_diff(have.splitlines(), want.splitlines(),
                                     'embedded', 'source', lineterm='', n=1))
    for line in diff[:24]:
      print('   ' + line)
    if len(diff) > 24:
      print(f'   ... {len(diff) - 24} more diff lines')
    if args.write:
      tmp = path + '.tmp'
      with open(tmp, 'w') as f:
        f.write(text[:span[0]] + want + text[span[1]:])
      os.replace(tmp, path)   # atomic; never a half-written launcher
      print(f'   rewrote {path}')

  print(f'\n{len(synced)} in sync, {len(drifted)} drifted, '
        f'{len(owned_elsewhere)} owned by sync_fork_locality.py, '
        f'{len(skipped)} without an embedded copy (nothing to sync)')
  for p in owned_elsewhere:
    print(f'   fork (sync_fork_locality.py owns it): {p}')
  for p in skipped:
    print(f'   no copy: {p}')
  if drifted and not args.write:
    print('\nRe-run with --write to repair.')
    return 1
  return 0


if __name__ == '__main__':
  sys.exit(main())
