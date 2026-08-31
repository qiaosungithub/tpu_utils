#!/usr/bin/env python3
"""Patch the FORKED xm_launcher copies with measured locality + a fail-closed
bucket resolver, BY HUNK, without touching anything else in them.

WHY THESE FILES ARE DIFFERENT FROM THE ONES sync_launcher_locality.py HANDLES.
Most checkouts symlink `~/work/tpu_cmd/xm_launcher.py`, so they inherit its fix
for free. A few carry a real fork, and their forks are DELIBERATE: parcae writes
its tul checkpoints to `nm-d` where its siblings use `oi-d` -- same metro, both
correct, and rewriting one to the other would be a silent behaviour change in
somebody's running line.

So this tool does NOT unify the buckets. It adds, beside each fork's own table:
  * the measured cell -> (metro, continent) rows;
  * `_assert_locality_sane()`, which refuses on a CROSS-METRO bucket row and on
    drift from the shared snapshot;
  * a `_local_bucket()` that keeps the fork's own rows first and FAILS CLOSED
    instead of falling through to `--bucket`'s default.

NEVER A WHOLE-FILE COPY. Each edit is a bounded hunk located by an exact anchor,
because these files carry per-project bootstrap that a wholesale copy destroys.
Run without --write first and read the report.
"""
from __future__ import annotations

import argparse
import ast
import glob
import importlib.util
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, 'cell_locality.py')
BLOCK = os.path.join(HERE, '_fork_locality_block.py.in')
FUNC = os.path.join(HERE, '_fork_local_bucket.py.in')

CANONICAL = os.path.realpath(os.path.expanduser('~/work/tpu_cmd/xm_launcher.py'))
EXCLUDE = re.compile(r'/(lyy-work|mesh_diffusion|migrate_arms|src_snapshot|'
                     r'eqr_ss20_launchlogs|backups|archive)/')

OLD_FN_ANCHOR = 'def _local_bucket() -> str:\n    """The durable root nearest the cell this job will run in."""'
OLD_FN_END = '    return _BUCKET.value\n'
BLOCK_MARKER = '# MEASURED CELL LOCALITY -- shared with ~/work/tpu_cmd/google3_tpu_utils/'


def forks():
  """Real forked launchers: not symlinks to the canonical one, not excluded."""
  out = []
  for p in glob.glob(os.path.expanduser('~/work/*/xm_launcher.py')):
    real = os.path.realpath(p)
    if real == CANONICAL or EXCLUDE.search(real + '/'):
      continue
    if real not in out:
      out.append(real)
  return sorted(out)


def suffix_of(path: str) -> str:
  """The per-project path under the CNS cell root, read from its own table.

  Taken from the fork's OWN rows rather than assumed, so a project that stores
  its checkpoints somewhere else keeps doing so.
  """
  tree = ast.parse(open(path).read())
  for node in ast.walk(tree):
    if (isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == '_CELL_BUCKETS'):
      table = ast.literal_eval(node.value)
      sufs = {v.split('-d/', 1)[1] for v in table.values() if '-d/' in v}
      if len(sufs) == 1:
        return sufs.pop()
      raise ValueError(f'{path}: _CELL_BUCKETS has several suffixes {sufs}; '
                       'patch it by hand rather than guessing')
  raise ValueError(f'{path}: no _CELL_BUCKETS table found')


def embedded_rows(text: str):
  """The `_CELL_LOCALITY` rows a patched fork carries, or None."""
  m = re.search(r'^_CELL_LOCALITY = \{.*?^\}$', text, re.S | re.M)
  if not m:
    return None
  ns: dict = {}
  exec(compile(m.group(0), '<rows>', 'exec'), ns)  # pylint: disable=exec-used
  return ns['_CELL_LOCALITY']


def truth_rows():
  spec = importlib.util.spec_from_file_location('cell_locality', SOURCE)
  m = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(m)
  return {c: (r[0], r[1]) for c, r in m._MEASURED.items()  # pylint: disable=protected-access
          if not c.endswith('-d')}


def patch(path: str, write: bool) -> str:
  text = open(path).read()
  if BLOCK_MARKER in text:
    # ALREADY PATCHED IS NOT ALREADY CORRECT. Re-verify the rows against the
    # snapshot -- otherwise this tool reports "done" for a file whose data has
    # since drifted, which is the exact failure the whole change removes.
    have = embedded_rows(text)
    want = truth_rows()
    if have == want:
      return 'in sync'
    drift = [c for c in set(have or {}) | set(want)
             if (have or {}).get(c) != want.get(c)]
    if not write:
      return f'DRIFTED in {len(drift)} rows e.g. {sorted(drift)[:4]}'
    rows = '\n'.join("    %-14s ('%s', '%s')," % (f"'{c}':", want[c][0], want[c][1])
                     for c in sorted(want, key=lambda c: (want[c][1], want[c][0], c)))
    new = re.sub(r'^_CELL_LOCALITY = \{.*?^\}$',
                 '_CELL_LOCALITY = {\n' + rows + '\n}', text, count=1, flags=re.S | re.M)
    ast.parse(new)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
      f.write(new)
    os.replace(tmp, path)
    return f're-synced {len(drift)} rows'
  if OLD_FN_ANCHOR not in text:
    return 'SKIP: _local_bucket does not match the known shape (patch by hand)'
  suffix = suffix_of(path)
  block = open(BLOCK).read()
  func = open(FUNC).read()

  start = text.index(OLD_FN_ANCHOR)
  end = text.index(OLD_FN_END, start) + len(OLD_FN_END)
  new = (text[:start]
         + f"# The path under a CNS cell root this project uses, read from its own\n"
           f"# _CELL_BUCKETS rows so a derived bucket lands in the same place.\n"
           f"_BUCKET_SUFFIX_FOR_DERIVED = {suffix!r}\n\n\n"
         + func + text[end:])
  # The locality block goes immediately BEFORE the (new) function, after the
  # project's own _CELL_BUCKETS table so it can validate it.
  anchor = '_BUCKET_SUFFIX_FOR_DERIVED = '
  i = new.index(anchor)
  new = new[:i] + block.lstrip('\n') + '\n\n' + new[i:]
  try:
    ast.parse(new)
  except SyntaxError as e:
    return f'REFUSED: patched file would not parse ({e})'
  if write:
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
      f.write(new)
    os.replace(tmp, path)
    return f'patched (suffix {suffix!r})'
  return f'would patch (suffix {suffix!r})'


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument('--write', action='store_true')
  args = ap.parse_args()
  rc = 0
  for path in forks():
    verdict = patch(path, args.write)
    print(f'{verdict:34s} {path}')
    if verdict.startswith(('SKIP', 'REFUSED')):
      rc = 1
  return rc


if __name__ == '__main__':
  sys.exit(main())
