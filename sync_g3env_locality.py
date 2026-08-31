#!/usr/bin/env python3
"""Widen each training checkout's `_CELL_TO_METRO` to the MEASURED snapshot.

THE BUG THIS FIXES. `utils/g3_env.py` is already fail-closed -- an unlisted cell
raises "Refusing to guess" rather than defaulting -- and every row it has is
CORRECT. Its problem is the rows it does not have: the router can pin a cell the
binary has never heard of, and the job then crash-loops at startup. XID
284266707 did exactly that on `yutulrf` (a real tul cell, co-located with the
data) 815 times. Fail-closed was the right behaviour and it still cost a run,
because the map was narrower than the fleet.

So this does NOT change the failure MODE -- an unknown cell still raises. It
removes the cells that should never have been unknown, by replacing the
hand-written table with the measured one.

WHAT IT DELIBERATELY DOES NOT TOUCH:
  * `_METRO_TO_REGION` -- a metro is listed there only when its GCP region has
    been read out of slicer_metros.pi, and that gate is what stops a guessed
    region reaching a dataset guard;
  * `_METRO_TO_CNS_CELLS` -- entry there means the GROUP has flex-registered
    storage in that metro, which is a quota fact, not a geography fact;
  * the second gate in `train.py::_init_run`, which asserts the resolved zone
    is one this project has data in.
Widening cell->metro therefore turns "Cannot determine where this task is
running" into a precise "no dataset replica in metro X" -- a better error, not a
weaker one.

    python3 sync_g3env_locality.py            # report, change nothing
    python3 sync_g3env_locality.py --write    # rewrite the table in place
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
TARGETS = (os.path.expanduser('~/work/*/utils/g3_env.py'),)
EXCLUDE = re.compile(r'/(lyy-work|migrate_arms|src_snapshot|backups|archive)/')

BEGIN = '_CELL_TO_METRO = {'
HEADER = '''# cell -> metro, MEASURED with `mach_locality -k metro <cell>` and regenerated
# by ~/work/tpu_cmd/google3_tpu_utils/sync_g3env_locality.py from the shared
# snapshot in google3_tpu_utils/cell_locality.py. Do not hand-edit rows: a cell
# missing here is a startup crash for any job the router pins there (XID
# 284266707 crash-looped 815 times on `yutulrf`, a real tul cell this table did
# not list), and a cell guessed here is a silent cross-region read.
#
# AN UNLISTED CELL IS STILL AN ERROR, NOT A DEFAULT -- that part was always
# right. What changed is that the list is now as wide as the fleet the
# scheduler can actually place into, so "unlisted" means "genuinely new",
# not "nobody got round to adding it".
#
# Being in this table says only WHERE a cell is. Whether this project may RUN
# there is decided downstream by _METRO_TO_REGION (a verified GCP region) and
# _METRO_TO_CNS_CELLS (group storage), both of which stay hand-curated.
'''


def load_source():
  spec = importlib.util.spec_from_file_location('cell_locality', SOURCE)
  m = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(m)
  return m


def render(cl) -> str:
  by_metro: dict[str, list[str]] = {}
  for cell, row in cl._MEASURED.items():  # pylint: disable=protected-access
    if cell.endswith('-d'):
      continue
    by_metro.setdefault(row[0], []).append(cell)
  lines = [HEADER + BEGIN]
  for metro in sorted(by_metro):
    cont = cl.continent_of(by_metro[metro][0])
    lines.append(f'    # {metro} ({cont})')
    for cell in sorted(by_metro[metro]):
      lines.append(f'    {cell!r}: {metro!r},')
  lines.append('}')
  return '\n'.join(lines) + '\n'


def table_span(text: str):
  start = text.index(BEGIN)
  # Walk back over the comment block immediately above the assignment.
  head = text.rfind('\n\n', 0, start)
  start = head + 2 if head != -1 else start
  close = text.index('\n}\n', start) + len('\n}\n')
  return start, close


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument('--write', action='store_true')
  args = ap.parse_args()
  cl = load_source()
  if cl.self_check():
    print('REFUSING: source of truth fails self_check()', file=sys.stderr)
    return 2
  want = render(cl)

  rc = 0
  for pattern in TARGETS:
    for path in sorted(glob.glob(pattern)):
      real = os.path.realpath(path)
      if EXCLUDE.search(real + '/'):
        continue
      text = open(real).read()
      if BEGIN not in text:
        continue
      before = ast.literal_eval(
          ast.parse(text[text.index(BEGIN) + len('_CELL_TO_METRO = '):
                         text.index('\n}\n', text.index(BEGIN)) + 2]).body[0].value)
      span = table_span(text)
      new_text = text[:span[0]] + want + text[span[1]:]
      after = ast.literal_eval(
          ast.parse(new_text[new_text.index(BEGIN) + len('_CELL_TO_METRO = '):
                             new_text.index('\n}\n', new_text.index(BEGIN)) + 2]
                    ).body[0].value)
      # ZERO REGRESSION GATE: every pre-existing row must survive with the SAME
      # metro. A widening that silently re-homes a cell is not a widening.
      changed = {c: (before[c], after.get(c)) for c in before if after.get(c) != before[c]}
      if changed:
        print(f'REFUSING {real}: widening would CHANGE existing rows: {changed}')
        rc = 1
        continue
      try:
        ast.parse(new_text)
      except SyntaxError as e:
        print(f'REFUSING {real}: result would not parse ({e})')
        rc = 1
        continue
      gained = sorted(set(after) - set(before))
      if not gained and new_text == text:
        print(f'in sync ({len(after)} cells)          {real}')
        continue
      print(f'{"widened" if args.write else "would widen"} '
            f'{len(before)} -> {len(after)} cells (+{len(gained)})  {real}')
      if args.write:
        tmp = real + '.tmp'
        with open(tmp, 'w') as f:
          f.write(new_text)
        os.replace(tmp, real)
  return rc


if __name__ == '__main__':
  sys.exit(main())
