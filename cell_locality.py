"""Cell -> (metro, continent, campus), MEASURED. The single source of truth.

WHY THIS FILE EXISTS. "Which metro is this cell in" and "which bucket should it
write to" were maintained in 70 independent hand-written tables across nine
checkouts, and they disagreed. Two jobs died of the disagreement: one landed in
a cell absent from the training binary's table and crash-looped at startup, and
one landed in a cell absent from the launcher's bucket table, silently inherited
a checkpoint prefix on ANOTHER CONTINENT, dropped below the utilization
threshold, and was deleted by the pruner. Both failures are a lookup that
answered when it should have refused.

THREE PROPERTIES, and every one of them is load-bearing:

  1. THE DATA IS MEASURED, NOT GUESSED. Every row below came out of
     `mach_locality -k {kind} {cell}` -- see `remeasure.py`, which regenerates this file and
     diffs it. A cell that the tool cannot resolve is ABSENT here rather than
     guessed from its name; the name-based heuristics this replaced were wrong
     for 26 of 57 cells (`is` is not metro "is", it is cbf).
  2. AN UNKNOWN CELL IS `UNKNOWN`, A SENTINEL THAT IS NOT A CELL, NOT A METRO,
     AND NOT AN EMPTY STRING. `''`, `None` and "the cell's own name" all read as
     data downstream; `UNKNOWN` cannot be mistaken for an answer, and the
     `*_or_die` helpers raise on it.
  3. METRO AND BUCKET ARE SEPARATE QUESTIONS. A training binary pins by metro; a
     launcher needs a bucket. They are resolved by different functions over the
     same measured base, so neither can silently substitute for the other.

A SNAPSHOT WITHOUT A REPRODUCTION METHOD IS THE NEXT STALE DEFAULT, so the
generator ships beside the data: `python3 remeasure.py --diff` reports drift and
`--write` regenerates. Cells are turned up and retired; re-run it rather than
editing rows by hand.

Dependency-free by design (stdlib only, no google3, no I/O at import) so every
layer -- launcher, router, training binary, dataloader -- can import the same
module and cannot disagree.

MEASURED 2026-08-28T17:37:46Z over 254 candidate cells: 254 resolved, 0 unresolved.
"""

from __future__ import annotations

# The sentinel. NOT '' and NOT None: an empty string compares equal to a missing
# environment variable and reads as "no constraint" in a filter, and None
# silently becomes the string 'None' in an f-string path. UNKNOWN is a distinct
# object whose repr says what it is, and `metro_of_or_die` raises on it.
class _Unknown:
  """Singleton meaning 'this was not measured', distinguishable from any value."""

  __slots__ = ()

  def __repr__(self) -> str:
    return 'UNKNOWN'

  def __str__(self) -> str:
    return 'UNKNOWN'

  def __bool__(self) -> bool:
    return False


UNKNOWN = _Unknown()

# cell -> (metro, continent, campus). Storage cells ('is-d') and compute cells
# ('is') are BOTH listed, measured separately -- do not assume one implies the
# other.
# Unresolved by the probe, therefore deliberately absent:
#   (none -- every probed cell resolved)
_MEASURED: dict[str, tuple[str, str, str]] = {
    'lcbomp':      ('bom', 'ap', 'muk'),
    'ly':          ('bom', 'ap', 'kwa'),
    'wu':          ('icn', 'ap', 'gmh'),
    'yukulwh':     ('kul', 'ap', 'ebp'),
    'rx':          ('nrt', 'ap', 'inz'),
    'sd':          ('sin', 'ap', 'lyw'),
    'se':          ('sin', 'ap', 'lyw'),
    'sf':          ('sin', 'ap', 'wen'),
    'sg':          ('sin', 'ap', 'lyw'),
    'sh':          ('sin', 'ap', 'lyw'),
    'si':          ('sin', 'ap', 'wen'),
    'si-d':        ('sin', 'ap', 'wen'),
    'sj':          ('sin', 'ap', 'lyw'),
    'sk':          ('sin', 'ap', 'lyw'),
    'sl':          ('sin', 'ap', 'wen'),
    'sm':          ('sin', 'ap', 'wen'),
    'sm-d':        ('sin', 'ap', 'wen'),
    'sn':          ('sin', 'ap', 'lyw'),
    'so':          ('sin', 'ap', 'lyw'),
    'lcsydv':      ('syd', 'ap', 'erk'),
    'ta':          ('tpe', 'ap', 'chg'),
    'tb':          ('tpe', 'ap', 'chg'),
    'tc':          ('tpe', 'ap', 'chg'),
    'td':          ('tpe', 'ap', 'chg'),
    'tg':          ('tpe', 'ap', 'chg'),
    'th':          ('tpe', 'ap', 'chg'),
    'tl':          ('tpe', 'ap', 'chg'),
    'tm':          ('tpe', 'ap', 'chg'),
    'tp':          ('tpe', 'ap', 'chg'),
    'tp-d':        ('tpe', 'ap', 'chg'),
    'rc':          ('bll', 'eu', 'frd'),
    'rd':          ('bll', 'eu', 'frd'),
    'wb':          ('bru', 'eu', 'gbl'),
    'wd':          ('bru', 'eu', 'gbl'),
    'we':          ('bru', 'eu', 'gbl'),
    'wf':          ('bru', 'eu', 'gbl'),
    'wg':          ('bru', 'eu', 'gbl'),
    'wh':          ('bru', 'eu', 'gbl'),
    'wi':          ('bru', 'eu', 'gbl'),
    'wq':          ('bru', 'eu', 'gbl'),
    'ra':          ('dhr', 'eu', 'agr'),
    'rb':          ('dhr', 'eu', 'agr'),
    'dg':          ('dub', 'eu', 'ppk'),
    'di':          ('dub', 'eu', 'ppk'),
    'dj':          ('dub', 'eu', 'ppk'),
    'lcfrai':      ('fra', 'eu', 'hnu'),
    'ea':          ('grq', 'eu', 'grq'),
    'eb':          ('grq', 'eu', 'grq'),
    'ec':          ('grq', 'eu', 'grq'),
    'ed':          ('grq', 'eu', 'grq'),
    'ef':          ('grq', 'eu', 'grq'),
    'ei':          ('grq', 'eu', 'grq'),
    'ej':          ('grq', 'eu', 'grq'),
    'ej-d':        ('grq', 'eu', 'grq'),
    'el':          ('grq', 'eu', 'grq'),
    'el-d':        ('grq', 'eu', 'grq'),
    'en':          ('grq', 'eu', 'grq'),
    'eq':          ('grq', 'eu', 'grq'),
    'lclhrb':      ('lhr', 'eu', 'mrd'),
    'sv':          ('lhr', 'eu', 'srl'),
    'yulhrp':      ('lhr', 'eu', 'wxt'),
    'yulhrp-d':    ('lhr', 'eu', 'wxt'),
    'yulhrs':      ('lhr', 'eu', 'wxt'),
    'la':          ('lpp', 'eu', 'lpp'),
    'la-d':        ('lpp', 'eu', 'lpp'),
    'lb':          ('lpp', 'eu', 'lpp'),
    'lb-d':        ('lpp', 'eu', 'lpp'),
    'le':          ('lpp', 'eu', 'lpp'),
    'lg':          ('lpp', 'eu', 'lpp'),
    'lh':          ('lpp', 'eu', 'lpp'),
    'li':          ('lpp', 'eu', 'lpp'),
    'li-d':        ('lpp', 'eu', 'lpp'),
    'lj':          ('lpp', 'eu', 'lpp'),
    'lk':          ('lpp', 'eu', 'lpp'),
    'lo':          ('lpp', 'eu', 'lpp'),
    'lq':          ('lpp', 'eu', 'lpp'),
    'lt':          ('lpp', 'eu', 'lpp'),
    'lu':          ('lpp', 'eu', 'lpp'),
    'lu-d':        ('lpp', 'eu', 'lpp'),
    'yulpptr':     ('lpp', 'eu', 'lpp'),
    'yuskedq':     ('ske', 'eu', 'vlb'),
    'yuskedq-d':   ('ske', 'eu', 'vlb'),
    'ym':          ('atl', 'na', 'idi'),
    'ym-d':        ('atl', 'na', 'idi'),
    'yo':          ('atl', 'na', 'idi'),
    'yo-d':        ('atl', 'na', 'idi'),
    'yq':          ('atl', 'na', 'idi'),
    'ys':          ('atl', 'na', 'idi'),
    'lcausi':      ('aus', 'na', 'pfg'),
    'lcausr':      ('aus', 'na', 'pfg'),
    'ib':          ('cbf', 'na', 'srp'),
    'if':          ('cbf', 'na', 'srp'),
    'ig':          ('cbf', 'na', 'srp'),
    'iq':          ('cbf', 'na', 'sln'),
    'is':          ('cbf', 'na', 'sln'),
    'is-d':        ('cbf', 'na', 'sln'),
    'it':          ('cbf', 'na', 'sln'),
    'ix':          ('cbf', 'na', 'sln'),
    'iy':          ('cbf', 'na', 'sln'),
    'iz':          ('cbf', 'na', 'sln'),
    'iz-d':        ('cbf', 'na', 'sln'),
    'jb':          ('cbf', 'na', 'sln'),
    'je':          ('cbf', 'na', 'sln'),
    'jg':          ('cbf', 'na', 'sln'),
    'ji':          ('cbf', 'na', 'sln'),
    'jj':          ('cbf', 'na', 'sln'),
    'jn':          ('cbf', 'na', 'sln'),
    'jo':          ('cbf', 'na', 'sln'),
    'jp':          ('cbf', 'na', 'sln'),
    'jq':          ('cbf', 'na', 'sln'),
    'jq-d':        ('cbf', 'na', 'sln'),
    'js':          ('cbf', 'na', 'sln'),
    'jt':          ('cbf', 'na', 'sln'),
    'jz':          ('cbf', 'na', 'sln'),
    'ny':          ('cbf', 'na', 'srp'),
    'nz':          ('cbf', 'na', 'srp'),
    'nz-d':        ('cbf', 'na', 'srp'),
    'yucbfaa':     ('cbf', 'na', 'srp'),
    'yucbfab':     ('cbf', 'na', 'srp'),
    'yucbfac':     ('cbf', 'na', 'uno'),
    'yucbfad':     ('cbf', 'na', 'uno'),
    'yucbfcd':     ('cbf', 'na', 'uno'),
    'yucbfiv':     ('cbf', 'na', 'sln'),
    'yucbflq':     ('cbf', 'na', 'sln'),
    'yucbfpv':     ('cbf', 'na', 'uno'),
    'yucbfrl':     ('cbf', 'na', 'sln'),
    'yucbfrl-d':   ('cbf', 'na', 'sln'),
    'yucbfsl':     ('cbf', 'na', 'sln'),
    'yucbfsr':     ('cbf', 'na', 'uno'),
    'yucbful':     ('cbf', 'na', 'sln'),
    'yucbfwv':     ('cbf', 'na', 'uno'),
    'ue':          ('chs', 'na', 'mnk'),
    'uj':          ('chs', 'na', 'mnk'),
    'ux':          ('chs', 'na', 'mnk'),
    'uy':          ('chs', 'na', 'mnk'),
    'vj':          ('chs', 'na', 'mnk'),
    'vk':          ('chs', 'na', 'mnk'),
    'vl':          ('chs', 'na', 'mnk'),
    'vz':          ('chs', 'na', 'mnk'),
    'yuchspe':     ('chs', 'na', 'mnk'),
    'yuchstz':     ('chs', 'na', 'sml'),
    'ma':          ('ckv', 'na', 'spc'),
    'mb':          ('ckv', 'na', 'spc'),
    'mb-d':        ('ckv', 'na', 'spc'),
    'md':          ('ckv', 'na', 'spc'),
    'me':          ('ckv', 'na', 'spc'),
    'me-d':        ('ckv', 'na', 'spc'),
    'mf':          ('ckv', 'na', 'spc'),
    'mg':          ('ckv', 'na', 'spc'),
    'mg-d':        ('ckv', 'na', 'spc'),
    'mh':          ('ckv', 'na', 'spc'),
    'mj':          ('ckv', 'na', 'spc'),
    'yuckvax':     ('ckv', 'na', 'spc'),
    'ga':          ('cmh', 'na', 'nby'),
    'gb':          ('cmh', 'na', 'nby'),
    'gh':          ('cmh', 'na', 'nby'),
    'gl':          ('cmh', 'na', 'nby'),
    'gm':          ('cmh', 'na', 'nby'),
    'go':          ('cmh', 'na', 'nby'),
    'go-d':        ('cmh', 'na', 'nby'),
    'rg':          ('cmh', 'na', 'nby'),
    'yucmhaa':     ('cmh', 'na', 'clb'),
    'yucmhab':     ('cmh', 'na', 'clb'),
    'yucmhcg':     ('cmh', 'na', 'clb'),
    'yucmhcg-d':   ('cmh', 'na', 'clb'),
    'yucmhfq':     ('cmh', 'na', 'lct'),
    'yucmhgs':     ('cmh', 'na', 'clb'),
    'yucmhnb':     ('cmh', 'na', 'nby'),
    'yucmhps':     ('cmh', 'na', 'nby'),
    'yucmhqa':     ('cmh', 'na', 'lct'),
    'yucmhsu':     ('cmh', 'na', 'nby'),
    'yucmhty':     ('cmh', 'na', 'clb'),
    'yucmhty-d':   ('cmh', 'na', 'clb'),
    'yucmhwf':     ('cmh', 'na', 'nby'),
    'rq':          ('dfw', 'na', 'mdn'),
    'rr':          ('dfw', 'na', 'mdn'),
    'rs':          ('dfw', 'na', 'mdn'),
    'rs-d':        ('dfw', 'na', 'mdn'),
    'rt':          ('dfw', 'na', 'mdn'),
    'rw':          ('dfw', 'na', 'mdn'),
    'rw-d':        ('dfw', 'na', 'mdn'),
    'yudfwra':     ('dfw', 'na', 'red'),
    'pw':          ('dls', 'na', 'dls'),
    'px':          ('dls', 'na', 'dls'),
    'py':          ('dls', 'na', 'dls'),
    'pz':          ('dls', 'na', 'dls'),
    'ts':          ('dls', 'na', 'gor'),
    'tt':          ('dls', 'na', 'gor'),
    'yufwahd':     ('fwa', 'na', 'adc'),
    'yufwakf':     ('fwa', 'na', 'adc'),
    'bh':          ('iad', 'na', 'ldn'),
    'bi':          ('iad', 'na', 'sbp'),
    'bk':          ('iad', 'na', 'sbp'),
    'pd':          ('iad', 'na', 'rlf'),
    'wv':          ('iad', 'na', 'ara'),
    'ww':          ('iad', 'na', 'ara'),
    'yuiadrs':     ('iad', 'na', 'rlf'),
    'yuiadtq':     ('iad', 'na', 'rlf'),
    'dd':          ('las', 'na', 'hen'),
    'dl':          ('las', 'na', 'hen'),
    'dl-d':        ('las', 'na', 'hen'),
    'dy':          ('las', 'na', 'hen'),
    'dz':          ('las', 'na', 'hen'),
    'qc':          ('mrn', 'na', 'lnr'),
    'qn':          ('mrn', 'na', 'lnr'),
    'qo':          ('mrn', 'na', 'lnr'),
    'qo-d':        ('mrn', 'na', 'lnr'),
    'qr':          ('mrn', 'na', 'lnr'),
    'qr-d':        ('mrn', 'na', 'lnr'),
    'yumrnel':     ('mrn', 'na', 'lnr'),
    'yuphxej':     ('phx', 'na', 'msa'),
    'yuphxer':     ('phx', 'na', 'msa'),
    'yuphxrp':     ('phx', 'na', 'msa'),
    'yuphxrp-d':   ('phx', 'na', 'msa'),
    'ro':          ('rno', 'na', 'sty'),
    'yurnoaa':     ('rno', 'na', 'sty'),
    'yurnolb':     ('rno', 'na', 'sty'),
    'yurnoyc':     ('rno', 'na', 'sty'),
    'na':          ('tul', 'na', 'pry'),
    'nf':          ('tul', 'na', 'pry'),
    'nk':          ('tul', 'na', 'pry'),
    'nl':          ('tul', 'na', 'pry'),
    'nm':          ('tul', 'na', 'pry'),
    'nm-d':        ('tul', 'na', 'pry'),
    'nn':          ('tul', 'na', 'pry'),
    'oa':          ('tul', 'na', 'pry'),
    'od':          ('tul', 'na', 'pry'),
    'oe':          ('tul', 'na', 'pry'),
    'oe-d':        ('tul', 'na', 'pry'),
    'oi':          ('tul', 'na', 'pry'),
    'oi-d':        ('tul', 'na', 'pry'),
    'oj':          ('tul', 'na', 'pry'),
    'ok':          ('tul', 'na', 'pry'),
    'oq':          ('tul', 'na', 'pry'),
    'ot':          ('tul', 'na', 'pry'),
    'ow':          ('tul', 'na', 'pry'),
    'oz':          ('tul', 'na', 'pry'),
    'pa':          ('tul', 'na', 'pry'),
    'pb':          ('tul', 'na', 'pry'),
    'yutulis':     ('tul', 'na', 'pry'),
    'yutulpz':     ('tul', 'na', 'pry'),
    'yutulrf':     ('tul', 'na', 'pry'),
    'gc':          ('uos', 'na', 'wck'),
    'gd':          ('uos', 'na', 'wck'),
    'gd-d':        ('uos', 'na', 'wck'),
    'ge':          ('uos', 'na', 'wck'),
    'ge-d':        ('uos', 'na', 'wck'),
    'gg':          ('uos', 'na', 'wck'),
    'lcyulk':      ('yul', 'na', 'qbe'),
    'ce':          ('scl', 'sa', 'qca'),
    'cf':          ('scl', 'sa', 'qca'),
    'cg':          ('scl', 'sa', 'qca'),
    'cj':          ('scl', 'sa', 'qca'),
    'lcscld':      ('scl', 'sa', 'cno'),
}

MEASURED_AT = '2026-08-28T17:37:46Z'
MEASURE_COMMAND = 'mach_locality -k {kind} {cell}'


# --------------------------------------------------------------------------
# METRO / CONTINENT -- "where is this cell"
# --------------------------------------------------------------------------
# Every lookup below is TABLE-ONLY. There is deliberately no name-based
# fallback: the two heuristics this replaced ("yu<metro>..." and "the cell name
# IS the metro") were measured wrong for 26 of 57 cells, and both failed
# SILENTLY -- `metro_of('is')` returned 'is', which is not a metro, and a
# `--metro=cbf` filter then dropped the cbf cell `is` as out-of-metro.


def metro_of(cell):
  """Measured metro for `cell`, or UNKNOWN. Never guesses from the name."""
  row = _MEASURED.get(_norm(cell))
  return row[0] if row else UNKNOWN


def continent_of(cell):
  """Measured continent ('na' / 'eu' / 'ap' / 'sa' / ...), or UNKNOWN."""
  row = _MEASURED.get(_norm(cell))
  return row[1] if row else UNKNOWN


def campus_of(cell):
  """Measured campus, or UNKNOWN. Campuses inside one metro are neighbours."""
  row = _MEASURED.get(_norm(cell))
  return row[2] if row else UNKNOWN


def is_known(cell) -> bool:
  """True when `cell` was measured. The one predicate every guard should use."""
  return _norm(cell) in _MEASURED


def metro_of_or_die(cell, what: str = 'resolve the metro of'):
  """Measured metro, or raise. Use this on any path where a wrong answer costs.

  FAIL CLOSED. The alternative -- returning a plausible default -- is what
  killed a job: an unlisted cell inherited a checkpoint prefix on another
  continent, and nothing in the logs said so.
  """
  m = metro_of(cell)
  if m is UNKNOWN:
    raise UnknownCellError(
        f'cannot {what} {cell!r}: it is not in the measured cell snapshot '
        f'({len(_MEASURED)} cells, measured {MEASURED_AT}). Refusing to guess '
        f'-- a guessed metro is how a job ends up writing across a continent. '
        f'Fix: run `python3 remeasure.py --write` (the cell may be newly turned '
        f'up), or pass the destination explicitly.')
  return m


class UnknownCellError(ValueError):
  """Raised when a cell cannot be resolved and guessing is not acceptable.

  A distinct type so a caller can catch exactly this and degrade deliberately
  (a dashboard may want to print 'unknown'), while a launcher or a training
  binary lets it propagate.
  """


def same_metro(a, b) -> bool:
  """True only when BOTH cells are known AND share a metro.

  Unknown-vs-anything is False, never True: "I cannot tell" must not read as
  "they match". Callers wanting three-way logic use `is_known` first.
  """
  ma, mb = metro_of(a), metro_of(b)
  return ma is not UNKNOWN and mb is not UNKNOWN and ma == mb


def same_continent(a, b) -> bool:
  """True only when BOTH cells are known AND share a continent.

  WHY THIS IS A SEPARATE QUESTION FROM `same_metro`. The two boundaries carry
  very different penalties, so a caller that can only ask one of them asks the
  wrong one: crossing a CONTINENT to WRITE checkpoints is catastrophic (a
  measured ~94x penalty; the storage stalls the accelerator, the duty cycle
  falls under the pruning threshold, and the job is deleted mid-run), while
  crossing one to READ is merely slow (measured 2.5x: 6.0 GiB transatlantic in
  13.97 s). So a dataset read may legitimately cross a continent where a
  checkpoint write may not.
  """
  ca, cb = continent_of(a), continent_of(b)
  return ca is not UNKNOWN and cb is not UNKNOWN and ca == cb


# --------------------------------------------------------------------------
# STORAGE -- "which bucket should this cell write to"
# --------------------------------------------------------------------------
# A SEPARATE QUESTION, deliberately answered by different functions. The
# training binaries pin by metro; the launcher needs a bucket. Conflating them
# is how one table's answer got used for the other's question.
#
# THE ENTRY CRITERION IS GROUP QUOTA IN THE FLEX REGISTRY, NOT PROXIMITY.
# `fileutil quota <group> <cell>` reports a plausible 500.00G for an
# UNREGISTERED group -- that is the default bucket it falls through to, not a
# ceiling -- so it cannot be used to decide this. Verified instead with
#   flex.par list_ceiling -s colossus -g deepmind-resources-colossus -l <cell>
# whose "Number of registrations found" is 0 for an unregistered cell.
#
# Each row is a METRO, not a cell: every cell in a metro shares its storage
# (cross-cell same-metro reads are effectively free, even across campuses).
# That is not an assumption -- the launcher's previous 18-cell cell->bucket
# table was checked and IS exactly this function of the measured metro, which
# is why collapsing it here changes no existing answer.
_METRO_STORAGE_CELL: dict[str, str] = {
    'cbf': 'is-d',       # 67.3 PiB, sp50   flex: 1 registration
    'tul': 'oi-d',       # 31.4 PiB, sp50   (nm-d is the older, fuller cell)
    'lpp': 'li-d',       # 85.6 PiB, sp50
    'sin': 'si-d',       # 9.25 PiB, sp50
    'mrn': 'qo-d',       # 9.71 PiB, sp50
    'grq': 'el-d',       # 94.2 PiB, sp50
    'ckv': 'mb-d',       # 11.3 PiB, sp50
    'dfw': 'rs-d',       # 45.0 PiB, sp50
    'las': 'dl-d',       # 32.0 PiB, sp10
    'cmh': 'go-d',       # 24.6 PiB, sp50
}

# Metros with NO group registration: writes there land on the PERSONAL 500 GiB
# per-cell ceiling. Listed explicitly rather than omitted, so `storage_cell_of`
# can say WHICH kind of "no" it is -- "the metro has no team storage" is a
# different problem from "I have never heard of this cell", and they have
# different fixes.
#
# `ske` is the documented case: `flex.par list_ceiling ... -l yuskedq-d` returns
# "Number of registrations found: 0", while `fileutil quota` would have claimed
# 500.00G. Exhausting a personal ceiling poisons every write in the cell.
_PERSONAL_ONLY_METROS: dict[str, str] = {
    'ske': 'yuskedq-d',
    'phx': 'yuphxrp-d',  # registered, but 500 TiB / sp20 -- a medium circle
}


def storage_cell_of(cell, *, allow_personal: bool = True):
  """The CNS cell (`'is-d'`) co-located with `cell`, or UNKNOWN.

  UNKNOWN means one of two things, and the caller usually wants to distinguish
  them -- use `explain_storage` for a sentence naming which:
    * the cell is not in the measured snapshot at all, or
    * its metro has no storage cell registered for the group.
  """
  m = metro_of(cell)
  if m is UNKNOWN:
    return UNKNOWN
  if m in _METRO_STORAGE_CELL:
    return _METRO_STORAGE_CELL[m]
  if allow_personal and m in _PERSONAL_ONLY_METROS:
    return _PERSONAL_ONLY_METROS[m]
  return UNKNOWN


def storage_cell_of_or_die(cell, *, allow_personal: bool = True) -> str:
  """The co-located CNS cell, or raise with a sentence naming the actual gap.

  THIS IS THE FUNCTION THAT REPLACES A SILENT DEFAULT. The bucket a job writes
  its checkpoints to is chosen once, at launch, from the landing cell; getting
  it wrong is not a slowdown but a deletion, and it leaves no trace in the logs
  because the wrong path is a perfectly valid path.
  """
  sc = storage_cell_of(cell, allow_personal=allow_personal)
  if sc is UNKNOWN:
    raise UnknownCellError(explain_storage(cell, allow_personal=allow_personal))
  return sc


def explain_storage(cell, *, allow_personal: bool = True) -> str:
  """One sentence saying why `cell` has (or lacks) a co-located bucket."""
  m = metro_of(cell)
  if m is UNKNOWN:
    return (f'cell {cell!r} is not in the measured snapshot '
            f'({len(_MEASURED)} cells, measured {MEASURED_AT}), so no bucket '
            f'can be proven co-located with it. Refusing to fall back to a '
            f'default prefix: an out-of-metro checkpoint stream is what got '
            f'a job pruned mid-run. Fix: `python3 remeasure.py --write` if the '
            f'cell is newly turned up, then add its metro to '
            f'_METRO_STORAGE_CELL if the group has quota there; or pass an '
            f'explicit bucket.')
  if m in _METRO_STORAGE_CELL:
    return f'cell {cell!r} is in metro {m!r}, whose group storage cell is {_METRO_STORAGE_CELL[m]!r}.'
  if m in _PERSONAL_ONLY_METROS:
    where = _PERSONAL_ONLY_METROS[m]
    if allow_personal:
      return (f'cell {cell!r} is in metro {m!r}, which has NO group storage '
              f'registration; {where!r} is the personal 500 GiB ceiling only.')
    return (f'cell {cell!r} is in metro {m!r}, which has no group storage '
            f'registration and personal storage was not allowed.')
  return (f'cell {cell!r} is in metro {m!r} (continent {continent_of(cell)}), '
          f'which has NO storage cell registered here. A checkpoint written '
          f'from this cell would cross a metro boundary at best. Fix: register '
          f'the group in a {m!r} cell and add it to _METRO_STORAGE_CELL, or '
          f'pass an explicit bucket, or do not run here.')


def bucket_for(cell, suffix: str, *, allow_personal: bool = True) -> str:
  """A full `/cns/<cell>-d/<suffix>` root co-located with `cell`, or raise.

  `suffix` is the project's own path under the cell root ('home/qiaos/eqr_data'),
  which stays with the project: the shared layer owns WHICH CELL, never which
  directory. Leading and trailing slashes are tolerated.
  """
  sc = storage_cell_of_or_die(cell, allow_personal=allow_personal)
  return f'/cns/{sc}/{suffix.strip("/")}'


def cells_in_metro(metro: str) -> list[str]:
  """Every measured COMPUTE cell in `metro` (storage `-d` cells excluded).

  Used to answer "if I pin --metro=X, where can this job land", which is the
  question a metro filter is really asking.
  """
  m = (metro or '').strip().lower()
  return sorted(c for c, row in _MEASURED.items()
                if row[0] == m and not c.endswith('-d'))


def metros_with_storage() -> list[str]:
  """Metros where a checkpoint can be written to GROUP quota, sorted."""
  return sorted(_METRO_STORAGE_CELL)


def _norm(cell) -> str:
  """Lowercase, stripped. Non-strings normalise to '' (never a partial match)."""
  return cell.strip().lower() if isinstance(cell, str) else ''


def self_check() -> list[str]:
  """Internal consistency of this module. Returns a list of problems ([] = ok).

  Cheap enough to call at import in a test, and it is what the per-checkout
  copies compare themselves against.
  """
  problems = []
  for metro, sc in _METRO_STORAGE_CELL.items():
    if sc not in _MEASURED:
      problems.append(f'{metro}: storage cell {sc!r} is not in the measured snapshot')
    elif _MEASURED[sc][0] != metro:
      problems.append(
          f'{metro}: storage cell {sc!r} measures as metro {_MEASURED[sc][0]!r} '
          f'-- the row claims a cell that is somewhere else')
  for metro, sc in _PERSONAL_ONLY_METROS.items():
    if metro in _METRO_STORAGE_CELL:
      problems.append(f'{metro} is listed both as group storage and personal-only')
    if sc in _MEASURED and _MEASURED[sc][0] != metro:
      problems.append(f'{metro}: personal cell {sc!r} measures as {_MEASURED[sc][0]!r}')
  for cell, row in _MEASURED.items():
    if len(row) != 3 or not row[0] or not row[1]:
      problems.append(f'{cell}: malformed row {row!r}')
  return problems

