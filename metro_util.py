"""Cell -> metro resolution. A thin, fail-closed facade over ``cell_locality``.

THIS MODULE NO LONGER OWNS A TABLE. It used to carry a nine-cell override list
plus two GUESSES -- "a cell named ``yu<metro><suffix>`` is in ``<metro>``" and,
failing that, "the cell's own name is its metro". The second guess was measured
wrong for **26 of 57** cells (`mach_locality -k metro is` is ``cbf``, not
``is``), and it failed in the worst direction: silently, and toward a value that
looks like a real metro. Under ``--metro=tul`` the tul cells ``oe``/``nf``/
``nm``/``oi`` were each computed into their own private metro and dropped, so
the filter reported "no placeable cell in this metro" -- indistinguishable from
a capacity shortage.

The measured snapshot in ``cell_locality`` is now the single owner, and an
unknown cell resolves to ``cell_locality.UNKNOWN`` rather than to a guess. Both
routers depend on this leaf (the smart-cell default via ``avail_provider`` and
the ``--power`` router via ``preflight.router``), so they cannot disagree.

``metro_of`` keeps returning a plain ``str`` for compatibility with the two
callers that put it straight into a set comparison; the unknown case returns the
sentinel, which compares equal to nothing and is falsy. A caller that must not
proceed on an unknown cell should use ``metro_of_or_die``.
"""

from google3.experimental.users.qiaos.tpu_utils import cell_locality

UNKNOWN = cell_locality.UNKNOWN
UnknownCellError = cell_locality.UnknownCellError

# Kept as a name because avail_provider re-exports it, but it is now DERIVED
# from the measured snapshot rather than hand-maintained: every short-named
# cell whose metro is not its own name. Nothing needs to be added here by hand
# any more -- run `remeasure_cell_locality.py --write` instead.
METRO_OVERRIDES: dict[str, str] = {
    cell: row[0]
    for cell, row in cell_locality._MEASURED.items()  # pylint: disable=protected-access
    if not cell.endswith('-d') and cell != row[0]
}


def metro_of(cell: str):
  """Measured metro for a borg cell, or ``cell_locality.UNKNOWN``.

  NEVER GUESSES. An unmeasured cell yields the sentinel, which is falsy and
  compares equal to no metro, so a metro allow-list drops it -- the safe
  direction, and the caller can tell the two cases apart with ``is UNKNOWN``.
  """
  return cell_locality.metro_of(cell)


def metro_of_or_die(cell: str) -> str:
  """Measured metro, or raise ``UnknownCellError``. Use where a wrong answer costs."""
  return cell_locality.metro_of_or_die(cell)


def is_known(cell: str) -> bool:
  """True when the cell is in the measured snapshot."""
  return cell_locality.is_known(cell)
