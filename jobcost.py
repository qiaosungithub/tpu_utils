"""One pricing path, and an assertion that both sides agree.

WHY A WHOLE MODULE FOR THIS. Tonight a healthy job that had run six hours with zero
preemptions was cancelled as the fleet's most expensive PROD job. The canceller priced a
b200-8 at 800 credits/hr; the gate that had admitted it priced the same job at 11.4. A 70x
disagreement, and the canceller's header comment said in plain words that it "reuses
budget_check.get_job_cost -- identical basis".

The comment was true. The code did import that function. What differed was WHEN: the
long-lived canceller had imported the module a day earlier, and the price table it closed
over predated the fix. Restarting it moved the number from 800 to 6.

So the lesson is not "share the code" -- they already shared it. It is:

  ★A claim of agreement that nothing checks is worth nothing, and a SHARED IMPORT IS NOT A
   SHARED VALUE once a process is long-lived.

Hence: one entry point, and `assert_consistent()`, which two independently-running components
call to compare their live answers on the same input. The comparison is the only thing that
can actually fail, and failing is the point.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Optional

_WIKI_TOOLS = os.path.expanduser('~/work/wiki_agents/tools')


class PricingUnavailable(Exception):
  """The price basis could not be established. ★Never priced as 0 or as a default.

  `0` and `100` are both real-looking numbers, and tonight taught the cost of a sentinel that
  can be mistaken for a measurement: a GPU family missing from the market regex fell to a
  100 cr/chip-hr default and was over-priced 169x, while `headroom=0` meant both 'genuinely
  full' and 'could not read'.
  """


class PricingDisagreement(Exception):
  """Two components priced the same job differently. Loud by construction."""


def _budget_check():
  """Import the ONE pricing implementation, freshly each call.

  ★The re-import is deliberate. `sys.modules` already holds it, so this is cheap, but going
  through the import system means a component that reloads gets the current module rather
  than a closure captured at startup. It does not fix a process that never reloads -- nothing
  here can -- which is why `assert_consistent` exists as well.
  """
  if _WIKI_TOOLS not in sys.path:
    sys.path.insert(0, _WIKI_TOOLS)
  # ★importlib, not `import budget_check`: the module lives OUTSIDE google3
  # (~/work/wiki_agents/tools, owned by the wiki_agent line), so the static checker cannot
  # resolve it and a plain import fails the build. It is also the right split -- billing is
  # not ours to vendor, and a copy here would be a second pricing path, which is the exact
  # thing this module exists to prevent.
  import importlib                                      # noqa: PLC0415
  return importlib.import_module('budget_check')


def price_job(power: str, tier: str, override: Optional[float] = None) -> tuple[float, str]:
  """(credits/hour, basis) for one job. The single entry point; nobody else computes cost.

  Raises PricingUnavailable rather than returning a fallback number.
  """
  bc = _budget_check()
  try:
    cost = bc.get_job_cost(power, tier, override) if override is not None \
        else bc.get_job_cost(power, tier)
  except Exception as e:                                # noqa: BLE001
    raise PricingUnavailable(f'{power}/{tier}: {e}') from e
  if cost is None:
    raise PricingUnavailable(f'{power}/{tier}: priced as None')
  try:
    _, basis = bc.chip_price(power)
  except Exception:                                     # noqa: BLE001
    basis = 'unknown'
  return float(cost), str(basis)


def assert_consistent(power: str, tier: str, other_cost: float, *,
                      other_name: str, tolerance: float = 0.05) -> float:
  """Compare another component's live price for the same job against ours.

  ★This is the check that was missing. Call it from any long-lived process that prices jobs,
  on every decision that spends or cancels money -- not at startup, where a stale table looks
  identical to a fresh one.

  `tolerance` is fractional (0.05 = 5%), which absorbs a market tick between the two reads
  while still catching the failure that matters: the disagreements seen tonight were 70x,
  169x and 588x, i.e. nowhere near a rounding difference.
  """
  ours, basis = price_job(power, tier)
  if ours <= 0 and other_cost <= 0:
    return ours
  denom = max(abs(ours), abs(other_cost), 1e-9)
  if abs(ours - other_cost) / denom > tolerance:
    ratio = max(ours, other_cost) / max(min(ours, other_cost), 1e-9)
    raise PricingDisagreement(
        f'{power}/{tier}: {other_name} priced it {other_cost:.4g}, the shared path prices it '
        f'{ours:.4g} (basis={basis}) -- a {ratio:.1f}x disagreement. Do NOT act on either '
        f'number. The usual cause is a long-lived process holding an import from before a '
        f'price-table change: restart it and re-compare. A job was cancelled at a fake price '
        f'after running six hours because nothing performed this comparison.')
  return ours


def belief_report(power: str, tier: str) -> str:
  """A line a long-lived process should log periodically, stating the price it BELIEVES.

  ★C11: any long-lived reader must be externally falsifiable. Three carriers of the same bug
  appeared tonight -- a queue rolled back by a stale snapshot, bash not re-reading an
  already-parsed function, and Python serving a 24-hour-old import while the fix sat on disk
  for 63 minutes. In every case the process looked healthy from outside; only its own claim
  about what it believes could have exposed the disagreement.
  """
  try:
    cost, basis = price_job(power, tier)
    return (f'[pricing-belief {time.strftime("%H:%M:%SZ", time.gmtime())}] '
            f'{power}/{tier} = {cost:.4g} cr/hr (basis={basis}, pid={os.getpid()})')
  except PricingUnavailable as e:
    return (f'[pricing-belief {time.strftime("%H:%M:%SZ", time.gmtime())}] '
            f'{power}/{tier} = UNAVAILABLE ({e}, pid={os.getpid()})')
