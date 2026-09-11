"""CLI for the local router.

  tpu route --power=v5p-32 [--tier=PROD] [--groups=1,3,5] [--top=3]
            [--explain] [--json]

The table names a CELL, not just a (group, tpu_type). GQM clears prices per
cell and the spread is several-fold, so which cell you land in decides both
what the run costs and -- when a limit order is in play -- whether it runs at
all. `--explain` lists the combinations that were excluded by a price cap,
with the cap, the market price, and who set the cap: "everything is blocked"
must read as a diagnosis, not as an empty result.
"""

import json
import sys

from absl import app
from absl import flags

from google3.experimental.users.qiaos.tpu_utils.preflight import market
from google3.experimental.users.qiaos.tpu_utils.preflight import preflight
from google3.experimental.users.qiaos.tpu_utils.preflight import router

_POWER   = flags.DEFINE_string('power', '', "e.g. 'v5p-32' or bare int '32'")
_TIER    = flags.DEFINE_string('tier', 'PROD', 'PROD | BATCH')
_GROUPS  = flags.DEFINE_string('groups', '', 'comma-separated group ids (default: all)')
_METROS  = flags.DEFINE_string(
    'metros', '',
    'comma-separated metro allow-list for data-locality (e.g. "cbf,tul"). '
    'When set, ONLY cells in those metros are recommended; a combo with no '
    'in-metro cell is dropped. Empty = any metro (roam the fleet).')
_TOP     = flags.DEFINE_integer('top', 3, 'return top-N recommendations')
_TOL     = flags.DEFINE_float('tolerance', 0.5, 'fractional slack around target')
_JSON    = flags.DEFINE_bool('json', False, 'emit JSON')
_VERBOSE = flags.DEFINE_bool('verbose', False, 'stream per-candidate probing')
_EXPLAIN = flags.DEFINE_bool(
    'explain', False,
    'also list combos excluded by a triggered limit order, with the cap, the '
    'market price, and who set the cap')

RED, YEL, GRN, DIM, BOLD, RESET = (
    '\033[31m', '\033[33m', '\033[32m', '\033[2m', '\033[1m', '\033[0m')


def _fmt_price(p):
  return '-' if p is None else f'{p:.2f}'


def _fmt_cost(c):
  if c is None:
    return '-'
  return f'{c:.0f}' if c >= 10 else f'{c:.2f}'


def _market_banner(snapshot: market.MarketSnapshot) -> list[str]:
  """One line about the market data: how old, or why we have none.

  A price-blind run must never be silent -- that is exactly how the router used
  to recommend a combination that a limit order had already blocked.
  """
  if not snapshot.available:
    return [f'{YEL}[market] {snapshot.warning or "no market data"} '
            f'Prices, costs and limit orders are NOT applied.{RESET}']
  age = snapshot.age_str()
  if snapshot.is_stale():
    return [f'{YEL}[market] data is {age} old (stale; the daemon refreshes '
            f'every ~60s). Prices may have moved.{RESET}']
  return [f'{DIM}[market] prices {age} old, {len(snapshot.limit_orders)} '
          f'limit order(s) known.{RESET}']


def _limit_order_json(lo):
  if lo is None:
    return None
  return {'cap': lo.cap, 'user': lo.user, 'tier': lo.tier,
          'resource_type': lo.resource_type, 'pool': lo.pool}


def _candidate_json(c, rank_index):
  """JSON for one candidate.

  tpu_wrapper.sh parses this. Existing keys keep their exact meaning; `cell`,
  `price`, `cost_per_hour`, `blocked` and `limit_order` are purely additive.
  """
  cap = c.verdict.capacity
  return {
      'rank': rank_index + 1,
      'group': c.group_id,
      'alloc': c.alloc,
      'tpu_type': c.tpu_type,
      'power_score': c.power_score,
      'status': c.verdict.status.value,
      'quota': cap.alloc_scoped_quota if cap else 0,
      'used': cap.alloc_scoped_used if cap else 0,
      # False when the floor_v2 read FAILED, so a consumer can tell "no floor"
      # from "could not look". `quota` stays an int either way; older parsers
      # (tpu_wrapper.sh) ignore this key.
      'quota_readable': (cap.quota_readable if cap else False),
      'reasons': list(c.verdict.reasons),
      # --- added by the GQM upgrade ---
      'cell': c.cell,
      'price': c.price,
      'cost_per_hour': c.cost_per_hour,
      'obtainable': c.obtainable,
      # The pool-wide price the cap is compared against; `price` above is the
      # recommended cell's own rate, used for cost only.
      'pool_price': c.pool_price,
      'blocked': c.blocked,
      'block_reason': c.block_reason,
      'limit_order': _limit_order_json(c.limit_order),
      'pool': cap.pool if cap else '',
  }


def _print_explain(blocked_candidates):
  """List combos a limit order removed from consideration.

  Dropping them silently is what produces the "no viable combos, and no idea
  why" state: the cap is invisible from every other client-side tool.

  The table shows the POOL-level clearing price, because that is the number the
  cap is compared against -- the V2 auction collapses all cells into one
  synthetic layer before evaluating limit orders. The cheapest per-cell price is
  shown alongside purely so the gap is visible; it is explicitly NOT an escape
  route, and the advice line says so.
  """
  print()
  if not blocked_candidates:
    print(f'{DIM}--explain: no combo was excluded by a limit order.{RESET}')
    return
  print(f'{BOLD}Excluded by a triggered limit order '
        f'({len(blocked_candidates)} combos){RESET}')
  print(f'{DIM}  A limit order is a price cap. When the market clears above '
        f'it the SCU is dropped before{RESET}')
  print(f'{DIM}  it is ever bucketized, so free quota and idle chips do not '
        f'help.{RESET}')
  print(f'{DIM}  The cap is compared against the POOL-WIDE price, not your '
        f'cell: the V2 auction merges{RESET}')
  print(f'{DIM}  every cell into one layer first. Moving cells will NOT '
        f'unblock these.{RESET}')
  print(f'{DIM}  Real fixes: a different card, a different tier, or raise/'
        f'remove the cap{RESET}')
  print(f'{DIM}  (/google/bin/releases/brain-quota/set_limit_order/'
        f'set_limit_order --price=...).{RESET}')
  print()
  header = (f"  {'group':<6} {'tpu_type':<11} {'cap':<8} {'pool price':<11} "
            f"{'cheapest cell':<20} {'set by':<16}")
  print(header)
  print('  ' + '-' * (len(header) - 2))
  for c in blocked_candidates:
    lo = c.limit_order
    cap_s = f'{lo.cap:.2f}' if lo else '-'
    user = (lo.user if lo else '') or '-'
    cheapest = min((o for o in c.offers if o.price is not None),
                   key=lambda o: o.price, default=None)
    cheap_s = (f'{cheapest.cell}@{cheapest.price:.2f}' if cheapest else '-')
    print(f"  g{c.group_id:<5} {c.tpu_type:<11} {RED}{cap_s:<8}{RESET} "
          f"{RED}{_fmt_price(c.pool_price):<11}{RESET} {cheap_s:<20} "
          f"{user:<16}")


def main(argv):
  del argv
  if not _POWER.value:
    print('Error: --power required', file=sys.stderr)
    return 2
  groups = None
  if _GROUPS.value:
    try:
      groups = [int(g) for g in _GROUPS.value.split(',') if g.strip()]
    except ValueError:
      print('Error: --groups must be comma-separated ints', file=sys.stderr)
      return 2

  metros = [m.strip() for m in _METROS.value.split(',') if m.strip()] or None

  progress = ((lambda s: print(f'  [router] {s}', file=sys.stderr))
              if _VERBOSE.value else None)

  # ALWAYS ask for the full ranking and slice locally. `--top` is a display
  # window, and the table has to be able to say "showing 3 of 47" -- a count it
  # cannot have if the truncation happened upstream. This costs nothing: every
  # candidate is probed and ranked regardless, and `route`'s `top_k` only
  # slices the finished list (see `router.route`'s last line). Asking for the
  # top 3 never made the router do less work, it only made it say less.
  try:
    ranked_all, snapshot = router.route_all(
        power=_POWER.value, tier=_TIER.value, groups=groups,
        tolerance=_TOL.value, progress_fn=progress, metros=metros)
  except Exception as e:  # pylint: disable=broad-except
    print(f'router error: {type(e).__name__}: {e}', file=sys.stderr)
    return 2

  runnable = [c for c in ranked_all if not c.blocked]
  blocked = [c for c in ranked_all if c.blocked]
  ranked = runnable[:_TOP.value]

  if _JSON.value:
    payload = [_candidate_json(c, i) for i, c in enumerate(ranked)]
    if _EXPLAIN.value:
      # Blocked combos are appended, still flagged, so a machine consumer can
      # tell "nothing is runnable" from "nothing exists".
      payload += [_candidate_json(c, len(ranked) + i)
                  for i, c in enumerate(blocked)]
    print(json.dumps(payload, indent=2))
    return 0 if ranked else 1

  for line in _market_banner(snapshot):
    print(line)

  if not ranked:
    if blocked:
      print(f'{RED}No runnable (group, tpu_type, cell) for '
            f'power={_POWER.value} @ {_TIER.value}: every surviving combo is '
            f'blocked by a limit order.{RESET}')
      if not _EXPLAIN.value:
        print('  Re-run with --explain to see the caps, prices, and who set '
              'them.')
      else:
        _print_explain(blocked)
    else:
      print(f'{RED}No viable (group, tpu_type) combos for '
            f'power={_POWER.value} @ {_TIER.value}.{RESET}')
      print('  Try relaxing --tolerance, or picking a different tier.')
    return 1

  # Human-readable ranking table.
  print(f"{BOLD}Router recommendations for power={_POWER.value} "
        f"@ {_TIER.value}{RESET}")
  print()
  header = (f"  {'rank':<5} {'group':<5} {'tpu_type':<11} {'cell':<9} "
            f"{'status':<8} {'quota':<8} {'headroom':<11} {'price':<8} "
            f"{'cost/hr':<9} {'reasons'}")
  print(header)
  print('  ' + '-' * (len(header) - 2))
  for i, c in enumerate(ranked):
    cap = c.verdict.capacity
    # A failed floor_v2 read used to print as a confident `0`, which reads as
    # "this alloc has no capacity" and has been acted on as such. `?` is the
    # honest rendering: the instrument did not answer.
    if cap is None:
      quota = '?'
    elif not cap.quota_readable:
      quota = '?'
    else:
      quota = str(cap.alloc_scoped_quota)
    # For BATCH the floor is never consulted at admission time, so showing
    # "quota headroom" there would be inviting the user to rank on noise.
    # Show obtainable chips in the chosen cell instead.
    if _TIER.value.upper() == 'BATCH':
      headroom = f'{c.obtainable}/{c.chips}={int(c.obtainable_ratio)}x obt'
    elif quota == '?':
      # Headroom is computed from the same unreadable number; printing
      # "0/32=0x" here would re-tell the same lie in a second column.
      headroom = f'?/{c.chips}' if c.chips else '?'
    else:
      headroom = (f'{c.remaining_quota}/{c.chips}={int(c.headroom_ratio)}x'
                  if c.chips else '?')
    color = GRN if c.verdict.status == preflight.Status.GREEN else YEL
    status_disp = f'{color}{c.verdict.status.value}{RESET}'
    reasons_str = ('; '.join(c.verdict.reasons)[:44]
                   if c.verdict.reasons else '-')
    print(f"  {i+1:<5} g{c.group_id:<4} {c.tpu_type:<11} {c.cell or '-':<9} "
          f"{status_disp:<17} {quota:<8} {headroom:<11} "
          f"{_fmt_price(c.price):<8} {_fmt_cost(c.cost_per_hour):<9} "
          f"{reasons_str}")

  # ★Say how much of the answer this is. The table is a ranked WINDOW (--top,
  # default 3), and a window that does not announce its own size reads as the
  # whole fleet: a line concluded "only one cell exists for v7-32" from three
  # rows, and planned a cross-metro checkpoint move on it. Printed whether or
  # not anything was truncated, so its absence never has to be interpreted.
  hidden = len(runnable) - len(ranked)
  scope = ('groups ' + ','.join(f'g{g}' for g in groups)
           if groups else 'all groups')
  if hidden > 0:
    print(f"{DIM}  showing {len(ranked)} of {len(runnable)} runnable "
          f"placements across {scope} ({hidden} hidden; --top=N for more)"
          f"{RESET}")
  else:
    print(f"{DIM}  showing all {len(runnable)} runnable placement(s) across "
          f"{scope}.{RESET}")

  if _TIER.value.upper() == 'BATCH':
    print(f"{DIM}  BATCH is ranked on obtainable chips, not quota: the BATCH "
          f"pass never reads your floor.{RESET}")
  if blocked and not _EXPLAIN.value:
    print(f"{YEL}  {len(blocked)} more combo(s) excluded by a limit order; "
          f"re-run with --explain.{RESET}")
  if _EXPLAIN.value:
    _print_explain(blocked)
  return 0


if __name__ == '__main__':
  app.run(main)
