from google3.experimental.users.qiaos.tpu_utils import group_utils
from google3.experimental.users.qiaos.tpu_utils.preflight import market
from google3.learning.deepmind.xmanager2.client import resource_service
from google3.devtools.production.pyspanner import pyspanner
from rich.console import Console
from rich.table import Table
from rich import box
from absl import app
import os
import re
import ast

# Sentinel decoding (INT64_MAX = unobtainable, INT64_MIN = no bid, 0 = a real
# free-pool price) lives in preflight.market.decode_price, so this renderer and
# the router that reads market.json can never disagree about what a value means.

# ResourceType ids come from //depot/google3/third_party/py/xmanager/xm/resources.py.
# GHOSTFISHLITE (101) is v7 and gets its own row -- it used to be folded into
# the v6p entry, which silently merged two different generations' prices.
TARGET_CARDS = [
    ("TPU v4", "tpu_pufferfish", [34]),
    ("TPU v5e", "tpu_viperlite_pod", [62, 60]),
    ("TPU v5p", "tpu_viperfish", [59]),
    ("TPU v6e", "tpu_ghostlite_pod", [76, 63]),
    ("TPU v6p", "tpu_ghostfish", [92]),
    ("TPU v7", "tpu_ghostfishlite", [101]),
]

class _BalanceByMdbQuery(pyspanner.Query):
    """SELECT t.ResourcePool, t.MilliCreditsBalance

    FROM MdbCreditBalance t
    WHERE t.Mdb = @p0
    """


def fetch_balance(gqm_tool, mdb_name):
    """Returns the GQM credit balance for one MDB, or None if unavailable.

    Balance is the accumulated stock of credits an MDB can spend in BATCH
    auctions; bidding power is the per-hour income flow. A static-pool MDB
    has no row here, which is reported as None rather than 0 so the caller
    can tell "no market participation" from "broke".
    """
    if not gqm_tool:
        return None
    try:
        client = gqm_tool.get_gqm_client()
        rows = list(client.BindAndQuery(_BalanceByMdbQuery, [mdb_name]))
    except Exception:
        return None
    if not rows:
        return None
    total = sum(float(r[1]) / 1000.0 for r in rows)
    # A static-pool MDB can still have a zero-balance row; it does not bid, so
    # report it as "no market participation" rather than an empty wallet.
    if total == 0.0:
        return None
    return total


class _ResourcePricesQuery(pyspanner.Query):
    """SELECT t.ResourcePool, t.ResourceType, t.Cell, t.Priority, t.MilliCreditsPerUnitHour

    FROM ResourcePrices t
    WHERE t.ResourcePool LIKE '%dynamic-pool'
    """


class _LimitOrdersQuery(pyspanner.Query):
    """SELECT t.Mdb, t.ResourceType, t.Priority, t.MilliCreditsPerUnitHour, t.LimitOrderUser, t.ResourcePool

    FROM LimitOrders t
    WHERE t.XManagerExperimentId = 0 AND t.ScuId = 0
    """


def fetch_limit_orders(gqm_tool, pools_out=None):
    """MDB-level limit orders: {(mdb, resource_type_int, priority): (mCPH, user)}.

    A limit order is a price cap. When the market clears ABOVE it, the SCU is
    moved to BUCKET_ID_TRIGGERED_LIMIT_ORDER, which "bypasses the main
    scheduling process" -- it is pulled from the queue BEFORE any capacity
    check, so free floor and free physical chips do not help. A cap set by a
    teammate (or a teammate's cron) silently applies to everyone in the MDB,
    since resolution order is SCU > XID > MDB. Without this table there is no
    way to see that from the client side.

    Only MDB-scoped rows are read (XManagerExperimentId = ScuId = 0). The more
    specific per-XID and per-SCU rows cannot exist for a job that has not been
    submitted yet, which is the only moment a pre-submit check runs.

    ``pools_out`` is an optional dict that receives the ResourcePool of each
    key. The cap is scoped to a pool -- the same MDB can hold different caps in
    different pools -- and the router needs that to join against the alloc's
    own pool, but the existing rendering here does not, so it stays an opt-in
    side channel rather than a change to the return type.
    """
    orders = {}
    try:
        client = gqm_tool.get_gqm_client()
        for row in client.Query(_LimitOrdersQuery):
            mdb, r_type, priority, milli, user = row[0], row[1], row[2], row[3], row[4]
            key = (mdb, r_type, priority)
            orders[key] = (milli, user)
            if pools_out is not None:
                pools_out[key] = row[5]
    except Exception:
        pass
    return orders


def _resolve_cap(limit_orders, my_mdbs, type_ids, tier):
    """Returns (cap_credits, user) for this (card, tier), or (None, None).

    Shared by the limit-order column and the per-cell colouring so the two can
    never disagree about which cells are actually reachable.
    """
    for mdb in sorted(my_mdbs):
        for type_id in type_ids:
            found = limit_orders.get((mdb, type_id, tier))
            if found:
                return found[0] / 1000.0, found[1]
    return None, None


def _limit_order_cell(limit_orders, my_mdbs, type_ids, tier, price_min, price_max):
    """Render the limit-order column: the cap, who set it, and whether it bites.

    Comparing the cap against the observed price range is the whole point --
    a cap only matters when the market clears above it, and that is exactly
    the state that silently strands a job in TRIGGERED_LIMIT_ORDER.
    """
    cap, user = _resolve_cap(limit_orders, my_mdbs, type_ids, tier)
    if cap is None:
        return "[dim]none[/dim]"
    if price_min is not None and price_min > cap:
        # Every cell we can see is above the cap: nothing can clear.
        return f"[bold red]{cap:.2f} BLOCKS ALL[/bold red] [dim]({user})[/dim]"
    if price_max is not None and price_max > cap:
        return f"[yellow]{cap:.2f} blocks dear cells[/yellow] [dim]({user})[/dim]"
    return f"[green]{cap:.2f} ok[/green] [dim]({user})[/dim]"


def _sample_cells(valid, per_band=2, cap=None):
    """Renders a price-stratified cell sample: dearest, median, cheapest.

    Each cell is coloured individually against the limit order ``cap``: green
    means the cell's clearing price is at or below the cap, so a job there can
    actually be admitted; red means it clears above the cap and would be
    stranded in TRIGGERED_LIMIT_ORDER. With no cap in force every cell is
    reachable, so all of them render green.

    Showing the top-N most expensive cells (the previous behaviour) answers
    "how bad can it get", but the question that actually decides a submit is
    "where can I land". A pool like v6p prices 2 cells at ~25000, 17 at ~59
    and 12 at 0.00: a dearest-only sample rendered four five-figure entries
    and hid the twelve free cells completely.

    Bands are cut by position in the sorted list, not by price, so they stay
    meaningful for any distribution. When bands would overlap (few cells) the
    sample degrades to a plain cheapest-first list rather than repeating a
    cell under two labels.

    Returns a rich-markup string, one band per line.
    """
    if not valid:
        return "[dim]-[/dim]"
    asc = sorted(valid, key=lambda x: x[1])
    n = len(asc)

    def fmt(entries):
        # Band labels stay dim on purpose: the red/green here means
        # "reachable or not", and colouring the labels too would compete
        # with that one signal.
        out = []
        for c, p, _ in entries:
            usable = cap is None or p <= cap
            color = 'green' if usable else 'red'
            out.append(f'[{color}]{c}:{p:.2f}[/{color}]')
        return ', '.join(out)

    # Not enough cells to stratify without repeats: just list them cheapest-up.
    if n < 3 * per_band:
        return fmt(asc[:3 * per_band])

    cheap = asc[:per_band]
    dear = asc[-per_band:][::-1]
    mid_start = (n - per_band) // 2
    mid = asc[mid_start:mid_start + per_band]
    return (f"[dim]dear[/dim]  {fmt(dear)}\n"
            f"[dim]mid[/dim]   {fmt(mid)}\n"
            f"[dim]cheap[/dim] {fmt(cheap)}")


def _extract_my_pools(resources) -> set[str]:
    """Given resources = xm_resources_lib.list_resources() output, return the set
    of ResourcePool names we participate in (derived from get_resource_alloc).
    Sentinel: we always include deepmind-dynamic-pool because most of our
    allocs live under it.
    """
    pools = set()
    for alloc_name in (resources or {}).keys():
        try:
            d = resource_service.get_resource_alloc(alloc_name)
            if d.resource_pool_name:
                pools.add(d.resource_pool_name)
        except Exception:
            pass
    if not pools:
        pools.add('deepmind-dynamic-pool')
    return pools

def write_to_cache(filename, content):
    cache_dir = os.path.expanduser("~/.tpu_quota_cache_dir")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def parse_bidding_power(raw_str):
    """Parses raw bidding power string into human-friendly format."""
    if not raw_str or not isinstance(raw_str, str):
        return "[dim]0.0 Credits/hr[/dim]"
    if "No bidding power information" in raw_str:
        # Kept deliberately short: the column is narrow and the long form
        # ('0.0 Credits/hr (Static Pool)') wrapped onto a second line for
        # every static-pool group, doubling the table height for no signal.
        return "[dim]0.0 (Static Pool)[/dim]"
    
    match = re.search(r"\{.*?\}", raw_str)
    if match:
        dict_str = match.group(0)
        try:
            val_dict = ast.literal_eval(dict_str)
            if isinstance(val_dict, dict) and val_dict:
                items = []
                for pool, val in val_dict.items():
                    color = "bold green" if float(val) > 0 else "yellow"
                    items.append(f"[{color}]{val:.1f} Credits/hr[/]")
                return ", ".join(items)
        except Exception:
            pass
    return f"[dim]{raw_str.strip()}[/dim]"

def fetch_prices_from_spanner(gqm_tool, my_pools=None):
    """Queries Spanner ResourcePrices table and aggregates BATCH + PROD prices.

    If my_pools is given, only entries in those pools are returned. Both BATCH
    and PROD prices are collected (both are broadcast by GQM cycle even though
    PROD admission is quota-driven; PROD price reflects the shadow cost).

    GQM encodes two distinct non-numeric states in this column, and they must
    not be collapsed into one another (or into a real 0.00 price):

    * ``INT64_MAX``  -- the resource is UNOBTAINABLE in that cell this cycle.
    * ``INT64_MIN``  -- no bid was recorded; GQM sanitizes it to 0.
    * ``0``          -- a genuine free-pool clearing price: supply met demand,
                        so the auction cleared at zero cost.

    That decoding now lives in ``preflight.market.decode_price``, which the
    router also uses, so the rendered table here and the machine-readable
    market.json can never disagree about what a sentinel means.

    Returns {(resource_type_int, priority_str): [(cell, price_or_None, pool)]}
    where ``price is None`` means UNOBTAINABLE specifically.
    """
    prices_by_key: dict[tuple, list] = {}
    try:
        spanner_client = gqm_tool.get_gqm_client()
        rows = list(spanner_client.Query(_ResourcePricesQuery))
        for row in rows:
            pool, r_type, cell, priority, milli_credits = row[0], row[1], row[2], row[3], row[4]
            if priority not in ('BATCH', 'PROD'):
                continue
            if my_pools is not None and pool not in my_pools:
                continue
            price = market.decode_price(milli_credits)
            key = (r_type, priority)
            prices_by_key.setdefault(key, []).append((cell, price, pool))
    except Exception as e:
        print(f"Warning: Failed to fetch prices from Spanner: {e}")
    return prices_by_key


def dump_market_json(spanner_prices, limit_orders, limit_order_pools,
                     my_pools, my_mdbs):
    """Also emit the full structured market data for the router to consume.

    The rendered money.txt keeps only four sample cells per card, which is
    enough to eyeball a price range but not to choose a cell -- and choosing a
    cheap cell is the zero-cost fix for a triggered limit order. Everything
    needed is already fetched and in memory at this point, so this costs one
    json.dump and no extra RPC: the daemon's round time is untouched.

    Failures here are logged, never raised. money.txt is the user-facing
    artifact of this binary and must still be written if the JSON dump breaks.
    """
    try:
        payload = market.build_payload(
            prices_by_key=spanner_prices,
            limit_orders=limit_orders,
            pools=my_pools,
            mdbs=my_mdbs,
            limit_order_pools=limit_order_pools,
        )
        market.write_snapshot(payload)
        n_cells = sum(len(v) for v in payload['prices'].values())
        return (f"market.json: {len(payload['prices'])} (pool,type,tier) keys, "
                f"{n_cells} cell prices, "
                f"{len(payload['limit_orders'])} limit orders")
    except (OSError, TypeError, ValueError) as e:
        return f"market.json NOT written ({type(e).__name__}: {e})"

def main(argv):
    del argv
    # height must be passed too: rich's Console.size only honours an explicit
    # width when height is ALSO set. Otherwise a dumb terminal (TERM=dumb, as
    # under a non-tty daemon or agent shell) short-circuits to a hard-coded
    # 80x25, silently squeezing the tables and re-introducing the wrapping
    # this rendering is trying to avoid.
    # 120 rather than 100: the stratified 'Sample cells' column needs room for
    # a band label plus two 'cell:price' pairs, and at 100 the price table was
    # silently clipped at the right border.
    console = Console(width=120, height=200, force_terminal=True,
                      color_system="standard")

    def render(printable):
        with console.capture() as cap:
            console.print(printable)
        return cap.get()

    out = render("[bold cyan]━━ GQM Money (Bidding Power) & Market Clearing Prices ━━[/bold cyan]\n")

    # Dynamically import gqm_tool
    gqm_tool = None
    try:
        import importlib
        gqm_module = importlib.import_module("google3.learning.agents.orcas.tools.gqm_tool.gqm_tool")
        gqm_tool = gqm_module
    except Exception:
        pass

    # 1. Ultra-clean MDB Groups Money Table
    resources, group_mapping = group_utils.get_group_mapping()

    money_table = Table(
        title="[bold magenta]MDB Groups Money (Bidding Power)[/bold magenta]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold bright_magenta",
        expand=False
    )
    money_table.add_column("Group", style="bold yellow", justify="center")
    money_table.add_column("PROD Usage (Chips)", justify="right")
    money_table.add_column("BATCH Usage (Chips)", justify="right")
    # Wide enough for the longest real value ('23575.0 Credits/hr') so the
    # numeric groups render on one line.
    money_table.add_column("Bidding Power (income/hr)", justify="left",
                           min_width=18, no_wrap=True)
    money_table.add_column("Balance (credits)", justify="right")

    for idx, alloc in group_mapping.items():
        mdb_name = alloc.split('/')[-1]
        bp_formatted = "[dim]0.0 Credits/hr[/dim]"
        if gqm_tool:
            try:
                fn_bp = getattr(gqm_tool, "get_bidding_power", None)
                if callable(fn_bp):
                    raw_bp = fn_bp(mdb_name)
                    bp_formatted = parse_bidding_power(raw_bp)
            except Exception as e:
                bp_formatted = f"[red]Err: {e}[/red]"

        # Calculate PROD and BATCH current usage
        prod_info = "0.0"
        batch_info = "0.0"
        try:
            prod_u = resource_service.get_resource_usage(alloc, ["HighlyAvailable"])
            batch_u = resource_service.get_resource_usage(alloc, ["NonProd"])
            # tpu_* fields in resource_model.proto are raw chip counts; there
            # is no milli-unit scaling to undo here.
            if prod_u:
                for f in prod_u.DESCRIPTOR.fields:
                    if f.name.startswith("tpu_") and getattr(prod_u, f.name):
                        prod_info = f"{float(getattr(prod_u, f.name)):,.0f}"
            if batch_u:
                for f in batch_u.DESCRIPTOR.fields:
                    if f.name.startswith("tpu_") and getattr(batch_u, f.name):
                        batch_info = f"{float(getattr(batch_u, f.name)):,.0f}"
        except Exception:
            pass

        balance = fetch_balance(gqm_tool, mdb_name)
        if balance is None:
            bal_str = "[dim]n/a (static pool)[/dim]"
        elif balance <= 0:
            bal_str = "[red]0[/red]"
        else:
            bal_str = f"[green]{balance:,.0f}[/green]"

        money_table.add_row(f"G{idx}", prod_info, batch_info, bp_formatted,
                            bal_str)

    out += render(money_table) + "\n"

    # 2. Market Prices Table — filtered to OUR pools + both BATCH & PROD.
    my_pools = _extract_my_pools(resources)
    spanner_prices = fetch_prices_from_spanner(gqm_tool, my_pools=my_pools) if gqm_tool else {}
    limit_order_pools: dict[tuple, str] = {}
    limit_orders = (fetch_limit_orders(gqm_tool, pools_out=limit_order_pools)
                    if gqm_tool else {})
    my_mdbs = {a.split('/')[-1] for a in group_mapping.values() if a}

    # Machine-readable twin of the table below, for `tpu route`. Written from
    # the data already in memory -- no extra RPC, no extra daemon latency.
    market_note = dump_market_json(spanner_prices, limit_orders,
                                   limit_order_pools, my_pools, my_mdbs)

    price_table = Table(
        title=f"[bold cyan]Clearing Prices in Your Pools ({', '.join(sorted(my_pools))})[/bold cyan]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold bright_cyan",
        expand=False
    )
    price_table.add_column("Card Type", style="bold yellow", justify="center")
    price_table.add_column("Tier", justify="center")
    # These two are capped so the sample column below can have its 40 columns
    # without pushing the table past the console width (rich clips the right
    # border rather than shrinking, so an over-wide table loses content).
    price_table.add_column("Price range (min–max, median)", justify="left",
                           max_width=24)
    price_table.add_column("Limit order", justify="left", max_width=22)
    # The stratified sample is pre-wrapped into three labelled lines, so it
    # needs room for the widest one; letting rich re-wrap it would interleave
    # the bands and destroy the alignment that makes it readable.
    price_table.add_column("Sample cells (dear/mid/cheap)", justify="left",
                           min_width=40, overflow='fold')

    for card_idx, (card_title, key_str, type_ids) in enumerate(TARGET_CARDS):
        # Rule off between card types: each card contributes a PROD and a
        # BATCH row, and with wrapped price/cell text the two cards' rows ran
        # together visually.
        if card_idx:
            price_table.add_section()
        # BOTH tiers are shown. PROD used to be hidden here on the theory that
        # "PROD admission is quota-based, not price-based" and its price was a
        # mere shadow cost. That theory is FALSE for dynamic (GQM) pools: the
        # market is partitioned by (resource_type, cell, *priority*), so PROD
        # has a real clearing price that really does gate admission. Hiding it
        # removed the one panel that could explain a PROD job stuck in the
        # NOT_SCHEDULED_TRIGGERED_LIMIT_ORDER bucket.
        for tier_label, tier_color in [('PROD', 'bright_red'), ('BATCH', 'bright_blue')]:
            entries = []
            for type_id in type_ids:
                entries.extend(spanner_prices.get((type_id, tier_label), []))
            # Keep only entries with a valid price (skip None sentinels).
            valid = [(cell, price, pool) for (cell, price, pool) in entries if price is not None]
            if not valid:
                # Distinguish "pool never offers this card" from "offered but
                # every cell is unobtainable right now" -- both used to render
                # as a bare N/A, which hid a real signal.
                if not entries:
                    note = "[dim]not offered in this pool[/dim]"
                else:
                    note = (f"[yellow]unobtainable in all "
                            f"{len(entries)} cells[/yellow]")
                lo_disp = _limit_order_cell(limit_orders, my_mdbs, type_ids,
                                            tier_label, None, None)
                price_table.add_row(f"[bold]{card_title}[/bold]", f"[{tier_color}]{tier_label}[/{tier_color}]", note, lo_disp, "[dim]-[/dim]")
                continue
            prices_only = sorted(p for (_, p, _) in valid)
            n = len(prices_only)
            mn = prices_only[0]
            mx = prices_only[-1]
            median = prices_only[n // 2]
            n_unobtainable = len(entries) - len(valid)
            unobt = (f" [dim](+{n_unobtainable} cells unobtainable)[/dim]"
                     if n_unobtainable else "")
            if mn == mx == 0.0:
                # Zero price means the auction cleared for free this cycle --
                # it does NOT imply the chips are actually schedulable, and
                # BATCH work stays preemptible either way.
                summary = f"[green]0.00 (free pool)[/green]{unobt}"
            elif mn == mx:
                summary = f"[bold green]{mn:.2f}[/bold green] Credits/hr{unobt}"
            else:
                summary = f"[bold]{mn:.2f}–{mx:.2f}[/bold] Credits/hr (median {median:.2f}, n={n}){unobt}"
            # Stratified sample: 2 dearest / 2 median / 2 cheapest, each cell
            # coloured green/red by whether it clears the limit order.
            cap, _ = _resolve_cap(limit_orders, my_mdbs, type_ids, tier_label)
            cell_disp = _sample_cells(valid, cap=cap)
            lo_disp = _limit_order_cell(limit_orders, my_mdbs, type_ids,
                                        tier_label, mn, mx)
            price_table.add_row(f"[bold]{card_title}[/bold]", f"[{tier_color}]{tier_label}[/{tier_color}]", summary, lo_disp, cell_disp)

    out += render(price_table)
    out += render(
        "[dim]Balance = accumulated credits (stock) -> buys above-floor DRF "
        "weight; Bidding Power = credits/hr (flow) -> buys your floor. BOTH "
        "tiers are price-gated on dynamic (GQM) pools: a PROD price above your "
        "limit order strands the job in TRIGGERED_LIMIT_ORDER before any "
        "capacity check.[/dim]")
    out += render(
        "[dim]'free pool' = the auction cleared at zero this cycle (supply met "
        "demand). It is a real price, not missing data -- but it does not "
        "guarantee free chips exist, and BATCH is always preemptible.[/dim]")
    out += render("[dim]Tip: Run 'tpu quota -l' to see full MDB allocation paths for each group.[/dim]")

    write_to_cache("money.txt", out)
    print("Successfully written money.txt cache (PROD+BATCH prices, limit orders).")
    print(market_note)

if __name__ == "__main__":
    app.run(main)
