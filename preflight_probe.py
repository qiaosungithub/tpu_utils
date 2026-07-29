"""Preflight probe: test what APIs are reachable from our workstation.

We test 4 candidates:
  A. `resource_service.get_resource_alloc()`      — auth + alloc metadata
  B. `resource_service.get_forecast_info(...)`    — per-cell free chips forecast
  C. `xm_resources_lib.list_resources()`          — richer, already used
  D. `BorgMaster.ProbeSliceAvailability`          — per-topology, per-cell
     (raw stubby to `master.<cell>.borg.google.com:9413`)

For each: print SUCCESS / FAILURE (and error). This tells us what building
blocks we can rely on for the real preflight() function.
"""

import sys
import traceback
from absl import app, flags

_ALLOC = flags.DEFINE_string(
    'alloc', 'group:deepmind-dynamic/vqfree-xm',
    'Resource alloc to probe.')
_CELL = flags.DEFINE_string(
    'cell', '', 'Optional single cell to probe with BorgMaster (else auto).')
_LOCUS = flags.DEFINE_string(
    'locus', 'locus:DEPLOYMENT_TYPE_GHOSTLITE_POD:4_4',
    'Locus string to probe (default v6e-16).')

def try_step(name, fn):
    print(f'\n===== [{name}] =====')
    try:
        result = fn()
        print(f'✅ SUCCESS')
        return result
    except Exception as e:
        print(f'❌ FAILED: {type(e).__name__}: {e}')
        print(traceback.format_exc(limit=3))
        return None

def probe_A_get_resource_alloc(alloc):
    from google3.learning.deepmind.xmanager2.client import resource_service
    details = resource_service.get_resource_alloc(alloc)
    print(f'  fields: {[f.name for f in details.DESCRIPTOR.fields]}')
    if hasattr(details, 'borg_cells'):
        print(f'  borg_cells: {list(details.borg_cells)[:10]}')
    # print short repr of top-level scalar fields
    for f in details.DESCRIPTOR.fields:
        val = getattr(details, f.name)
        if f.type not in (11, 14):  # not message, not enum
            print(f'    {f.name} = {val!r}')
    return details

def probe_B_get_forecast(alloc):
    from google3.learning.deepmind.xmanager2.client import resource_service
    forecast = resource_service.get_forecast_info(
        alloc,
        with_global_batch_accelerators_availability=True,
        with_full_availability=True)
    print(f'  forecast type: {type(forecast).__name__}')
    print(f'  top-level fields: {[f.name for f in forecast.DESCRIPTOR.fields]}')
    # `usage` and `forecast` are map<string, X>; inspect a sample cell
    for top_field in ['usage', 'forecast']:
        sub = getattr(forecast, top_field, None)
        if sub is None:
            continue
        print(f'  .{top_field} type: {type(sub).__name__}, len={len(sub)}')
        for cell_name in sorted(sub.keys())[:3]:
            cell_val = sub[cell_name]
            print(f'    [{cell_name}] type: {type(cell_val).__name__}')
            if hasattr(cell_val, 'DESCRIPTOR'):
                print(f'      fields: {[f.name for f in cell_val.DESCRIPTOR.fields][:20]}')
                # Try priorities
                if hasattr(cell_val, 'priorities'):
                    print(f'      priorities keys: {list(cell_val.priorities.keys())}')
                    for pk in list(cell_val.priorities.keys())[:1]:
                        pv = cell_val.priorities[pk]
                        print(f'        [{pk}] type: {type(pv).__name__}, fields: {[f.name for f in pv.DESCRIPTOR.fields if f.name.startswith("tpu_")][:15]}')
                        # print any topology-related field
                        for f in pv.DESCRIPTOR.fields:
                            if 'topo' in f.name.lower() or 'shape' in f.name.lower() or 'slice' in f.name.lower():
                                print(f'          topo-field {f.name}: {getattr(pv, f.name)!r}')
            break  # one sample cell is enough
    return forecast

def probe_C_list_resources(alloc):
    from google3.learning.deepmind.xmanager2.contrib.xm_resources import xm_resources_lib
    res = xm_resources_lib.list_resources(resource_allocs=[alloc])
    if not res or alloc not in res:
        print(f'  ⚠️  alloc not in result. Got: {list((res or {}).keys())[:5]}')
        return None
    capacities, forecasts = res[alloc]
    print(f'  cells in capacities: {sorted(capacities.keys())[:8]}')
    return res

def _find_v6e_cell(alloc):
    """Find a cell that has v6e for probing."""
    try:
        from google3.learning.deepmind.xmanager2.contrib.xm_resources import xm_resources_lib
        res = xm_resources_lib.list_resources(resource_allocs=[alloc])
        capacities, _ = res[alloc]
        for cell, cap in sorted(capacities.items()):
            for p_name in ['HighlyAvailable', 'NonProd', 'BestEffort']:
                p = cap.priorities.get(p_name)
                if p and getattr(p, 'tpu_ghostlite_pod', 0) > 0:
                    return cell
    except Exception:
        pass
    return None

def probe_D_goodput_cell_availability(alloc):
    """Call GoodputService.GetCellAvailability via blade:xborg-prod-routing-layer."""
    import datetime
    from google3.borg.xborg.frontend.goodput_optimizer.proto import (
        goodput_optimizer_service_pb2)
    from google3.net.rpc.python.contrib import rpc_factory_factory
    from google3.net.rpc2.contrib.smartservice.python import smartservice_util

    # Resolve alloc -> pool + allotment
    from google3.learning.deepmind.xmanager2.client import resource_service
    details = resource_service.get_resource_alloc(alloc)
    pool_name = details.resource_pool_name  # 'deepmind-dynamic-pool'
    allotment = details.xborg_allotment_name  # 'group:vqfree-xm'
    print(f'  pool={pool_name}, allotment={allotment}')

    stub = smartservice_util.new_stub(
        goodput_optimizer_service_pb2.GoodputService,
        smartservice_util.parse('blade:xborg-prod-routing-layer'),
        rpc_factory=rpc_factory_factory.new_factory(
            deadline=datetime.timedelta(seconds=30)))

    # Inspect request fields to know what to fill
    req = goodput_optimizer_service_pb2.GetCellAvailabilityRequest()
    print(f'  Request fields: {[f.name for f in req.DESCRIPTOR.fields]}')

    # Fill request: pool + allotment + platform=GHOSTLITE_POD (v6e)
    from google3.borg.common import scalar_resource_pb2
    req.resource_pool = pool_name
    req.allowed_allotments.append(allotment)
    req.platforms.append(scalar_resource_pb2.ScalarResource.Key.GHOSTLITE_POD)

    print(f'  Sending request:\n{req}')
    resp = stub.GetCellAvailability(req)
    print(f'  ✓ got response, type={type(resp).__name__}')
    print(f'  Response top-level fields: {[f.name for f in resp.DESCRIPTOR.fields]}')
    # Dump response briefly
    resp_str = str(resp)
    print(f'  Response length: {len(resp_str)} chars')
    print(f'  Response first 3000 chars:')
    print(resp_str[:3000])
    return resp

def main(argv):
    del argv
    alloc = _ALLOC.value
    print(f'Testing pre-check APIs against alloc={alloc}')

    a = try_step('A: get_resource_alloc (auth + metadata)',
                 lambda: probe_A_get_resource_alloc(alloc))
    b = try_step('B: get_forecast_info (per-cell forecast)',
                 lambda: probe_B_get_forecast(alloc))
    c = try_step('C: xm_resources_lib.list_resources',
                 lambda: probe_C_list_resources(alloc))
    d = try_step(f'D: GoodputService.GetCellAvailability (real RPC)',
                 lambda: probe_D_goodput_cell_availability(alloc))

    print('\n===== SUMMARY =====')
    print(f'  A auth+alloc:              {"OK" if a else "FAIL"}')
    print(f'  B forecast_info:           {"OK" if b else "FAIL"}')
    print(f'  C list_resources:          {"OK" if c else "FAIL"}')
    print(f'  D GetCellAvailability:     {"OK" if d else "FAIL"}')

if __name__ == '__main__':
    app.run(main)
