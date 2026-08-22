

load("//devtools/python/blaze:pytype.bzl", "pytype_binary", "pytype_strict_binary")

pytype_strict_binary(
    name = "quota_check",
    srcs = ["quota_check.py"],
    deps = [
        ":group_utils",
        "//learning/deepmind/xmanager2/client:resource_service",
        "//third_party/py/rich",
        "//third_party/py/absl:app",
    ],
)

pytype_strict_binary(
    name = "money_check",
    srcs = ["money_check.py"],
    deps = [
        ":group_utils",
        "//devtools/production/pyspanner:pyspanner",
        "//experimental/users/qiaos/tpu_utils/preflight:market",
        "//learning/agents/orcas/tools/gqm_tool:gqm_tool",
        "//learning/deepmind/xmanager2/client:resource_service",
        "//third_party/py/absl:app",
        "//third_party/py/rich",
    ],
)


pytype_strict_binary(
    name = "inspect_gqm",
    srcs = ["inspect_gqm.py"],
    deps = [
        "//learning/agents/orcas/tools/gqm_tool:gqm_tool",
        "//third_party/py/absl:app",
    ],
)

load("//devtools/python/blaze:pytype.bzl", "pytype_strict_binary", "pytype_strict_library", "pytype_strict_contrib_test")

pytype_strict_library(
    name = "group_utils",
    srcs = ["group_utils.py"],
    deps = [
        "//learning/deepmind/xmanager2/contrib/xm_resources:xm_resources_lib",
    ],
)

pytype_strict_binary(
    name = "infra_check",
    srcs = ["infra_check.py"],
    deps = [
        ":group_utils",
        "//learning/deepmind/xmanager2/client:xmanager_api",
        "//third_party/py/etils/epath",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags:flags",
        "//third_party/py/rich",
    ],
)

pytype_strict_binary(
    name = "test_xmanager_api",
    srcs = ["test_xmanager_api.py"],
    deps = [
        "//learning/deepmind/xmanager2/client:xmanager_api",
        "//third_party/py/absl:app",
    ],
)

pytype_strict_binary(
    name = "preflight_probe",
    srcs = ["preflight_probe.py"],
    deps = [
        "//borg/public:master_py_pb2",
        "//borg/common:scalar_resource_py_pb2",
        "//borg/xborg/frontend/goodput_optimizer/proto:goodput_optimizer_service_py_pb2",
        "//learning/deepmind/xmanager2/client:resource_service",
        "//learning/deepmind/xmanager2/contrib/xm_resources:xm_resources_lib",
        "//net/rpc/python/contrib:rpc_factory_factory",
        "//net/rpc2/contrib/smartservice/python:smartservice_util",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags:flags",
    ],
)
















pytype_strict_binary(
    name = "why_probe",
    srcs = ["why_probe.py"],
    deps = [
        "//learning/deepmind/xmanager2/client:xmanager_api",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags:flags",
    ],
)

pytype_binary(
    name = "deep_probe",
    srcs = ["deep_probe.py"],
    tags = ["ignore_pytype"],
    deps = [
        "//learning/deepmind/xmanager2/client:xmanager_api",
        "//net/proto2/python/public",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags:flags",
    ],
)

# `infra_check` is a pytype_strict_binary, which a test target cannot depend on,
# so the test compiles infra_check.py into itself rather than linking it. Same
# self-asserting style as preflight/*_test.py -- it sys.exit(1)s on failure, and
# must be a *_contrib_test or blaze silently never runs it.
pytype_strict_contrib_test(
    name = "infra_check_test",
    srcs = [
        "infra_check.py",
        "infra_check_test.py",
    ],
    main = "infra_check_test.py",
    deps = [
        ":group_utils",
        "//learning/deepmind/xmanager2/client:xmanager_api",
        "//third_party/py/etils/epath",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags:flags",
        "//third_party/py/rich",
    ],
)

# Placeable slices per cell, which is NOT the obtainable-chip count preflight
# prints: `yutulpz` showed 1616 free v7 chips while the production run was
# being descheduled there for want of a contiguous 2x4x4.
pytype_strict_binary(
    name = "slice_probe",
    srcs = ["slice_probe.py"],
    deps = [
        "//borg/common:scalar_resource_py_pb2",
        "//borg/xborg/frontend/goodput_optimizer/proto:goodput_optimizer_service_py_pb2",
        "//learning/deepmind/xmanager2/client:resource_service",
        "//net/rpc/python/contrib:rpc_factory_factory",
        "//net/rpc2/contrib/smartservice/python:smartservice_util",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags:flags",
    ],
)

# --- Local-queue router --------------------------------------------------
# Pure scheduling core: queue schema, placement, priority/fairness, topology
# lock, effective-price type selection. No I/O, no RPC -- unit-tested in full.
pytype_strict_library(
    name = "route_lib",
    srcs = ["route_lib.py"],
)

pytype_strict_contrib_test(
    name = "route_lib_test",
    srcs = ["route_lib_test.py"],
    deps = [
        ":route_lib",
    ],
)

# Live availability provider: wraps the SAME GetCellAvailability RPC as
# slice_probe (free chips decide, obtainable lies) and the money market cache,
# producing the router's (avail_by_cell, arch_price, arch_pool). The google3
# RPC/proto imports are lazy (inside fetch), so the pure helpers unit-test with
# fakes; the deps are still declared for the live path.
pytype_strict_library(
    name = "avail_provider",
    srcs = ["avail_provider.py"],
    deps = [
        ":route_lib",
        "//borg/common:scalar_resource_py_pb2",
        "//borg/xborg/frontend/goodput_optimizer/proto:goodput_optimizer_service_py_pb2",
        "//learning/deepmind/xmanager2/client:resource_service",
        "//net/rpc/python/contrib:rpc_factory_factory",
        "//net/rpc2/contrib/smartservice/python:smartservice_util",
    ],
)

pytype_strict_contrib_test(
    name = "avail_provider_test",
    srcs = ["avail_provider_test.py"],
    deps = [
        ":avail_provider",
        ":route_lib",
    ],
)

# Router tick: drains the local queue into the XM queue via `tpu queue`.
# Default is --dry_run; the submit/cancel side effects sit behind a Submitter
# seam so the whole tick unit-tests with a fake submitter (no shell, no RPC).
# Logic lives in a LIBRARY so both the route_check binary AND queue_cli can
# depend on it (a py-strict binary/test may not depend on another binary).
pytype_strict_library(
    name = "route_check_lib",
    srcs = ["route_check.py"],
    deps = [
        ":avail_provider",
        ":route_lib",
        "//learning/deepmind/xmanager2/client:xmanager_api",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags:flags",
    ],
)

pytype_strict_binary(
    name = "route_check",
    srcs = ["route_check.py"],
    deps = [":route_check_lib"],
)

pytype_strict_contrib_test(
    name = "route_check_test",
    srcs = ["route_check_test.py"],
    deps = [
        ":avail_provider",
        ":route_check_lib",
        ":route_lib",
    ],
)

# Local-queue CLI: `tpu enqueue` / `tpu queue-status` / `tpu dequeue`. The
# side-by-side smart-queue path; does NOT touch the existing one-shot `tpu
# queue`. Thin arg-marshalling over route_lib (schema) + route_check (queue
# persistence + a dry-run planning probe for the status view).
pytype_strict_library(
    name = "queue_cli_lib",
    srcs = ["queue_cli.py"],
    deps = [
        ":avail_provider",
        ":route_check_lib",
        ":route_lib",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags:flags",
    ],
)

pytype_strict_binary(
    name = "queue_cli",
    srcs = ["queue_cli.py"],
    deps = [":queue_cli_lib"],
)

pytype_strict_contrib_test(
    name = "queue_cli_test",
    srcs = ["queue_cli_test.py"],
    deps = [
        ":queue_cli_lib",
        ":route_check_lib",
        ":route_lib",
        "//third_party/py/absl/flags:flags",
    ],
)
