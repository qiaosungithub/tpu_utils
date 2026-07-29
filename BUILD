

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

load("//devtools/python/blaze:pytype.bzl", "pytype_strict_binary", "pytype_strict_library")

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
