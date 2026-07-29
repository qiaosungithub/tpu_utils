load("//devtools/python/blaze:pytype.bzl", "pytype_strict_binary", "pytype_strict_library", "pytype_strict_contrib_test")

package(default_visibility = ["//experimental/users/qiaos:__subpackages__"])

pytype_strict_library(
    name = "topology",
    srcs = ["topology.py"],
)

pytype_strict_library(
    name = "capacity",
    srcs = ["capacity.py"],
    deps = [
        "//borg/common:scalar_resource_py_pb2",
        "//borg/xborg/frontend/goodput_optimizer/proto:goodput_optimizer_service_py_pb2",
        "//learning/deepmind/xmanager2/client:resource_service",
        "//net/rpc/python/contrib:rpc_factory_factory",
        "//net/rpc2/contrib/smartservice/python:smartservice_util",
    ],
)

pytype_strict_library(
    name = "market",
    srcs = ["market.py"],
    deps = [
        ":topology",
        "//borg/common:scalar_resource_py_pb2",
    ],
)

pytype_strict_library(
    name = "preflight",
    srcs = ["preflight.py"],
    deps = [
        ":capacity",
        ":topology",
    ],
)

pytype_strict_binary(
    name = "preflight_cli",
    srcs = ["preflight_cli.py"],
    deps = [
        ":preflight",
        "//experimental/users/qiaos/tpu_utils:group_utils",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags:flags",
    ],
)

pytype_strict_library(
    name = "router",
    srcs = ["router.py"],
    deps = [
        ":market",
        ":preflight",
        ":topology",
        "//experimental/users/qiaos/tpu_utils:group_utils",
    ],
)

pytype_strict_binary(
    name = "router_cli",
    srcs = ["router_cli.py"],
    deps = [
        ":market",
        ":preflight",
        ":router",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags:flags",
    ],
)

pytype_strict_contrib_test(
    name = "market_test",
    srcs = ["market_test.py"],
    deps = [":market"],
)

pytype_strict_contrib_test(
    name = "router_test",
    srcs = ["router_test.py"],
    deps = [
        ":capacity",
        ":preflight",
        ":router",
        "//experimental/users/qiaos/tpu_utils:group_utils",
    ],
)
