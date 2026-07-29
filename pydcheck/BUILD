load("//devtools/python/blaze:pytype.bzl", "pytype_binary")

# Read-only forensic check: does //third_party/py/pydantic expose BaseModel?
pytype_binary(
    name = "with_v1",
    srcs = ["chk.py"],
    main = "chk.py",
    tags = ["ignore_pytype"],
    strict_deps = False,
    deps = ["//third_party/py/pydantic"],
)

pytype_binary(
    name = "with_v2",
    srcs = ["chk.py"],
    main = "chk.py",
    tags = ["ignore_pytype"],
    strict_deps = False,
    deps = ["//third_party/py/pydantic:pydantic_v2"],
)

# Full EqR-jax import-graph probe with pydantic_v2 substituted in.
# Mirrors the staged target's deps exactly, except pydantic -> pydantic_v2.
pytype_binary(
    name = "graph_v2",
    srcs = ["graph.py"],
    main = "graph.py",
    tags = ["ignore_pytype"],
    strict_deps = False,
    deps = [
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags:flags",
        "//third_party/py/chex",
        "//third_party/py/colorama",
        "//third_party/py/coolname",
        "//third_party/py/flax",
        "//third_party/py/huggingface_hub",
        "//third_party/py/hydra",
        "//third_party/py/jax",
        "//third_party/py/matplotlib",
        "//third_party/py/ml_collections",
        "//third_party/py/numpy",
        "//third_party/py/omegaconf",
        "//third_party/py/optax",
        "//third_party/py/orbax/checkpoint",
        "//third_party/py/pandas",
        "//third_party/py/plotly",
        "//third_party/py/psutil",
        "//third_party/py/pydantic:pydantic_v2",
        "//third_party/py/scamper:wandb_mock",
        "//third_party/py/scipy",
        "//third_party/py/sklearn",
        "//third_party/py/torch",
        "//third_party/py/tqdm",
        "//third_party/py/xmanager",
        "//third_party/py/yaml",
    ],
)
