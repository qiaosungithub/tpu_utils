"""Import the EqR-jax staged package's module graph with pydantic_v2 in place.

Read-only. Points sys.path at the already-built staged runfiles tree so the
application sources are the exact ones Borg ran, but resolves third_party deps
from THIS binary's runfiles (where pydantic == pydantic_v2).
"""
import importlib
import os
import sys

STAGED = ('/google/src/cloud/qiaos/EqR-jax/google3/blaze-bin/experimental/qiaos/'
          'eqr_jax_final_stages/eqr_run_260728_234402/main.runfiles/google3/'
          'experimental/qiaos/eqr_jax_final_stages/eqr_run_260728_234402')


def main():
  sys.path.insert(0, STAGED)
  # Replicate main.py::_add_bazel_imports_dirs(): the wandb_mock py_library
  # relies on imports=["wandb_mock"], which is not honoured here either.
  here = os.path.dirname(os.path.abspath(__file__))
  parts = here.split(os.sep)
  if 'google3' in parts:
    g3root = os.sep.join(parts[: len(parts) - parts[::-1].index('google3')])
    cand = os.path.join(g3root, 'third_party/py/scamper/wandb_mock')
    print('wandb_mock dir exists =', os.path.isdir(cand), cand, flush=True)
    if os.path.isdir(cand):
      sys.path.append(cand)
  import pydantic
  print('pydantic BaseModel =', hasattr(pydantic, 'BaseModel'), flush=True)
  for mod in ('dataset.common', 'models.losses', 'dataset.puzzle_dataset',
              'utils.logging_util', 'utils.ckpt_util', 'models.eqr',
              'models.trm', 'train'):
    try:
      importlib.import_module(mod)
      print(f'IMPORT OK    {mod}', flush=True)
    except Exception as e:  # pylint: disable=broad-except
      import traceback
      print(f'IMPORT FAIL  {mod}: {type(e).__name__}: {e}', flush=True)
      traceback.print_exc()
      break


if __name__ == '__main__':
  main()
