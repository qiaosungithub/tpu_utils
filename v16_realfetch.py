"""Real-RPC proof for the sys.modules half-init fix (infra-v16).

Run: blaze run //experimental/users/qiaos/tpu_utils:v16_realfetch
"""

import sys

from absl import app

from google3.experimental.users.qiaos.tpu_utils import avail_provider as AP


def main(argv):
  del argv
  p = AP.AvailabilityProvider(group='9')
  a, _, pool = p.fetch()
  print(f'[REAL-1 healthy] cells={len(a)} '
        f'pools={ {k: int(v) for k, v in pool.items()} }')
  assert a, 'empty availability from a real RPC'

  # Poison the process the way the outage did: strip attributes off the live
  # stubby modules so the next lazy attribute lookup raises the exact
  # AttributeError shape. Generated protos are left alone.
  poisoned = 0
  for name in list(sys.modules):
    if 'stubby' not in name.lower() or '_pb2' in name.lower():
      continue
    mod = sys.modules[name]
    if mod is None:
      continue
    for attr in list(vars(mod)) if hasattr(mod, '__dict__') else []:
      if attr.startswith('__'):
        continue
      try:
        delattr(mod, attr)
        poisoned += 1
      except Exception:  # pylint: disable=broad-except
        pass
  print(f'[REAL-2] stripped {poisoned} attrs off live stubby modules')

  a2, _, _ = p.fetch()
  print(f'[REAL-2 recovered] cells={len(a2)}')
  assert a2, 'fetch did not recover after poisoning'
  print('[REAL] OK')


if __name__ == '__main__':
  app.run(main)
