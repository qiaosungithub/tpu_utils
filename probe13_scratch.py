from absl import app
from google3.learning.deepmind.xmanager2.client import xmanager_api

SH = [284877582, 284761523, 284914837, 284989851, 284717807, 284718363]
SM = [284774627, 284785595, 284804525, 284904632]


def main(argv):
  del argv
  c = xmanager_api.XManagerApi()
  for tag, ids in (('SH', SH), ('SM', SM)):
    for x in ids:
      try:
        e = c.get_experiment(int(x))
        wus = list(e.get_work_units(populate_detailed_executable_status=True))
        if not wus:
          print(f'{tag} {x}: GONE (0 wu)')
          continue
        w = wus[0]
        print(f'{tag} {x}: pend={getattr(w,"is_pending",None)} run={getattr(w,"is_running",None)} '
              f'fail={getattr(w,"is_failed",None)} stop={getattr(w,"is_stopped",None)} '
              f'compl={getattr(w,"is_completed",None)}')
      except Exception as ex:  # pylint: disable=broad-except
        print(f'{tag} {x}: EXC {type(ex).__name__}')


if __name__ == '__main__':
  app.run(main)
