from google3.learning.deepmind.xmanager2.contrib.xm_resources import xm_resources_lib
from absl import app

def main(argv):
    res = xm_resources_lib.list_resources()
    if not res: 
        print('No resources')
    else:
        for k, v in list(res.items())[:1]:
            print(f'tuple unpack len: {len(v)}')
            print(f'dir: {dir(v)}')
            print(f'type: {type(v)}')
            if len(v) >= 3:
                print(f'v[2]: {v[2]}')

if __name__ == '__main__':
    app.run(main)
