from google3.learning.deepmind.xmanager2.contrib.xm_resources import xm_resources_lib
from google3.learning.deepmind.xmanager2.client import resource_service
from absl import app

def run(argv):
    del argv
    print(dir(xm_resources_lib))

if __name__ == "__main__":
    app.run(run)
