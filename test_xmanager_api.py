from google3.learning.deepmind.xmanager2.client import xmanager_api
from absl import app

def main(argv):
    c = xmanager_api.XManagerApi()
    print("Fetching XID...")
    for xid in [274789818]:
        try:
            exp = c.get_experiment(xid)
            wus = list(exp.get_work_units())
            print(f"XID {xid} has {len(wus)} work units.")
            for wu in wus:
                print(f"WorkUnit ID: {wu.id}")
                print(f"Status Name: {getattr(wu, 'status_name', 'N/A')}")
                for attr in ['status_message', 'error_message', 'failure_reason']:
                    print(f"Attr {attr}: {getattr(wu, attr, 'N/A')}")
                if hasattr(wu, 'status'):
                    st = wu.status
                    print(f"Status State: {getattr(st, 'state', 'N/A')}")
                    print(f"Status Message: {getattr(st, 'message', 'N/A')}")
                    print(f"Status detailed_executable_statuses: {getattr(st, 'detailed_executable_statuses', 'N/A')}")
                    print(f"Status structured_message: {getattr(st, 'structured_message', 'N/A')}")
                print("All dir(wu):", [a for a in dir(wu) if not a.startswith('_')])
        except Exception as e:
            print(f"Error fetching {xid}: {e}")

if __name__ == '__main__':
    app.run(main)
