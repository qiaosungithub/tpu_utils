"""Dump full work-unit status.message (traceback) for an XID."""
from absl import app
from google3.learning.deepmind.xmanager2.client import xmanager_api


def main(argv):
    xid = int(argv[1])
    c = xmanager_api.XManagerApi(xm_deployment_env='alphabet')
    exp = c.get_experiment(xid)
    print(f"=== XID {xid} : {exp.name} ===")
    for wu in exp.get_work_units():
        st = getattr(wu, 'status', None)
        msg = ''
        if st is not None and hasattr(st, 'message'):
            msg = st.message or ''
        print(f"\n--- WU {wu.id}  state={wu.status_name} ---")
        print(msg if msg else "(empty status.message)")


if __name__ == '__main__':
    app.run(main)
