"""Dump the raw work-unit status message for an XID -- jobs.md's route #2."""
import sys
from absl import app
from google3.learning.deepmind.xmanager2.client import xmanager_api


def main(argv):
    xid = int(argv[1])
    c = xmanager_api.XManagerApi()
    exp = c.get_experiment(xid)
    print(f"XID {xid}: {exp.name}")
    wus = list(exp.get_work_units())
    print(f"work units: {len(wus)}")
    for wu in wus:
        print("=" * 70)
        print("id:", getattr(wu, "id", "?"), "state:", getattr(wu, "state", "?"))
        st = getattr(wu, "status", None)
        if st is not None:
            print("status.message:")
            print(getattr(st, "message", "") or "(empty)")
        for attr in ("status_message", "error_message", "failure_reason"):
            v = getattr(wu, attr, None)
            if v:
                print(f"{attr}: {v}")
        try:
            print("details:", wu.get_details())
        except Exception as e:
            print("get_details failed:", e)


if __name__ == "__main__":
    app.run(main)
