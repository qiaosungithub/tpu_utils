from absl import app
import importlib

def main(argv):
    del argv
    gqm_tool = importlib.import_module("google3.learning.agents.orcas.tools.gqm_tool.gqm_tool")

    print("=== Querying Spanner ResourcePrices for TPU Types ===")
    try:
        spanner_client = gqm_tool.get_gqm_client()
        query = """SELECT t.ResourcePool, t.ResourceType, t.Cell, t.Priority, t.MilliCreditsPerUnitHour
                   FROM ResourcePrices t
                   WHERE t.Priority = 'BATCH'"""
        rows = list(spanner_client.Query(query))
        
        target_types = {
            34: "TPU v4 (pufferfish)",
            59: "TPU v5p (viperfish)",
            63: "TPU v6e (ghostlite_pod)",
            76: "TPU v6e (robot/other)",
            92: "TPU v6p (ghostfish)",
            101: "TPU v6p (ghostfishlite)"
        }
        
        print(f"Total BATCH rows fetched: {len(rows)}")
        for row in rows:
            pool, r_type, cell, priority, milli = row[0], row[1], row[2], row[3], row[4]
            if r_type in target_types:
                print(f"[{target_types[r_type]}] Pool: {pool} | Cell: {cell} | MilliCredits: {milli}")
    except Exception as e:
        print(f"Error querying spanner: {e}")

if __name__ == "__main__":
    app.run(main)




