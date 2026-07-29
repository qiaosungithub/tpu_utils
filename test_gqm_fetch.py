from absl import app
from google3.learning.agents.orcas.tools.gqm_tool import gqm_tool

def main(argv):
    del argv
    print("=== Testing get_resource_prices ===")
    try:
        prices = gqm_tool.get_resource_prices()
        print("Prices sample:")
        print(prices[:1500])
    except Exception as e:
        print(f"Error fetching prices: {e}")

    print("\n=== Testing get_bidding_power ===")
    try:
        bp = gqm_tool.get_bidding_power("viscam-interns")
        print(f"viscam-interns BP: {bp}")
        bp2 = gqm_tool.get_bidding_power("deepmind-dynamic")
        print(f"deepmind-dynamic BP: {bp2}")
    except Exception as e:
        print(f"Error fetching bidding power: {e}")

if __name__ == "__main__":
    app.run(main)

