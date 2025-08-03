from playwright.sync_api import sync_playwright
import json
import pandas as pd

def capture_stats_from_network():
    results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # 👈 run non-headless for testing
        context = browser.new_context()
        page = context.new_page()

        def handle_response(response):
            if "getplayerstatistics" in response.url and response.status == 200:
                print(f"📥 Found JSON: {response.url}")
                json_data = response.json()
                for p in json_data["playerTableStats"]:
                    results.append({
                        "name": p["name"],
                        "team": p["teamName"],
                        "position": p["positionText"],
                        "goals": p["goal"],
                        "assists": p["assistTotal"],
                        "rating": p["rating"],
                        "apps": p["apps"],
                        "mins": p["minsPlayed"]
                    })

        page.on("response", handle_response)

        # 🔍 This loads the stats page where the JSON is requested internally
        page.goto("https://www.whoscored.com/Statistics")
        page.wait_for_timeout(10000)  # allow all scripts to load

        browser.close()

    return results


# Run it
data = capture_stats_from_network()
df = pd.DataFrame(data)
print(df.head())
df.to_csv("whoscored_players_page1.csv", index=False)