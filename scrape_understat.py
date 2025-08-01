# scrape_understat.py (v6 - Final Diagnostic Version)
#
# Final attempt using 'shotsData' as the target variable.
# Includes a DEBUG_MODE to save a page's HTML for manual inspection if it fails.
#
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import asyncio
import aiohttp
import re
import random

# --- CONFIGURATION ---
SEASONS = ['2023', '2022']
OUTPUT_FILE = 'understat_data.csv'
BASE_URL = 'https://understat.com/league/EPL'
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'}

# --- DEBUG MODE ---
# If the script fails again, change this to True and run it.
# It will save the HTML of a single player page to 'debug_page.html' for inspection.
DEBUG_MODE = False
DEBUG_PLAYER_ID = '227' # Son Heung-Min, a player with lots of data

def parse_json_from_string(script_text, variable_name):
    """Uses RegEx to safely find and parse JSON data from a script tag."""
    pattern = re.compile(rf"{variable_name}\s*=\s*JSON\.parse\('([^']*)'\)")
    match = pattern.search(script_text)
    if match:
        json_data = match.group(1).encode('utf-8').decode('unicode_escape')
        return json.loads(json_data)
    return None

async def get_player_data(session, player_id):
    """Fetches detailed data for a single player by looking for 'shotsData'."""
    url = f"https://understat.com/player/{player_id}"
    try:
        async with session.get(url, headers=HEADERS) as response:
            if response.status != 200: return None
            html_content = await response.text()
            soup = BeautifulSoup(html_content, 'html.parser')
            scripts = soup.find_all('script')
            for script in scripts:
                player_data = parse_json_from_string(script.text, 'shotsData')
                if player_data:
                    return player_data # Returns a list of SHOT dictionaries
    except Exception:
        return None
    return None

async def run_debug_mode():
    """Runs in debug mode to save a single page's HTML for inspection."""
    print(f"--- RUNNING IN DEBUG MODE FOR PLAYER {DEBUG_PLAYER_ID} ---")
    url = f"https://understat.com/player/{DEBUG_PLAYER_ID}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        with open("debug_page.html", "w", encoding="utf-8") as f:
            f.write(response.text)
        print("Successfully saved page content to 'debug_page.html'.")
        print("Please check this file for 'playersData', 'datesData', or 'shotsData' to find the correct variable.")
    else:
        print(f"Failed to fetch debug page. Status code: {response.status_code}")

async def main():
    """Main function to scrape and process data."""
    if DEBUG_MODE:
        await run_debug_mode()
        return

    print("Starting Understat data scrape (v6)...")
    # ... (the first part of the function is the same) ...
    player_list = {}
    for season in SEASONS:
        print(f"Fetching player list for {season} season...")
        response = requests.get(f"{BASE_URL}/{season}", headers=HEADERS)
        soup = BeautifulSoup(response.content, 'html.parser')
        scripts = soup.find_all('script')
        
        found_data_for_season = False
        for script in scripts:
            data = parse_json_from_string(script.text, 'playersData')
            if data:
                player_list[season] = {p['id']: p['player_name'] for p in data}
                found_data_for_season = True
                break
        if not found_data_for_season:
            print(f"Warning: Could not find player data for {season} season. It will be skipped.")
            
    if not player_list:
        print("No players found for any season. Exiting.")
        return

    async with aiohttp.ClientSession() as session:
        tasks = []
        for season, players in player_list.items():
            for player_id, player_name in players.items():
                tasks.append({'season': season, 'player_name': player_name, 'player_id': player_id})

        print(f"Beginning fetch for {len(tasks)} total player histories...")
        
        all_results = []
        for i, task_info in enumerate(tasks):
            shot_data = await get_player_data(session, task_info['player_id'])
            if shot_data:
                all_results.append({'shots': shot_data, 'task_info': task_info})
            if (i + 1) % 50 == 0:
                print(f"  ...fetched {i + 1} / {len(tasks)}")
            await asyncio.sleep(random.uniform(0.1, 0.25))

        print("Processing all fetched data...")
        all_players_df = pd.DataFrame()
        # Process shot-level data into game-level data
        for result in all_results:
            shots_df = pd.DataFrame(result['shots'])
            task_info = result['task_info']
            shots_df['xG'] = pd.to_numeric(shots_df['xG'])

            # --- KEY FIX: REMOVED THE LINE THAT CRASHED ---
            # shots_df['xA'] = pd.to_numeric(shots_df['xA']) # This line is now deleted.

            # Aggregate shots into games
            game_data = shots_df.groupby(['season', 'match_id', 'date', 'h_team', 'a_team']).agg(
                xG_understat=('xG', 'sum')
                # --- KEY FIX: REMOVED xA FROM THE AGGREGATION ---
                # xA_understat=('xA', 'sum')
            ).reset_index()
            
            game_data['player_name'] = task_info['player_name']
            all_players_df = pd.concat([all_players_df, game_data], ignore_index=True)

    if not all_players_df.empty:
        print(f"\nSuccessfully scraped data for {len(all_players_df['player_name'].unique())} unique players across valid seasons.")
        all_players_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Data saved to '{OUTPUT_FILE}'")
    else:
        print("No data was scraped successfully despite new measures. Understat may have stronger protections in place.")

if __name__ == '__main__':
    asyncio.run(main())