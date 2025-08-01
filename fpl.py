import requests
import pandas as pd

# URLs
url = "https://fantasy.premierleague.com/api/bootstrap-static/"

# Get JSON data
response = requests.get(url)
response.raise_for_status()
data = response.json()

# Create team and position lookup tables
teams = {team['id']: team['name'] for team in data['teams']}
positions = {ptype['id']: ptype['singular_name'] for ptype in data['element_types']}

# Extract player data
players = []
for player in data['elements']:
    player_info = {
        "Name": f"{player['first_name']} {player['second_name']}",
        "Team": teams[player['team']],
        "Position": positions[player['element_type']],
        "Cost (M)": player['now_cost'] / 10,
        "Total Points": player['total_points'],
        "Goals Scored": player['goals_scored'],
        "Assists": player['assists'],
        "Minutes": player['minutes'],
        "Selected By (%)": float(player['selected_by_percent']),
        "Form": float(player['form']),
        "Points Per Game": float(player['points_per_game']),
        "ICT Index": float(player['ict_index']),
        "Influence": float(player['influence']),
        "Creativity": float(player['creativity']),
        "Threat": float(player['threat'])
    }
    players.append(player_info)

# Convert to DataFrame
df = pd.DataFrame(players)

# Save to CSV
df.to_csv("fpl_players.csv", index=False)
print("Exported to fpl_players.csv ✅")