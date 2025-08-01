
import pandas as pd
import glob

# === Load your main FPL GW Data ===
# Adjust this to your actual path where the GW CSV files are stored
gw_files = glob.glob('/Users/justin/Documents/repos/fpl/gws/gw*.csv')

# Load all gameweek files into one DataFrame
df_list = [pd.read_csv(file) for file in gw_files]
df = pd.concat(df_list, ignore_index=True)

print(f"✅ Loaded {len(df)} rows from {len(gw_files)} gameweeks.")
# === Step 1.1: Load team ID → team name mapping ===
teams_df = pd.read_csv('/Users/justin/Documents/repos/fpl/teams.csv')  # ✅ Adjust path if needed
team_map = teams_df.set_index('id')['name'].to_dict()

# === Step 1.2: Map opponent_team ID to team name ===
df['opponent_name'] = df['opponent_team'].map(team_map)

# === Step 1.3: Assign fixture difficulty scores ===
fixture_difficulty = {
    'Manchester City': 5,
    'Arsenal': 4,
    'Liverpool': 4,
    'Manchester United': 4,
    'Chelsea': 3,
    'Tottenham': 3,
    'Newcastle': 3,
    'Brighton': 3,
    'Aston Villa': 3,
    'Brentford': 2,
    'Crystal Palace': 2,
    'West Ham': 2,
    'Fulham': 2,
    'Wolves': 2,
    'Everton': 2,
    'Nottingham Forest': 1,
    'Sheffield United': 1,
    'Luton': 1,
    'Burnley': 1,
    'Bournemouth': 1
}
name_aliases = {
    'Man City': 'Manchester City',
    'Man Utd': 'Manchester United',
    'Spurs': 'Tottenham',
    'Wolves': 'Wolverhampton',
    'Nott\'m Forest': 'Nottingham Forest',
    'Sheffield Utd': 'Sheffield United'
    # Add more if needed
}
df['opponent_name'] = df['opponent_name'].replace(name_aliases)

df['fixture_difficulty'] = df['opponent_name'].map(fixture_difficulty)

# === [Optional] Preview results ===
print(df[['name', 'team', 'opponent_name', 'fixture_difficulty']].head())

# Save the combined gameweek data to disk
df.to_csv("combined_gameweeks.csv", index=False)
print("✅ Saved as combined_gameweeks.csv")