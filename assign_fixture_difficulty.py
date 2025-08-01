import pandas as pd

# === Load Data ===
players = pd.read_csv('fpl_players.csv')
fixtures = pd.read_csv('fixtures.csv')

# === Difficulty Map (adjust as needed) ===
difficulty = {
    'Manchester City': 4,
    'Arsenal': 4,
    'Liverpool': 5,
    'Manchester United': 3,
    'Chelsea': 4,
    'Tottenham': 3,
    'Newcastle': 3,
    'Brighton': 3,
    'Aston Villa': 3,
    'Brentford': 2,
    'Wolverhampton': 2,
    'West Ham': 2,
    'Crystal Palace': 2,
    'Fulham': 3,
    'Bournemouth': 2,
    'Everton': 2,
    'Nottingham Forest': 3,
    'Sheffield United': 1,
    'Sunderland': 1,
    'Burnley': 1,
    'Leeds': 1
}

# === Team Name Normalization ===
team_name_map = {
    'Man City': 'Manchester City',
    'Man Utd': 'Manchester United',
    'Spurs': 'Tottenham',
    'Wolves': 'Wolverhampton',
    'Nott\'m Forest': 'Nottingham Forest',
    'Sheffield Utd': 'Sheffield United'
    # Add more if needed
}

# === Extract upcoming fixtures (e.g., Gameweek 1) ===
gw1 = fixtures[fixtures['Round Number'] == fixtures['Round Number'].min()]

# Create Home and Away entries
home = gw1[['Home Team', 'Away Team']].copy()
home.columns = ['Team', 'Opponent']
home['Is_Home'] = True

away = gw1[['Away Team', 'Home Team']].copy()
away.columns = ['Team', 'Opponent']
away['Is_Home'] = False

matches = pd.concat([home, away])

# Normalize opponent team names for difficulty mapping
matches['Opponent'] = matches['Opponent'].replace(team_name_map)

# Map difficulty score
matches['Fixture Difficulty'] = matches['Opponent'].map(difficulty)

# === Merge with players ===
players_with_fixture = players.merge(matches, left_on='Team', right_on='Team', how='left')

# Drop the merge key
players_with_fixture.drop(columns=['Team'], inplace=True)

# Save to new CSV
players_with_fixture.to_csv('fpl_players_with_fixtures.csv', index=False)
print("✅ Saved enriched player data with fixture difficulty to 'fpl_players_with_fixtures.csv'")