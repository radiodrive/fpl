import pandas as pd

# === CONFIG ===
fixtures_file = 'fixtures.csv'
players_file = 'fpl_players.csv'
output_file = 'future_gw1_players.csv'
round_number = 1

# === Load Data ===
fixtures = pd.read_csv(fixtures_file)
players = pd.read_csv(players_file)

# === Filter for GW1 Fixtures ===
gw1_fixtures = fixtures[fixtures['Round Number'] == round_number]

# === Map Team Aliases (adjust if needed) ===
alias_map = {
    'Spurs': 'Tottenham',
    'Man City': 'Manchester City',
    'Man Utd': 'Manchester United',
    'Wolves': 'Wolverhampton Wanderers',
    'Nott\'m Forest': 'Nottingham Forest'
}
players['Team'] = players['Team'].replace(alias_map)
gw1_fixtures['Home Team'] = gw1_fixtures['Home Team'].replace(alias_map)
gw1_fixtures['Away Team'] = gw1_fixtures['Away Team'].replace(alias_map)

# === Build Fixture Difficulty Map ===
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

# === Assign Fixtures to Players ===
fixture_rows = []
for _, player in players.iterrows():
    player_team = player['Team']
    for _, fixture in gw1_fixtures.iterrows():
        if player_team == fixture['Home Team']:
            opponent = fixture['Away Team']
            was_home = True
        elif player_team == fixture['Away Team']:
            opponent = fixture['Home Team']
            was_home = False
        else:
            continue

        fixture_rows.append({
            **player,
            'opponent_name': opponent,
            'was_home': was_home,
            'fixture_difficulty': fixture_difficulty.get(opponent, None)
        })
        break  # one fixture per player

# === Save Output ===
future_df = pd.DataFrame(fixture_rows)
future_df.to_csv(output_file, index=False)
print(f"✅ Created {output_file} with {len(future_df)} players.")