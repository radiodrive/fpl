import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Config ===
csv_path = 'fpl_players.csv'  # Replace with your CSV path
positions_to_plot = ['Midfielder', 'Forward', 'Defender', 'Goalkeeper']
top_n = 10
stats_to_compare = ['Form', 'Points Per Game', 'ICT Index']  # Customize here

# === Load and Clean Data ===
df = pd.read_csv(csv_path)

# Ensure stats are numeric
for stat in stats_to_compare:
    df[stat] = pd.to_numeric(df[stat], errors='coerce')

# Drop rows where Position or any stat is missing
df = df.dropna(subset=stats_to_compare + ['Position'])

# Filter out players with all selected stats equal to zero
df = df[(df[stats_to_compare].fillna(0) > 0).any(axis=1)]

# Add a CombinedScore for ranking (sum of selected stats)
df['CombinedScore'] = df[stats_to_compare].sum(axis=1)

# === Loop Through Positions and Plot ===
for position in positions_to_plot:
    position_df = df[df['Position'] == position]
    if position_df.empty:
        continue  # Skip if no players found for this position

    # Sort by CombinedScore (or by Form if you prefer)
    top_players = position_df.sort_values('CombinedScore', ascending=False).head(top_n)

    # Melt the data for seaborn barplot
    melted = top_players[['Name'] + stats_to_compare].melt(
        id_vars='Name', var_name='Stat', value_name='Value'
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(
        x='Value',
        y='Name',
        hue='Stat',
        data=melted,
        palette='Set2'
    )
    plt.title(f'Top {top_n} {position}s by ' + ', '.join(stats_to_compare))
    plt.xlabel('Value')
    plt.ylabel('Player')
    plt.legend(title='Stat', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()