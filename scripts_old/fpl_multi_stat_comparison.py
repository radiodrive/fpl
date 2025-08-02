import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Config ===
csv_path = 'fpl_players.csv'  # Replace with your file path
top_n = 20  # Number of top players to show
stats_to_compare = ['Form', 'Total Points', 'ICT Index']  # Stats you want to compare

# === Load and Clean Data ===
df = pd.read_csv(csv_path)

# Ensure selected stats are numeric
for stat in stats_to_compare:
    df[stat] = pd.to_numeric(df[stat], errors='coerce')

# Drop rows with missing values for any selected stat
df = df.dropna(subset=stats_to_compare)

# Get top players by first stat (e.g., Form)
top_players = df.sort_values(stats_to_compare[0], ascending=False).head(top_n)

# === Prepare Data for Plotting ===
melted = top_players[['Name'] + stats_to_compare].melt(
    id_vars='Name', var_name='Stat', value_name='Value'
)

# === Plot ===
plt.figure(figsize=(12, 6))
sns.barplot(
    x='Value',
    y='Name',
    hue='Stat',
    data=melted,
    palette='Set2'
)

plt.title(f'Top {top_n} Players by {", ".join(stats_to_compare)}')
plt.xlabel('Value')
plt.ylabel('Player')
plt.legend(title='Stat', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()