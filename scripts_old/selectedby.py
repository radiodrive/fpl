import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Config ===
csv_path = 'fpl_players.csv'  # Replace with your file
top_n = 40  # Show top N most-picked players

# === Load and Clean Data ===
df = pd.read_csv(csv_path)

# Ensure "Selected By (%)" is numeric
df['Selected By (%)'] = (
    df['Selected By (%)']
    .astype(str)
    .str.replace('%', '', regex=False)
    .pipe(pd.to_numeric, errors='coerce')
)

# Drop rows with missing % data
df = df.dropna(subset=['Selected By (%)'])

# === Sort by popularity and plot ===
top_players = df.sort_values('Selected By (%)', ascending=False).head(top_n)

plt.figure(figsize=(10, 8))
sns.barplot(
    x='Selected By (%)',
    y='Name',
    data=top_players,
    palette='mako'
)
plt.title(f'Top {top_n} Most Picked FPL Players')
plt.xlabel('% Selected By')
plt.ylabel('Player Name')
plt.xlim(0, df['Selected By (%)'].max() + 5)
plt.tight_layout()
plt.show()