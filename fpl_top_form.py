import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Config ===
csv_path = 'fpl_players.csv'  # Replace with your file path
top_n = 20  # Number of top players to show by form

# === Load and Clean Data ===
df = pd.read_csv(csv_path)

# Ensure 'Form' column is numeric
df['Form'] = pd.to_numeric(df['Form'], errors='coerce')
df = df.dropna(subset=['Form'])

# Sort by highest form
top_form_players = df.sort_values('Form', ascending=False).head(top_n)

# === Plot ===
plt.figure(figsize=(10, 8))
sns.barplot(
    x='Form',
    y='Name',
    data=top_form_players,
    palette='coolwarm'
)
plt.title(f'Top {top_n} FPL Players by Form')
plt.xlabel('Form')
plt.ylabel('Player Name')
plt.xlim(0, df['Form'].max() + 1)
plt.tight_layout()
plt.show()