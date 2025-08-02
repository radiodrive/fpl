import pandas as pd

# Load your dataset
df = pd.read_csv('fpl_players.csv')

# Ensure necessary fields are numeric
df['Form'] = pd.to_numeric(df['Form'], errors='coerce')
df['Cost (M)'] = pd.to_numeric(df['Cost (M)'], errors='coerce')
df['Total Points'] = pd.to_numeric(df['Total Points'], errors='coerce')

# Drop missing
df = df.dropna(subset=['Form', 'Cost (M)', 'Total Points'])

# Compute value metric
df['Points per Million'] = df['Total Points'] / df['Cost (M)']

# Filter cheap and in-form players
filtered = df[(df['Cost (M)'] <= 7.0) & (df['Form'] >= 4.0)]

# Sort by best value
top_value_players = filtered.sort_values('Points per Million', ascending=False).head(20)

# Output
print("🎯 Underpriced In-Form Players:")
print(top_value_players[['Name', 'Position', 'Cost (M)', 'Form', 'Total Points', 'Points per Million']])