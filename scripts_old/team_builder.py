import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus
import matplotlib.pyplot as plt

# === Load Data ===
df = pd.read_csv('fpl_players.csv')

# Required columns
df['Cost (M)'] = pd.to_numeric(df['Cost (M)'], errors='coerce')
df['Total Points'] = pd.to_numeric(df['Total Points'], errors='coerce')
df['Position'] = df['Position'].astype(str)

# Optional: Projected Points
if 'Projected Points' in df.columns:
    df['Projected Points'] = pd.to_numeric(df['Projected Points'], errors='coerce')
    use_proj = True
else:
    use_proj = False
    df['Projected Points'] = df['Total Points']

# Drop bad rows
df = df.dropna(subset=['Cost (M)', 'Total Points', 'Position', 'Projected Points'])

# Build optimization model
model = LpProblem("FPL_Optimizer", LpMaximize)
player_vars = {name: LpVariable(name, cat='Binary') for name in df['Name']}

# Maximize projected points
model += lpSum(df.loc[df['Name'] == name, 'Projected Points'].values[0] * var
               for name, var in player_vars.items()), "TotalProjectedPoints"

# Constraints
model += lpSum(df.loc[df['Name'] == name, 'Cost (M)'].values[0] * var
               for name, var in player_vars.items()) <= 100, "Budget"

model += lpSum(player_vars.values()) == 15, "SquadSize"

# Position rules
position_limits = {'Goalkeeper': 2, 'Defender': 5, 'Midfielder': 5, 'Forward': 3}
for pos, limit in position_limits.items():
    model += lpSum(player_vars[name] for name in df[df['Position'] == pos]['Name']) == limit, f"{pos}_count"

# Solve
model.solve()

# === Output Team ===
team_df = df[df['Name'].isin([name for name, var in player_vars.items() if var.value() == 1])]
team_df = team_df[['Name', 'Position', 'Cost (M)', 'Total Points', 'Projected Points']]
team_df = team_df.sort_values(by='Position')

print(f"\n✅ Solver Status: {LpStatus[model.status]}")
print("\n💡 Optimal Team:")
print(team_df.to_string(index=False))

# === Plot Team Breakdown ===
plt.figure(figsize=(10, 6))
for position in team_df['Position'].unique():
    subset = team_df[team_df['Position'] == position]
    plt.barh(subset['Name'], subset['Projected Points'], label=position)

plt.xlabel('Projected Points')
plt.title('Optimized FPL Team by Projected Points')
plt.legend(title='Position')
plt.tight_layout()
plt.show()