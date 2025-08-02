import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# === Load historical training data ===
df = pd.read_csv("combined_gameweeks.csv")

# Fix inconsistent team names
alias_map = {
    'Spurs': 'Tottenham',
    'Man City': 'Manchester City',
    'Man Utd': 'Manchester United',
    'Wolves': 'Wolverhampton Wanderers',
    "Nott'm Forest": 'Nottingham Forest'
}
df['opponent_name'] = df['opponent_name'].replace(alias_map)

# Map fixture difficulty
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
    'Wolverhampton Wanderers': 2,
    'Everton': 2,
    'Nottingham Forest': 1,
    'Sheffield United': 1,
    'Luton': 1,
    'Burnley': 1,
    'Bournemouth': 1
}
df['fixture_difficulty'] = df['opponent_name'].map(fixture_difficulty)

# === Select features & label ===
features = ['was_home', 'minutes', 'ict_index', 'influence', 'creativity', 'threat', 'value', 'fixture_difficulty']
label = 'total_points'

df_model = df.dropna(subset=features + [label])
X = df_model[features]
y = df_model[label]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\n📉 MAE: {mae:.2f}")
print(f"📈 R² Score: {r2:.3f}")

# === Load future GW1 player data ===
future_players = pd.read_csv("gw1_players.csv")

# Replace team aliases
future_players['opponent_name'] = future_players['opponent_name'].replace(alias_map)
future_players['fixture_difficulty'] = future_players['opponent_name'].map(fixture_difficulty)

# Rename 'Cost (M)' to 'value' if needed
if 'value' not in future_players.columns and 'Cost (M)' in future_players.columns:
    future_players['value'] = future_players['Cost (M)']

# Handle missing columns safely
existing_features = [f for f in features if f in future_players.columns]
missing = list(set(features) - set(existing_features))
if missing:
    print(f"\n⚠️ Missing features in future_players: {missing}")
future_players = future_players.dropna(subset=existing_features)

# Predict
future_players['predicted_points'] = model.predict(future_players[existing_features])

# Show top 10 predictions
top_preds = future_players[['Name', 'Team', 'Position', 'predicted_points']].sort_values(
    by='predicted_points', ascending=False
).head(10)

print("\n🔮 Top 10 Predicted Scorers for GW1:")
print(top_preds.to_string(index=False))

# Optional: Plot feature importance
model.get_booster().feature_names = existing_features
model.get_booster().feature_names = [str(f) for f in existing_features]
model.get_booster().plot_importance(importance_type='weight')
plt.title("Feature Importance")
plt.tight_layout()
plt.show()