import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import xgboost as xgb

# === Load Historical Data (for training) ===
df = pd.read_csv('combined_gameweeks.csv')

# === Clean & Map Team Aliases ===
alias_map = {
    'Spurs': 'Tottenham',
    'Man City': 'Manchester City',
    'Man Utd': 'Manchester United',
    'Wolves': 'Wolverhampton Wanderers',
    'Nott\'m Forest': 'Nottingham Forest'
}
df['opponent_name'] = df['opponent_name'].replace(alias_map)

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
df['fixture_difficulty'] = df['opponent_name'].map(fixture_difficulty)

# === Features & Label ===
features = ['was_home', 'minutes', 'ict_index', 'influence', 'creativity', 'threat', 'value', 'fixture_difficulty']
label = 'total_points'
df_model = df.dropna(subset=features + [label])

# === Train/Test Split ===
X = df_model[features]
y = df_model[label]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train Model ===
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# === Evaluate on test data ===
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"📉 MAE: {mae:.2f}")
print(f"📈 R² Score: {r2:.3f}")

# === Load Future GW1 Players ===
future_players = pd.read_csv('future_gw1_players.csv')

# === Drop missing data and predict ===
future_players = future_players.dropna(subset=features)
future_players['predicted_points'] = model.predict(future_players[features])

# === Output Top Predictions ===
top_preds = future_players[['Name', 'Team', 'Position', 'predicted_points']].sort_values(by='predicted_points', ascending=False).head(15)
print("\n🔮 Top Predicted GW1 Scorers:")
print(top_preds.to_string(index=False))

# === Plot feature importance ===
xgb.plot_importance(model, height=0.5)
plt.title("Feature Importance (XGBoost)")
plt.tight_layout()
plt.show()