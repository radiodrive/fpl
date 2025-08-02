import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import xgboost as xgb

# === Load training data ===
train_df = pd.read_csv('combined_gameweeks.csv')

# === Feature cleanup ===
alias_map = {
    'Spurs': 'Tottenham',
    'Man City': 'Manchester City',
    'Man Utd': 'Manchester United',
    'Wolves': 'Wolverhampton Wanderers',
    "Nott'm Forest": 'Nottingham Forest'
}
train_df['opponent_name'] = train_df['opponent_name'].replace(alias_map)

# === Select features and label ===
features = [
    'was_home', 'minutes', 'ict_index', 'influence', 'creativity',
    'threat', 'value', 'fixture_difficulty'
]
label = 'total_points'

train_df = train_df.dropna(subset=features + [label])
X = train_df[features]
y = train_df[label]

# === Train model ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\n📉 MAE: {mae:.2f}")
print(f"📈 R² Score: {r2:.3f}")

# === Feature importance plot ===
xgb.plot_importance(model, height=0.5)
plt.title("Feature Importance (XGBoost)")
plt.tight_layout()
plt.show()

# === Predict future gameweek points ===
future_players = pd.read_csv('future_players.csv')
future_players = future_players.dropna(subset=features)
future_players['predicted_points'] = model.predict(future_players[features])

top_preds = future_players[['name', 'team', 'position', 'predicted_points']].sort_values(by='predicted_points', ascending=False).head(15)
print("\n🔮 Top Predicted Scorers for Upcoming GW:")
print(top_preds.to_string(index=False))