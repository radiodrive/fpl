# FPL Player Point Prediction Script (with Fixture Difficulty)
#
# This script builds a model to predict the total points a player will score
# over the next N gameweeks, now including fixture difficulty as a feature.
#
# Date: August 2, 2025
#
# --- LIBRARIES ---
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# --- CONFIGURATION ---
N_GAMEWEEKS_TO_PREDICT = 4
SEASONS = ['2024-25', '2023-24', '2022-23', '2025-26']
DATA_BASE_URL = 'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{}/gws/merged_gw.csv'
FPL_API_URL = 'https://fantasy.premierleague.com/api/bootstrap-static/'


def load_historical_data(seasons):
    """Loads historical FPL data for the specified seasons."""
    print("Loading historical data...")
    all_gws = []
    for season in seasons:
        try:
            url = DATA_BASE_URL.format(season)
            df = pd.read_csv(url)
            df['season'] = season
            all_gws.append(df)
            print(f"Successfully loaded data for {season} season.")
        except Exception as e:
            print(f"Could not load data for {season}. Error: {e}")
    if not all_gws:
        print("No data loaded. Exiting."); exit()
    return pd.concat(all_gws)

def get_live_fpl_data():
    """Fetches live data from the FPL API for current players and fixtures."""
    print("Fetching live data from FPL API...")
    try:
        response = requests.get(FPL_API_URL)
        response.raise_for_status()
        data = response.json()
        players_df = pd.DataFrame(data['elements'])
        teams_df = pd.DataFrame(data['teams'])
        
        fixtures_response = requests.get('https://fantasy.premierleague.com/api/fixtures/')
        fixtures_response.raise_for_status()
        fixtures_df = pd.DataFrame(fixtures_response.json())
        return players_df, teams_df, fixtures_df
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch data from FPL API: {e}")
        return None, None, None

def feature_engineering(df):
    """Creates target variable (y) and predictor variables (X) from the raw data."""
    print("Performing feature engineering...")
    df.sort_values(by=['season', 'name', 'GW'], inplace=True)
    
    # 1. Create the Target Variable (y)
    df['target_points'] = df.groupby(['name', 'season'])['total_points'].shift(-N_GAMEWEEKS_TO_PREDICT).rolling(window=N_GAMEWEEKS_TO_PREDICT, min_periods=N_GAMEWEEKS_TO_PREDICT).sum()
    
    # 2. Create Predictor Variables (X)
    past_window = 5
    stats_to_roll = ['goals_scored', 'assists', 'minutes', 'ict_index', 'influence', 'creativity', 'threat', 'xG_understat']
    for stat in stats_to_roll:
        df[f'{stat}_last_{past_window}'] = df.groupby(['name', 'season'])[stat].shift(1).rolling(window=past_window, min_periods=1).mean()

    df = pd.get_dummies(df, columns=['position'], drop_first=True)
    df.dropna(subset=['target_points'], inplace=True)
    df.fillna(0, inplace=True)
    
    # --- NEW: Add fixture difficulty to the list of features ---
    feature_cols = [f'{s}_last_{past_window}' for s in stats_to_roll] + \
                   ['value', 'opponent_team_difficulty', 'position_FWD', 'position_GK', 'position_MID']
    
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
            
    X = df[feature_cols]
    y = df['target_points']
    return X, y, df

def main():
    """Main function to run the entire pipeline."""
    historical_df = load_historical_data(SEASONS)

    try:
        print("Loading and merging Understat data...")
        understat_df = pd.read_csv('understat_data.csv')
        understat_df['season_fpl'] = understat_df['season'].apply(lambda x: f"{x}-{str(x+1)[-2:]}")
        historical_df = pd.merge(historical_df, understat_df[['player_name', 'season_fpl', 'xG_understat']], how='left', left_on=['name', 'season'], right_on=['player_name', 'season_fpl'])
        # Correctly handle fillna to avoid FutureWarning
        historical_df['xG_understat'] = historical_df['xG_understat'].fillna(0)
        print("Successfully merged Understat xG data.")
    except FileNotFoundError:
        print("Warning: 'understat_data.csv' not found. xG features will not be used.")
        historical_df['xG_understat'] = 0

    X_train, y_train, _ = feature_engineering(historical_df.copy())
    
    print("\nTraining the prediction model...")
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model training complete.")

    print("\n--- Model Feature Importance ---")
    importances = model.feature_importances_
    feature_names = X_train.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    importance_df['Importance'] = importance_df['Importance'] * 100
    print(importance_df.to_string(index=False))

    # --- PREPARE DATA FOR LIVE PREDICTION ---
    live_players, live_teams, live_fixtures = get_live_fpl_data()
    if live_players is None: return
        
    print("\nPreparing live data for prediction...")
    
    # --- CORRECTED: Calculate average future fixture difficulty ---
    team_fdr = {}
    # Find the first gameweek to consider, using the correct column 'event'
    future_fixtures = live_fixtures[live_fixtures['finished'] == False]
    if not future_fixtures.empty and 'event' in future_fixtures.columns:
        next_gw = future_fixtures['event'].min()
        for team_id in live_teams['id']:
            team_fixtures = live_fixtures[
                ((live_fixtures['team_h'] == team_id) | (live_fixtures['team_a'] == team_id)) &
                (live_fixtures['event'] >= next_gw)
            ].head(N_GAMEWEEKS_TO_PREDICT)
            
            difficulties = []
            for _, row in team_fixtures.iterrows():
                # The difficulty for the opponent is what matters
                difficulties.append(row['team_h_difficulty'] if row['team_a'] == team_id else row['team_a_difficulty'])
            
            team_fdr[team_id] = np.mean(difficulties) if difficulties else 3 # Default to average difficulty
    else:
        print("No future fixtures found. Defaulting FDR to 3 for all teams.")
        for team_id in live_teams['id']:
            team_fdr[team_id] = 3

    # Add this FDR to the live player data
    live_players['avg_fdr_next_5'] = live_players['team'].map(team_fdr)
    # --- END OF CORRECTION ---

    historical_df.sort_values(by=['season', 'GW'], ascending=False, inplace=True)
    latest_historical = historical_df.drop_duplicates(subset=['element'], keep='first')
    
    live_players.rename(columns={'id': 'element'}, inplace=True)
    prediction_df = pd.merge(live_players, latest_historical, on='element', how='left', suffixes=('_live', '_hist'))

    prediction_df['position'] = prediction_df['element_type'].map({1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'})
    prediction_df = pd.get_dummies(prediction_df, columns=['position'], drop_first=True)
    
    # Use the calculated average FPL for the 'opponent_team_difficulty' feature
    prediction_df['opponent_team_difficulty'] = prediction_df['avg_fdr_next_5']

    stats_to_roll = ['goals_scored', 'assists', 'minutes', 'ict_index', 'influence', 'creativity', 'threat', 'xG_understat']
    for stat in stats_to_roll:
        hist_col = f'{stat}_hist'
        if hist_col in prediction_df.columns:
            prediction_df[f'{stat}_last_5'] = prediction_df[hist_col]
        elif stat in prediction_df.columns:
            prediction_df[f'{stat}_last_5'] = prediction_df[stat]
        else:
            prediction_df[f'{stat}_last_5'] = 0
    
    prediction_df['value'] = prediction_df['now_cost'] / 10

    feature_cols = X_train.columns
    for col in feature_cols:
        if col not in prediction_df.columns:
            prediction_df[col] = 0
    prediction_df = prediction_df[feature_cols].fillna(0)
    
    print("\nMaking predictions for the next {} gameweeks...".format(N_GAMEWEEKS_TO_PREDICT))
    predictions = model.predict(prediction_df)
    
    results_df = live_players[['web_name', 'team', 'element_type', 'now_cost']].copy()
    results_df['team'] = results_df['team'].map(live_teams.set_index('id')['name'])
    results_df['position'] = results_df['element_type'].map({1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'})
    results_df['cost'] = results_df['now_cost'] / 10
    results_df['predicted_points'] = predictions
    
    final_display = results_df[['web_name', 'team', 'position', 'cost', 'predicted_points']]
    final_display = final_display.sort_values(by='predicted_points', ascending=False)
    
    final_display.to_csv('fpl_predictions_v2.csv', index=False)
    
    print("\n--- TOP 30 PLAYER PREDICTIONS (Next {} Gameweeks) ---".format(N_GAMEWEEKS_TO_PREDICT))
    print(final_display.head(30).to_string(index=False))
    
    print("\n--- SCRIPT COMPLETE ---")

if __name__ == '__main__':
    main()
