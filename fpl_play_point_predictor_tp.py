# FPL Player Point Prediction Script
#
# This script builds a model to predict the total points a player will score
# over the next N gameweeks.
#
# Date: August 1, 2025
#
# --- LIBRARIES ---
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# --- CONFIGURATION ---
# The number of future gameweeks to predict points for.
N_GAMEWEEKS_TO_PREDICT = 4

# Seasons to use for training the model. More data is generally better.
SEASONS = ['2024-25', '2023-24', '2022-23', '2025-26']

# URL for historical data from a public GitHub repository
DATA_BASE_URL = 'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{}/gws/merged_gw.csv'

# URL for the live FPL API
FPL_API_URL = 'https://fantasy.premierleague.com/api/bootstrap-static/'


def load_historical_data(seasons):
    """
    Loads historical FPL data for the specified seasons.
    """
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
        print("No data loaded. Exiting.")
        exit()
        
    return pd.concat(all_gws)

def get_live_fpl_data():
    """
    Fetches live data from the FPL API for current players and fixtures.
    """
    print("Fetching live data from FPL API...")
    response = requests.get(FPL_API_URL)
    if response.status_code != 200:
        print("Failed to fetch data from FPL API.")
        return None, None
    
    data = response.json()
    players_df = pd.DataFrame(data['elements'])
    teams_df = pd.DataFrame(data['teams'])
    
    # Get future fixtures to calculate future difficulty
    fixtures_response = requests.get('https://fantasy.premierleague.com/api/fixtures/')
    fixtures_df = pd.DataFrame(fixtures_response.json())

    return players_df, fixtures_df, teams_df

def feature_engineering(df):
    """
    Creates target variable (y) and predictor variables (X) from the raw data.
    """
    print("Performing feature engineering...")
    
    df.sort_values(by=['season', 'name', 'GW'], inplace=True)
    
    # 1. Create the Target Variable (y)
    df['target_points'] = df.groupby(['name', 'season'])['total_points'].shift(-N_GAMEWEEKS_TO_PREDICT).rolling(window=N_GAMEWEEKS_TO_PREDICT, min_periods=N_GAMEWEEKS_TO_PREDICT).sum()
    
    # 2. Create Predictor Variables (X)
    past_window = 5
    
    # --- UPDATED LIST OF STATS (REMOVED xA and npxG as they aren't available) ---
    stats_to_roll = [ 'goals_scored', 'assists', 'minutes', 
                     'ict_index', 'influence', 'creativity', 'threat',
                     'xG_understat'] # Only xG remains from Understat
                     
    for stat in stats_to_roll:
        df[f'{stat}_last_{past_window}'] = df.groupby(['name', 'season'])[stat].shift(1).rolling(window=past_window, min_periods=1).mean()

    df = pd.get_dummies(df, columns=['position'], drop_first=True)

    df.dropna(subset=['target_points'], inplace=True)
    df.fillna(0, inplace=True)
    
    feature_cols = [f'{s}_last_{past_window}' for s in stats_to_roll] + \
                   ['value', 'opponent_team_difficulty','position_FWD', 'position_GK', 'position_MID']
    
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
            
    X = df[feature_cols]
    y = df['target_points']
    
    return X, y, df

def main():
    """
    Main function to run the entire pipeline.
    """
    # 1. This first part loads the main FPL historical data.
    historical_df = load_historical_data(SEASONS)

    # ===================================================================
    # Load and merge the xG data
    # ===================================================================
    try:
        print("Loading and merging Understat data...")
        understat_df = pd.read_csv('understat_data.csv')
        understat_df['season_fpl'] = understat_df['season'].apply(lambda x: f"{x}-{str(x+1)[-2:]}")
        historical_df = pd.merge(historical_df, 
                                 understat_df[['player_name', 'season_fpl', 'xG_understat']], 
                                 how='left', 
                                 left_on=['name', 'season'], 
                                 right_on=['player_name', 'season_fpl'])
        historical_df['xG_understat'].fillna(0, inplace=True)
        print("Successfully merged Understat xG data.")
    except FileNotFoundError:
        print("Warning: 'understat_data.csv' not found. xG features will not be used.")
        historical_df['xG_understat'] = 0
    # ===================================================================

    # 2. Now we pass the newly merged dataframe to the feature engineering function.
    X_train, y_train, _ = feature_engineering(historical_df.copy())
    
    # 3. The rest of your main() function continues as before...
    print("\nTraining the prediction model...")
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model training complete.")

    # --- DISPLAY FEATURE IMPORTANCE ---
    print("\n--- Model Feature Importance ---")
    importances = model.feature_importances_
    feature_names = X_train.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    importance_df['Importance'] = importance_df['Importance'] * 100
    print("The model bases its predictions on these features (most important first):")
    print(importance_df.to_string(index=False))

    # --- PREPARE DATA FOR LIVE PREDICTION ---
    live_players, live_fixtures, live_teams = get_live_fpl_data()
    if live_players is None:
        return
        
    print("\nPreparing live data for prediction...")

     # --- NEW: Calculate average future fixture difficulty ---
    team_fdr = {}
    # Find the first gameweek to consider
    next_gw = live_fixtures[live_fixtures['finished'] == False]['gameweek'].min()
    for team_id in live_teams['id']:
        team_fixtures = live_fixtures[
            ((live_fixtures['team_h'] == team_id) | (live_fixtures['team_a'] == team_id)) &
            (live_fixtures['gameweek'] >= next_gw)
        ].head(N_GAMEWEEKS_TO_PREDICT)
        
        difficulties = []
        for _, row in team_fixtures.iterrows():
            difficulties.append(row['team_h_difficulty'] if row['team_a'] == team_id else row['team_a_difficulty'])
        
        team_fdr[team_id] = np.mean(difficulties) if difficulties else 3 # Default to average difficulty

    # Add this FDR to the live player data
    live_players['avg_fdr_next_5'] = live_players['team'].map(team_fdr)
    # --- END OF NEW BLOCK ---
    
    historical_df.sort_values(by=['season', 'GW'], ascending=False, inplace=True)
    latest_historical = historical_df.drop_duplicates(subset=['element'], keep='first')
    
    live_players.rename(columns={'id': 'element'}, inplace=True)
    prediction_df = pd.merge(live_players, latest_historical, on='element', how='left', suffixes=('_live', '_hist'))

    prediction_df['position'] = prediction_df['element_type'].map({1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'})
    prediction_df = pd.get_dummies(prediction_df, columns=['position'], drop_first=True)

    # Use the calculated average FPL for the 'opponent_team_difficulty' feature
    prediction_df['opponent_team_difficulty'] = prediction_df['avg_fdr_next_5']


    # --- THIS BLOCK IS NOW CORRECTED ---
    # This list now correctly matches the one in feature_engineering
    stats_to_roll = ['goals_scored', 'assists', 'minutes', 'ict_index', 
                     'influence', 'creativity', 'threat', 'xG_understat']
                     
    for stat in stats_to_roll:
        # Use a robust way to get the historical stat, checking for the '_hist' suffix first
        hist_col = f'{stat}_hist'
        if hist_col in prediction_df.columns:
            prediction_df[f'{stat}_last_5'] = prediction_df[hist_col]
        elif stat in prediction_df.columns:
            prediction_df[f'{stat}_last_5'] = prediction_df[stat]
        else:
            prediction_df[f'{stat}_last_5'] = 0
    # --- END OF CORRECTION ---
    
    prediction_df['value'] = prediction_df['now_cost'] / 10

    # Ensure all feature columns exist and fill NaNs
    feature_cols = X_train.columns
    for col in feature_cols:
        if col not in prediction_df.columns:
            prediction_df[col] = 0
    prediction_df = prediction_df[feature_cols].fillna(0)
    
    # Make predictions
    print("\nMaking predictions for the next {} gameweeks...".format(N_GAMEWEEKS_TO_PREDICT))
    predictions = model.predict(prediction_df)
    
    # Display results
    results_df = live_players[['web_name', 'team', 'element_type', 'now_cost']].copy()
    results_df['team'] = results_df['team'].map(live_teams.set_index('id')['name'])
    results_df['position'] = results_df['element_type'].map({1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'})
    results_df['cost'] = results_df['now_cost'] / 10
    results_df['predicted_points'] = predictions
    
    final_display = results_df[['web_name', 'team', 'position', 'cost', 'predicted_points']]
    final_display = final_display.sort_values(by='predicted_points', ascending=False)
    
    # Save the predictions to a CSV file for the optimizer
    final_display.to_csv('fpl_predictions_v2.csv', index=False)
    
    print("\n--- TOP 30 PLAYER PREDICTIONS (Next {} Gameweeks) ---")
    print(final_display.head(30).to_string(index=False))
    
    print("\n--- SCRIPT COMPLETE ---")
    print("Next step: Use these predictions in an optimization script (e.g., with PuLP) to build your squad.")

if __name__ == '__main__':
    main()