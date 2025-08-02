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
    stats_to_roll = ['total_points', 'goals_scored', 'assists', 'minutes', 
                     'ict_index', 'influence', 'creativity', 'threat',
                     'xG_understat'] # Only xG remains from Understat
                     
    for stat in stats_to_roll:
        df[f'{stat}_last_{past_window}'] = df.groupby(['name', 'season'])[stat].shift(1).rolling(window=past_window, min_periods=1).mean()

    df = pd.get_dummies(df, columns=['position'], drop_first=True)

    df.dropna(subset=['target_points'], inplace=True)
    df.fillna(0, inplace=True)
    
    feature_cols = [f'{s}_last_{past_window}' for s in stats_to_roll] + \
                   ['value', 'position_FWD', 'position_GK', 'position_MID']
    
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
    # START: BLOCK TO ADD
    # The new code for loading and merging the xG data goes right here.
    # ===================================================================
    try:
        print("Loading and merging Understat data...")
        understat_df = pd.read_csv('understat_data.csv')
        # Understat seasons are e.g. 2022, FPL seasons are 2022-23. We need to align them.
        # Create a new column for merging that matches the FPL season format.
        understat_df['season_fpl'] = understat_df['season'].apply(lambda x: f"{x}-{str(x+1)[-2:]}")
        
        # Merge the two dataframes.
        # We use a 'left' merge to keep all FPL players, even if they have no Understat data.
        historical_df = pd.merge(historical_df, 
                                 understat_df[['player_name', 'season_fpl', 'xG_understat']], 
                                 how='left', 
                                 left_on=['name', 'season'], 
                                 right_on=['player_name', 'season_fpl'])
                                 
        # For players where there was no match, fill the xG data with 0.
        historical_df['xG_understat'].fillna(0, inplace=True)
        print("Successfully merged Understat xG data.")

    except FileNotFoundError:
        print("Warning: 'understat_data.csv' not found. xG features will not be used.")
        # If the file doesn't exist, create an empty 'xG_understat' column so the script doesn't crash.
        historical_df['xG_understat'] = 0
    # ===================================================================
    # END: BLOCK TO ADD
    # ===================================================================

    # 2. Now we pass the newly merged dataframe to the feature engineering function.
    X_train, y_train, _ = feature_engineering(historical_df.copy())
    
    # 3. The rest of your main() function continues as before...
    # Train the model
    print("\nTraining the prediction model...")
    # model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model training complete.")

    # --- DISPLAY FEATURE IMPORTANCE ---
    print("\n--- Model Feature Importance ---")
    importances = model.feature_importances_
    feature_names = X_train.columns
    
    # Create a DataFrame for better visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Calculate as a percentage
    importance_df['Importance'] = importance_df['Importance'] * 100
    
    print("The model bases its predictions on these features (most important first):")
    print(importance_df.to_string(index=False))
    # --- END OF NEW SECTION ---

    # --- PREPARE DATA FOR LIVE PREDICTION ---
    live_players, live_fixtures, live_teams = get_live_fpl_data()
    if live_players is None:
        return
        
    print("\nPreparing live data for prediction...")
    
    # Get the single latest historical record for each player
    historical_df.sort_values(by=['season', 'GW'], ascending=False, inplace=True)
    latest_historical = historical_df.drop_duplicates(subset=['element'], keep='first')
    
    live_players.rename(columns={'id': 'element'}, inplace=True)
    prediction_df = pd.merge(live_players, latest_historical, on='element', how='left', suffixes=('_live', '_hist'))

    # One-hot encode position
    prediction_df['position'] = prediction_df['element_type'].map({1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'})
    prediction_df = pd.get_dummies(prediction_df, columns=['position'], drop_first=True)

    # Use historical rolling features for prediction
    past_window = 5
    # This list must match the one in feature_engineering
    stats_to_roll = ['total_points', 'goals_scored', 'assists', 'minutes', 'ict_index', 'influence', 'creativity', 'threat']
    for stat in stats_to_roll:
         prediction_df[f'{stat}_last_{past_window}'] = prediction_df[f'{stat}_hist']
    
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
    final_display.to_csv('fpl_predictions_v1.csv', index=False)
    
    print("\n--- TOP 30 PLAYER PREDICTIONS (Next {} Gameweeks) ---")
    print(final_display.head(30).to_string(index=False))
    
    print("\n--- SCRIPT COMPLETE ---")
    print("Next step: Use these predictions in an optimization script (e.g., with PuLP) to build your squad.")


if __name__ == '__main__':
    main()