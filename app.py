# app.py
#
# An interactive Streamlit web application to run the FPL prediction and optimization pipeline.
#
# To Run:
# 1. Make sure all your scripts (scrape_understat.py, etc.) and data files are in the same directory.
# 2. In your terminal, run: streamlit run app.py
#
import streamlit as st
import pandas as pd
import numpy as np
import requests
import pulp
import subprocess
import sys
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="FPL AI Team Picker",
    page_icon="⚽",
    layout="wide"
)

# --- Constants ---
FPL_API_URL = 'https://fantasy.premierleague.com/api/bootstrap-static/'
DATA_BASE_URL = 'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{}/gws/merged_gw.csv'
UNDERSTAT_FILE = 'understat_data.csv'
PREDICTIONS_FILE = 'fpl_predictions.csv'

# --- Caching Functions ---
# Use caching to avoid re-downloading data on every app interaction.
@st.cache_data
def load_historical_data(seasons):
    all_gws = []
    for season in seasons:
        try:
            url = DATA_BASE_URL.format(season)
            df = pd.read_csv(url)
            df['season'] = season
            all_gws.append(df)
        except Exception:
            st.warning(f"Could not load historical data for {season} season.")
    return pd.concat(all_gws) if all_gws else pd.DataFrame()

@st.cache_data
def load_understat_data():
    try:
        return pd.read_csv(UNDERSTAT_FILE)
    except FileNotFoundError:
        return None

@st.cache_data
def get_live_fpl_data():
    response = requests.get(FPL_API_URL)
    if response.status_code != 200:
        return None, None
    data = response.json()
    players_df = pd.DataFrame(data['elements'])
    teams_df = pd.DataFrame(data['teams'])
    return players_df, teams_df

# --- Core Logic Functions (Adapted from your scripts) ---
def feature_engineering(df, model_type='involvement', n_gameweeks=5):
    df.sort_values(by=['season', 'name', 'GW'], inplace=True)
    df['target_points'] = df.groupby(['name', 'season'])['total_points'].shift(-n_gameweeks).rolling(window=n_gameweeks, min_periods=n_gameweeks).sum()
    
    past_window = 5
    stats_to_roll = ['goals_scored', 'assists', 'minutes', 'ict_index', 'influence', 'creativity', 'threat', 'xG_understat']
    if model_type == 'form':
        stats_to_roll.append('total_points')
        
    for stat in stats_to_roll:
        df[f'{stat}_last_{past_window}'] = df.groupby(['name', 'season'])[stat].shift(1).rolling(window=past_window, min_periods=1).mean()

    df = pd.get_dummies(df, columns=['position'], drop_first=True)
    df.dropna(subset=['target_points'], inplace=True)
    df.fillna(0, inplace=True)
    
    feature_cols = [f'{s}_last_{past_window}' for s in stats_to_roll] + ['value', 'position_FWD', 'position_GK', 'position_MID']
    
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
            
    X = df[feature_cols]
    y = df['target_points']
    return X, y

def train_model(X_train, y_train, algorithm='XGBoost'):
    if algorithm == 'XGBoost':
        model = XGBRegressor(n_estimators=100, random_state=42)
    else: # RandomForest
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def solve_fpl_squad(predictions_df, budget=100.0, blacklist=[]):
    df = predictions_df[~predictions_df['web_name'].isin(blacklist)].reset_index(drop=True)

    prob = pulp.LpProblem("FPL_Team_Optimization", pulp.LpMaximize)
    player_vars = pulp.LpVariable.dicts("Player", df.index, cat='Binary')
    prob += pulp.lpSum([df.loc[i, 'predicted_points'] * player_vars[i] for i in df.index]), "Total Predicted Points"
    prob += pulp.lpSum([df.loc[i, 'cost'] * player_vars[i] for i in df.index]) <= budget, "Total Cost"
    prob += pulp.lpSum([player_vars[i] for i in df.index]) == 15, "Total Players"
    
    positions = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
    for pos, limit in positions.items():
        prob += pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, 'position'] == pos]) == limit, f"Total {pos}s"
    
    teams = df['team'].unique()
    for team_name in teams:
        prob += pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, 'team'] == team_name]) <= 3, f"Max 3 from {team_name}"

    prob.solve(pulp.PULP_CBC_CMD(msg=0)) # Suppress solver messages
    
    if pulp.LpStatus[prob.status] == 'Optimal':
        selected_indices = [i for i in df.index if player_vars[i].varValue == 1]
        return df.loc[selected_indices]
    return pd.DataFrame()


# --- Streamlit App UI ---
st.title("⚽ FPL AI Team Picker")
st.write("An interactive dashboard to build your FPL squad using data science.")

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("⚙️ Controls")
    
    st.subheader("1. Data Scraping")
    if st.button("Scrape Latest Understat Data"):
        with st.spinner("Scraping Understat... This may take a few minutes."):
            # We run the script as a separate process
            process = subprocess.run([f"{sys.executable}", "scrape_understat.py"], capture_output=True, text=True)
            if process.returncode == 0:
                st.success("Understat data scraped successfully!")
                st.text_area("Scraper Log", process.stdout, height=200)
                # Clear cache to force a reload
                st.cache_data.clear()
            else:
                st.error("Scraping failed.")
                st.text_area("Error Log", process.stderr, height=200)

    st.subheader("2. Model Configuration")
    model_type = st.selectbox("Select Model Philosophy", ["Involvement", "Form"], help="**Involvement:** A balanced model using all stats. **Form:** A model that heavily prioritizes recent FPL points.")
    algorithm = st.selectbox("Select Algorithm", ["XGBoost", "RandomForest"])
    n_gameweeks = st.slider("Gameweeks to Predict Ahead", 1, 8, 5)

# --- Main Page Layout ---
col1, col2 = st.columns([0.6, 0.4])

with col1:
    st.header("📊 Model Predictions")
    if st.button("Train Model & Generate Predictions"):
        with st.spinner("Processing data and training model..."):
            # Load all data
            historical_df = load_historical_data(['2023-24', '2022-23'])
            understat_df = load_understat_data()
            live_players, live_teams = get_live_fpl_data()

            if historical_df.empty or live_players is None:
                st.error("Could not load necessary FPL data. Aborting.")
            else:
                # Merge data
                if understat_df is not None:
                    understat_df['season_fpl'] = understat_df['season'].apply(lambda x: f"{x}-{str(x+1)[-2:]}")
                    historical_df = pd.merge(historical_df, understat_df[['player_name', 'season_fpl', 'xG_understat']], how='left', left_on=['name', 'season'], right_on=['player_name', 'season_fpl'])
                else:
                    st.warning("Understat data not found. Running without xG features.")
                    historical_df['xG_understat'] = 0

                # Feature Engineering & Training
                X_train, y_train = feature_engineering(historical_df.copy(), model_type.lower(), n_gameweeks)
                model = train_model(X_train, y_train, algorithm)
                
                # Prepare live data for prediction
                historical_df.sort_values(by=['season', 'GW'], ascending=False, inplace=True)
                latest_historical = historical_df.drop_duplicates(subset=['element'], keep='first')
                live_players.rename(columns={'id': 'element'}, inplace=True)
                prediction_df = pd.merge(live_players, latest_historical, on='element', how='left', suffixes=('_live', '_hist'))
                prediction_df['position'] = prediction_df['element_type'].map({1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'})
                prediction_df = pd.get_dummies(prediction_df, columns=['position'], drop_first=True)
                prediction_df['value'] = prediction_df['now_cost'] / 10
                
                # --- THIS BLOCK IS CORRECTED TO PREVENT THE KEYERROR ---
                # Define the base stats the model needs
                base_feature_stats = ['goals_scored', 'assists', 'minutes', 'ict_index', 'influence', 'creativity', 'threat', 'xG_understat']
                if model_type.lower() == 'form':
                    base_feature_stats.append('total_points')

                # Prepare the features for the live prediction by checking for suffixed and unsuffixed columns
                for stat in base_feature_stats:
                    hist_col_name = f'{stat}_hist'
                    # Check if the suffixed column exists (for standard FPL stats)
                    if hist_col_name in prediction_df.columns:
                        prediction_df[f'{stat}_last_5'] = prediction_df[hist_col_name]
                    # Fallback to the unsuffixed column (for our merged xG stat)
                    elif stat in prediction_df.columns:
                        prediction_df[f'{stat}_last_5'] = prediction_df[stat]
                    # If neither exists, create a column of zeros
                    else:
                        prediction_df[f'{stat}_last_5'] = 0
                # --- END OF CORRECTION ---

                for col in X_train.columns:
                    if col not in prediction_df.columns:
                        prediction_df[col] = 0
                prediction_df = prediction_df[X_train.columns].fillna(0)

                # Make predictions
                predictions = model.predict(prediction_df)
                
                # Store results in session state
                results_df = live_players[['web_name', 'team', 'element_type', 'now_cost']].copy()
                results_df['team'] = results_df['team'].map(live_teams.set_index('id')['name'])
                results_df['position'] = results_df['element_type'].map({1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'})
                results_df['cost'] = results_df['now_cost'] / 10
                results_df['predicted_points'] = predictions
                st.session_state['predictions'] = results_df.sort_values(by='predicted_points', ascending=False)
                
                # Store feature importances
                importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
                st.session_state['importances'] = importance_df
                
                st.success("Model trained and predictions generated!")

    if 'predictions' in st.session_state:
        st.dataframe(st.session_state['predictions'].head(20))
    else:
        st.info("Click the button above to generate player predictions.")

with col2:
    st.header("🧠 Model Insights")
    if 'importances' in st.session_state:
        fig = px.bar(st.session_state['importances'].head(15), 
                     x='Importance', 
                     y='Feature', 
                     orientation='h',
                     title='Top 15 Most Important Features')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Train a model to see its feature importances.")

st.divider()

st.header("⭐ Optimized FPL Squad")
if 'predictions' in st.session_state:
    all_players = st.session_state['predictions']['web_name'].unique().tolist()
    blacklist = st.multiselect("Select players to EXCLUDE from optimization:", all_players, key="blacklist_select")
    
    if st.button("Optimize Squad", type="primary"):
        with st.spinner("Finding the optimal squad..."):
            optimized_squad = solve_fpl_squad(st.session_state['predictions'], blacklist=blacklist)
            st.session_state['optimized_squad'] = optimized_squad

    if 'optimized_squad' in st.session_state:
        squad = st.session_state['optimized_squad']
        if not squad.empty:
            total_points = squad['predicted_points'].sum()
            total_cost = squad['cost'].sum()
            
            st.subheader(f"Total Predicted Points: {total_points:.2f} | Total Cost: £{total_cost:.1f}m")
            
            # Sort for display
            pos_order = {'GK': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}
            squad['pos_order'] = squad['position'].map(pos_order)
            squad.sort_values(by='pos_order', inplace=True)
            
            st.dataframe(squad[['web_name', 'team', 'position', 'cost', 'predicted_points']])
        else:
            st.error("Could not find an optimal solution with the given constraints.")
else:
    st.info("Generate predictions first to enable squad optimization.")
