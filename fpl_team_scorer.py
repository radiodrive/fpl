# FPL Team Scorer Script
#
# This script loads the master list of player predictions and calculates
# the total predicted points for a user-defined 15-player squad.
# It also validates the squad against FPL rules.
#
# Author: Gemini
# Date: August 1, 2025
#
import pandas as pd

# --- CONFIGURE YOUR TEAM HERE ---
# Enter the exact 'web_name' from the FPL site for each of your 15 players.
MY_TEAM = [
    # Goalkeepers
    'Petrović',
    'Kelleher',
    # Defenders
    'Collins',
    'Cucurella',
    'Aït-Nouri',
    'Pembele',
    'Murillo',
    # Midfielders
    'Semenyo',
    'Rogers',
    'Palmer',
    'Hudson-Odoi',
    'Savinho',
    # Forwards
    'Evanilson',
    'João Pedro',
    'Gyökeres'
]

PREDICTIONS_FILE = 'fpl_predictions.csv'
BUDGET = 100.0

def score_team():
    """
    Loads predictions and scores the user-defined team.
    """
    print("--- FPL Team Scorer ---")
    
    # 1. Load the master predictions file
    try:
        df = pd.read_csv(PREDICTIONS_FILE)
    except FileNotFoundError:
        print(f"Error: The file '{PREDICTIONS_FILE}' was not found.")
        print("Please run the 'fpl_play_point_predictor.py' script first.")
        return

    # 2. Filter the DataFrame to get only the players in your squad
    squad_df = df[df['web_name'].isin(MY_TEAM)].copy()

    # 3. Validate the squad
    print("\n--- Validating Your Squad ---")
    
    # Check if all players were found
    if len(squad_df) != 15:
        print(f"Error: Found {len(squad_df)} out of 15 players.")
        found_players = squad_df['web_name'].tolist()
        missing_players = [p for p in MY_TEAM if p not in found_players]
        if missing_players:
            print(f"Could not find: {', '.join(missing_players)}")
            print("Please check for typos in the 'MY_TEAM' list.")
        return

    # Check positional counts
    position_counts = squad_df['position'].value_counts()
    print(f"Position Counts:\n{position_counts.to_string()}")
    
    # Check team counts
    team_counts = squad_df['team'].value_counts()
    print(f"\nTeam Counts (players per team):\n{team_counts[team_counts > 3].to_string()}")
    if (team_counts > 3).any():
        print("Warning: Your squad violates the 'max 3 players per team' rule.")
    else:
        print("Team rule check: OK")
        
    # 4. Calculate totals
    total_predicted_points = squad_df['predicted_points'].sum()
    total_cost = squad_df['cost'].sum()

    # 5. Display the results
    print("\n--- Your Team's Predictions ---")
    
    # Sort for clean viewing
    pos_order = {'GK': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}
    squad_df['pos_order'] = squad_df['position'].map(pos_order)
    squad_df.sort_values(by='pos_order', inplace=True)
    squad_df.drop('pos_order', axis=1, inplace=True)
    
    print(squad_df[['web_name', 'team', 'position', 'cost', 'predicted_points']].to_string(index=False))

    print("\n--- Final Score ---")
    print(f"Total Predicted Points (Next 5 GWs): {total_predicted_points:.2f}")
    print(f"Total Cost: £{total_cost:.1f}m")
    
    if total_cost > BUDGET:
        print(f"Warning: Your team is over budget by £{total_cost - BUDGET:.1f}m!")


if __name__ == '__main__':
    score_team()