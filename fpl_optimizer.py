# FPL Squad Optimization Script
#
# This script reads player predictions from a CSV file and uses linear programming
# to select the optimal 15-man squad that maximizes total predicted points
# while adhering to all FPL rules.
#
# Author: Gemini
# Date: August 1, 2025
#
# --- LIBRARIES ---
import pandas as pd
import pulp

# --- CONFIGURATION ---
PREDICTIONS_FILE = 'fpl_predictions.csv'
BUDGET = 100.0

def solve_fpl_squad():
    """
    Loads player data and solves for the optimal FPL squad.
    """
    # 1. Load the Data
    print(f"Loading player predictions from '{PREDICTIONS_FILE}'...")
    try:
        df = pd.read_csv(PREDICTIONS_FILE)
    except FileNotFoundError:
        print(f"Error: The file '{PREDICTIONS_FILE}' was not found.")
        print("Please run the prediction script first to generate this file.")
        return

    # 2. Initialize the Problem
    print("Setting up the optimization problem...")
    prob = pulp.LpProblem("FPL_Team_Optimization", pulp.LpMaximize)

    # 3. Define Decision Variables
    # A dictionary of binary variables (0 or 1) for each player
    # 1 if the player is in the squad, 0 if they are not.
    player_vars = pulp.LpVariable.dicts("Player", df.index, cat='Binary')

    # 4. Set the Objective Function
    # We want to maximize the sum of predicted_points for the selected players.
    prob += pulp.lpSum([df.loc[i, 'predicted_points'] * player_vars[i] for i in df.index]), "Total Predicted Points"

    # 5. Add Constraints
    print("Adding FPL rules as constraints...")

    # Budget Constraint
    prob += pulp.lpSum([df.loc[i, 'cost'] * player_vars[i] for i in df.index]) <= BUDGET, "Total Cost"

    # Squad Size Constraint (Exactly 15 players)
    prob += pulp.lpSum([player_vars[i] for i in df.index]) == 15, "Total Players"

    # Positional Constraints
    positions = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
    for pos, limit in positions.items():
        prob += pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, 'position'] == pos]) == limit, f"Total {pos}s"

    # Team Constraint (Max 3 players from any single team)
    teams = df['team'].unique()
    for team_name in teams:
        prob += pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, 'team'] == team_name]) <= 3, f"Max 3 from {team_name}"

    # 6. Solve the Problem
    print("Solving the problem... (This may take a moment)")
    prob.solve()
    status = pulp.LpStatus[prob.status]
    print(f"Status: {status}")

    # 7. Display the Results
    if status == 'Optimal':
        print("\n--- Optimal FPL Squad Found ---")
        
        selected_players = []
        for i in df.index:
            if player_vars[i].varValue == 1:
                selected_players.append(df.loc[i])
        
        squad_df = pd.DataFrame(selected_players)
        
        # Sort by position for clean viewing
        pos_order = {'GK': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}
        squad_df['pos_order'] = squad_df['position'].map(pos_order)
        squad_df.sort_values(by='pos_order', inplace=True)
        squad_df.drop('pos_order', axis=1, inplace=True)
        
        print(squad_df[['web_name', 'team', 'position', 'cost', 'predicted_points']].to_string(index=False))

        total_predicted_points = pulp.value(prob.objective)
        total_cost = squad_df['cost'].sum()

        print("\n--- Summary ---")
        print(f"Total Predicted Points: {total_predicted_points:.2f}")
        print(f"Total Cost: £{total_cost:.1f}m")
    else:
        print("Could not find an optimal solution.")


if __name__ == '__main__':
    solve_fpl_squad()