# fpl_transfer_optimizer.py
#
# This script takes your current FPL squad and suggests the optimal transfers
# to maximize your predicted points for the next gameweek.
#
# Date: August 2, 2025
#
import pandas as pd
import pulp

# --- 1. CONFIGURE YOUR TEAM AND TRANSFERS ---

# Enter the exact 'web_name' for each of your 15 players.
CURRENT_SQUAD_NAMES = [
    'Areola', 'Turner', 'Alexander-Arnold', 'Saliba', 'Udogie', 'Cash', 'Gusto',
    'Saka', 'Foden', 'Palmer', 'Eze', 'Garnacho', 'Haaland', 'Watkins', 'Solanke'
]

# Enter the amount of money you have in the bank (e.g., 0.5 for £0.5m).
MONEY_IN_THE_BANK = 0.5

# Enter the number of free transfers you want to make (usually 1 or 2).
# The script will find the best combination of transfers up to this number.
NUM_TRANSFERS = 1

# Set to True if you want to consider taking a -4 hit for an extra transfer.
# This will only be recommended if the net point gain is positive.
CONSIDER_POINT_HITS = True 

# --- 2. SCRIPT CONFIGURATION ---
PREDICTIONS_FILE = 'fpl_predictions.csv'
POINTS_HIT_COST = 4

def solve_transfers():
    """
    Loads player data and solves for the optimal transfers.
    """
    print("--- FPL Weekly Transfer Optimizer ---")
    
    # Load player predictions (ensure this is for the next gameweek only)
    try:
        df = pd.read_csv(PREDICTIONS_FILE)
    except FileNotFoundError:
        st.error(f"Error: The file '{PREDICTIONS_FILE}' was not found.")
        st.info("Please run the prediction script with N_GAMEWEEKS_TO_PREDICT = 1.")
        return

    # Identify players in your current squad from the main dataframe
    squad_df = df[df['web_name'].isin(CURRENT_SQUAD_NAMES)].copy()
    if len(squad_df) != 15:
        print("Error: Could not find all 15 players from your squad in the predictions file. Please check names.")
        return

    # Calculate your total budget for the new team
    current_squad_value = squad_df['cost'].sum()
    budget = current_squad_value + MONEY_IN_THE_BANK
    print(f"\nYour current squad value: £{current_squad_value:.1f}m")
    print(f"Your total budget for the new team: £{budget:.1f}m")

    # --- Set up the optimization problem ---
    prob = pulp.LpProblem("FPL_Transfer_Optimization", pulp.LpMaximize)

    # Decision Variables
    # player_vars: 1 if player `i` is in the NEW squad, 0 otherwise
    player_vars = pulp.LpVariable.dicts("Player", df.index, cat='Binary')
    # players_to_sell: 1 if player `j` (from your current squad) is sold, 0 otherwise
    players_to_sell = pulp.LpVariable.dicts("Sell", squad_df.index, cat='Binary')

    # Objective Function: Maximize points of the new team, minus any point hits
    point_hits = 0
    if CONSIDER_POINT_HITS and NUM_TRANSFERS > 1: # Assuming 1 free transfer
        # This calculates the cost of any transfers beyond the first free one
        point_hits = (pulp.lpSum(players_to_sell) - 1) * POINTS_HIT_COST
        
    prob += pulp.lpSum([df.loc[i, 'predicted_points'] * player_vars[i] for i in df.index]) - point_hits, "Total Net Points"

    # --- Constraints ---
    # New squad must adhere to all FPL rules
    prob += pulp.lpSum([df.loc[i, 'cost'] * player_vars[i] for i in df.index]) <= budget, "Budget"
    prob += pulp.lpSum(player_vars) == 15, "Squad Size"
    
    positions = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
    for pos, limit in positions.items():
        prob += pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, 'position'] == pos]) == limit, f"Total {pos}s"
    
    teams = df['team'].unique()
    for team_name in teams:
        prob += pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, 'team'] == team_name]) <= 3, f"Max 3 from {team_name}"

    # Transfer Constraints
    # Link the players we sell to the total number of transfers allowed
    prob += pulp.lpSum(players_to_sell) == NUM_TRANSFERS, "Number of Transfers"

    # For each player in the original squad, they are either kept or sold.
    # If kept: player_vars[j] = 1 and players_to_sell[j] = 0
    # If sold: player_vars[j] = 0 and players_to_sell[j] = 1
    for j in squad_df.index:
        prob += player_vars[j] + players_to_sell[j] == 1, f"Keep_Or_Sell_{squad_df.loc[j, 'web_name']}"

    # --- Solve ---
    print(f"\nOptimizing for {NUM_TRANSFERS} transfer(s)...")
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # --- Display Results ---
    if pulp.LpStatus[prob.status] == 'Optimal':
        print("\n--- Optimal Transfers Found ---")
        new_squad_indices = [i for i in df.index if player_vars[i].varValue == 1]
        new_squad_df = df.loc[new_squad_indices]

        players_out = squad_df.loc[[j for j in squad_df.index if players_to_sell[j].varValue == 1]]
        players_in = new_squad_df[~new_squad_df['web_name'].isin(squad_df['web_name'])]

        print("\nPlayers to SELL:")
        if not players_out.empty:
            print(players_out[['web_name', 'team', 'cost']].to_string(index=False))
        else:
            print("None")

        print("\nPlayers to BUY:")
        if not players_in.empty:
            print(players_in[['web_name', 'team', 'cost', 'predicted_points']].to_string(index=False))
        else:
            print("None")
            
        net_points_gain = pulp.value(prob.objective) - squad_df['predicted_points'].sum()
        print(f"\nPredicted Net Points Gain from transfers: {net_points_gain:.2f}")

    else:
        print("Could not find an optimal solution.")


if __name__ == '__main__':
    solve_transfers()