import numpy as np

def run_gridworld():
    """
    Implements Iterative Policy Evaluation for a 4x4 GridWorld.
    Goal: Calculate Value Function V(s) for a random policy.
    """
    # --- 1. Initialization ---
    N = 4             # Grid Size (4x4)
    gamma = 1.0       # Discount factor (No discounting)
    theta = 1e-4      # Convergence threshold
    
    # Initialize Value Function V(s) to 0 for all states
    V = np.zeros((N, N))
    
    # Define Actions: (row_change, col_change)
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right
    
    # --- 2. Iterative Policy Evaluation Loop ---
    iteration = 0
    while True:
        delta = 0  # Track maximum change in value across all states
        V_new = np.copy(V)  # Create a copy for synchronous updates
        
        # Loop over all states
        for r in range(N):
            for c in range(N):
                # Skip Terminal State (Bottom-Right: 3, 3)
                # Value of terminal state is always 0
                if (r, c) == (N-1, N-1): 
                    continue
                
                v = V[r, c] # Current value
                new_v = 0   # Accumulator for expected value
                
                # --- 3. Bellman Equation Update ---
                # V(s) = sum( prob(a|s) * [ reward + gamma * V(s') ] )
                # Policy is equi-probable: 0.25 probability for each action
                for dr, dc in actions:
                    next_r, next_c = r + dr, c + dc
                    
                    # Boundary Checks: 
                    # If agent hits a wall, it stays in the same state (s' = s)
                    if next_r < 0 or next_r >= N or next_c < 0 or next_c >= N:
                        next_r, next_c = r, c
                    
                    reward = -1 # Reward is -1 for every step
                    
                    # Add expected return for this action
                    new_v += 0.25 * (reward + gamma * V[next_r, next_c])
                
                # Update the new value function
                V_new[r, c] = new_v
                
                # Update max change (delta)
                delta = max(delta, abs(v - V_new[r, c]))
        
        # Update Value Function for next iteration
        V = V_new
        iteration += 1
        
        # --- 4. Check Convergence ---
        if delta < theta:
            break
            
    print(f"Converged in {iteration} iterations")
    print("Final Value Function:")
    print(V)

if __name__ == "__main__":
    run_gridworld()
