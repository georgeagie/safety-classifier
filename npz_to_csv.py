import numpy as np
import pandas as pd
import os

def npz_to_csv_multiple_arrays(npz_file_path, csv_file_path):
    """
    Loads data from a .npz file with multiple keys and saves the combined data to a .csv file.
    
    Args:
        npz_file_path (str): The path to the input .npz file (e.g., 'data/safe_profile.npz').
        csv_file_path (str): The path where the output .csv file should be saved.
    """
    try:
        with np.load(npz_file_path) as data_archive:
            trajectory_data = data_archive['trajectory']
            labels_data = data_archive['labels']
            slack_data = data_archive['slack']
            obstacle_state = data_archive['obstacle']
            
        # Determine the number of time steps from the main trajectory data
        time_steps = trajectory_data.shape[0]

        # --- FIX: Aligning Dimensions ---
        
        # 1. Reshape time-series data to be column vectors (if they aren't already 2D)
        labels_col = labels_data.reshape(-1, 1)
        slack_col = slack_data.reshape(-1, 1)
        
        # 2. Replicate the static obstacle state for every time step
        # The obstacle_state array is 1D (e.g., [5.5, 5]), so we expand it and tile it.
        # First, ensure it's a 1-row array (1, 2)
        obstacle_row = obstacle_state.reshape(1, -1) 
        
        # Then, tile (repeat) this row 'time_steps' times to match the trajectory length
        obstacle_repeated = np.tile(obstacle_row, (time_steps, 1))
        
        # --- Concatenation ---
        
        combined_data = np.concatenate([
            trajectory_data,       # [state_x, state_y, action_x, action_y] (50 rows)
            slack_col,             # [slack_value] (50 rows)
            labels_col,            # [label_safety_critical] (50 rows)
            obstacle_repeated      # [obstacle_x, obstacle_y] (50 rows)
        ], axis=1)
        
        # 3. Convert the combined NumPy array to a Pandas DataFrame
        df = pd.DataFrame(combined_data)
        
        # 4. Define clean column names (6 from trajectory/slack/label + 2 from obstacle)
        df.columns = [
            'state_x', 
            'state_y', 
            'action_x', 
            'action_y', 
            'slack_value',
            'label_safety_critical',
            'obstacle_x',          # Corrected column name
            'obstacle_y'           # Added column for the second obstacle dimension
        ]
        
        # 5. Save the DataFrame to a CSV file
        df.to_csv(csv_file_path, index=False)
        print(f"Total rows in final dataset: {df.shape[0]}")
        print(f"Successfully saved combined data to CSV: {csv_file_path}")
        
    except FileNotFoundError:
        print(f"Error: File not found at {npz_file_path}. Please check your path.")
    except KeyError as e:
        print(f"Error: Required key {e} not found in the .npz archive. Check the keys used during saving.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Execution ---

# You might need to adjust the input file name based on your data generation script
input_file = "data/safe_profile.npz" 
output_file = "data/safe_critical_full_dataset.csv"

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Run the conversion
npz_to_csv_multiple_arrays(input_file, output_file)