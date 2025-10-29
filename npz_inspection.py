import numpy as np

master_filename = "data/master_sim_data.npz"

try:
    # 1. Load the NPZ file
    # The 'allow_pickle=True' is often necessary for NPZ files, though less so 
    # when only saving simple JAX/NumPy arrays. It's a good practice to include.
    data = np.load(master_filename, allow_pickle=True)

    print(f"--- Inspection of: {master_filename} ---\n")

    # 2. View the Keys (Array Names)
    print("✅ Arrays contained in the file (Keys):")
    print(list(data.keys()))
    print("-" * 35)

    # 3. Inspect the Shape and Type of Key Arrays
    print("📋 Data Structure Details:")
    
    # Iterate through each saved array and print its properties
    for key in data.keys():
        array = data[key]
        
        # Check if the array is a scalar (like time_steps or dt)
        if array.shape == ():
            print(f"  - {key:<20}: Value={array.item():<8} | Type={array.dtype}")
        else:
            print(f"  - {key:<20}: Shape={str(array.shape):<15} | Type={array.dtype}")

    # 4. Access a specific array
    print("\n💡 Example: Inspecting the primary training features (X_data_flat)")
    X_flat_shape = data['X_data_flat'].shape
    print(f"  - Shape of X_data_flat: {X_flat_shape}")
    print(f"  - First 3 samples:\n{data['X_data_flat'][:3]}")
    
    # Don't forget to close the NpzFile object when done (good practice)
    data.close()

except FileNotFoundError:
    print(f"ERROR: File not found at '{master_filename}'. Please check the path.")