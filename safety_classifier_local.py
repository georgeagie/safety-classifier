import jax
import jax.numpy as jnp

from dynamaxsys.integrators import SingleIntegrator2D
from cbfax.cbf import *

import qpax

import numpy as np

import matplotlib
matplotlib.use("Agg") # Using 'Agg' is safer for non-GUI environments like Colab
import matplotlib.pyplot as plt

def plot_trajectory(trajectory, slack_values, obstacle_state, radius, time_steps, goal_state):
    _, axs = plt.subplots(4)
    axs[0].plot(trajectory[:,0], trajectory[:,1])
    from matplotlib.patches import Circle
    obstacle_circle = Circle((obstacle_state[0], obstacle_state[1]), radius, color='red', alpha=0.3)
    axs[0].add_patch(obstacle_circle)
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_aspect('equal') 

    axs[0].set_xlim(-1, 16)
    axs[0].set_ylim(-1, 16)

    axs[0].plot(
        goal_state[0], 
        goal_state[1], 
        marker='*', 
        markersize=12, 
        color='red', 
        label='Goal State',
        zorder=5 # Ensure the marker is on top of the line
    )

    axs[1].plot(jnp.arange(time_steps), trajectory[:,2])
    axs[1].set_xlabel('time step')
    axs[1].set_ylabel('x control')

    axs[2].plot(jnp.arange(time_steps), trajectory[:,3])
    axs[2].set_xlabel('time step')
    axs[2].set_ylabel('y control')

    # TODO: uncomment this when CBF code is finished
    axs[3].plot(jnp.arange(time_steps), slack_values)
    axs[3].set_xlabel('time step')
    axs[3].set_ylabel('slack variable')

@jax.jit
def policy(state: jax.Array, goal: jax.Array, scale: float = 1.0) -> jax.Array:
    """Simple proportional controller.
    """
    control = scale*(goal - state)
    return control

def calculate_LQR_gains(Q, R, A, B, time_steps) :
    N = time_steps
    P = []
    K = []

    for i in range(N + 1):
        P.append(jnp.zeros((2, 2)))

    for i in range(N):
        K.append(jnp.zeros((2, 2)))

    P[N] = Q

    for i in range(N - 1, -1, -1):
        eq_first = R + B.T @ P[i + 1] @ B
        eq_second = B.T @ P[i + 1] @ A
        K[i] = jnp.linalg.solve(eq_first, eq_second) 

        P[i] = Q + A.T @ P[i + 1] @ A - A.T @ P[i + 1] @ B @ K[i]
    
    return K

def LQR_policy(state: jax.Array, goal: jax.Array, K) -> jax.Array:
    state_error = state - goal
    control = -1 * K @ state_error
    return control

@jax.jit
def form_cbf_qp(state: jax.Array, nominal_control: jax.Array, obstacle_state: jax.Array, radius: float, alpha: float) -> tuple:
    p = state - obstacle_state # 2 x 1
    h_x = p.T @ p - radius**2 
    grad_h = 2.0 * p
    Lf_h = 0.0
    Lg_h = grad_h

    # QP parameters
    G = -jnp.expand_dims(Lg_h, axis=0)
    h_qp = Lf_h + alpha * h_x
    h = jnp.array([h_qp])
    Q = jnp.eye(2)
    q = -2.0 * nominal_control
    A = jnp.empty((0, 2))
    b = jnp.empty((0,))
    return Q, q, A, b, G, h

@jax.jit
def apply_CBF(state: jax.Array, control: jax.Array, obstacle_state: jax.Array, radius: float, alpha: float) -> tuple:
    Q, q, A, b, G, h = form_cbf_qp(state, control, obstacle_state, radius, alpha)
    safe_control, slack_vector, _, _, _, _ = qpax.solve_qp(Q, q, A, b, G, h, solver_tol=1e-3)
    slack = slack_vector[0]
    return safe_control, slack

def generate_labels(slack_values: jax.Array) -> jax.Array:
    threshold = 0.5
    labels = (slack_values <= threshold).astype(jnp.int32)
    return labels

def run_sim(state, goal_state, obstacle_state, run_id):
    
    # simulation loop
    augmented_trajectory_history = [] 
    slack_values = []
    active_steps = []
    
    for k in range(time_steps):
        # 1. nominal control
        nominal_control = LQR_policy(state, goal_state, K[k]) # Renamed to nominal_control
        
        # 2. augment control for safety (control is now the safe control)
        safe_control, slack = apply_CBF(state, nominal_control, obstacle_state, radius, alpha) 

        # 3. Augmentation: Calculate Context Features
        relative_goal = goal_state - state
        relative_obstacle = state - obstacle_state # Vector 'p' used in CBF
        
        # 4. Form the complete 8D feature vector
        augmented_feature_vector = jnp.concatenate([
            state, 
            safe_control, 
            relative_goal, 
            relative_obstacle
        ]).squeeze()

        # 5. Check for Active Step (using the SAFE control magnitude)
        control_magnitude = jnp.linalg.norm(safe_control)
        is_active = (control_magnitude > CONTROL_THRESHOLD)
        active_steps.append(is_active)

        # 6. Update history
        augmented_trajectory_history.append(augmented_feature_vector)
        slack_values.append(slack)

        # 7. Apply dynamics
        state = state + dt*dynamics(state, safe_control, dt*k) # Use the SAFE control


    # --- FIX: Convert the list of vectors into a 2D array (N_steps, 8) ---
    augmented_trajectory = jnp.array(augmented_trajectory_history) 
    slack_values = jnp.array(slack_values).squeeze()
    labels = generate_labels(slack_values)
    active_steps = jnp.array(active_steps).squeeze() 
    
    # The plotting trajectory is [x1, x2, u1, u2] which are the first 4 elements.
    plotting_trajectory = augmented_trajectory[:, :4] 

    # Return the augmented data, slack, labels, filter, and the plotting data
    return augmented_trajectory, slack_values, labels, active_steps, plotting_trajectory

# constants for all sims
time_steps = 50
dt = 0.1
dynamics = SingleIntegrator2D()
CONTROL_THRESHOLD = 0.05 # if control is below this, data point is not worth looking at
# safety profile
radius = 3
alpha = 2
# LQR params
Q = jnp.eye(2) 
R = jnp.eye(2)
A = jnp.eye(2)
B = jnp.eye(2) * dt
K_list = calculate_LQR_gains(Q, R, A, B, time_steps)
K = jnp.array(K_list)
# key for random state generation
# Set a PRNG key for reproducibility
key = jax.random.PRNGKey(42)

# --- CONSTANTS ---
N_TOTAL = 200 # Total number of simulation runs
N_PER_ZONE = 50 # Number of points per zone
ZONE_SIZE = 3.0 # The 3x3 area size
MAX_COORD = 15.0 # Max coordinate bound

# Define the boundaries for the 4 zones (min_x, max_x, min_y, max_y)
ZONE_BOUNDARIES = {
    # Start Zones (Top-Left: 0,0 to 3,3)
    'BL': (0.0, ZONE_SIZE, 0.0, ZONE_SIZE),             # Bottom-Left
    'TL': (0.0, ZONE_SIZE, MAX_COORD - ZONE_SIZE, MAX_COORD), # Top-Left
    'TR': (MAX_COORD - ZONE_SIZE, MAX_COORD, MAX_COORD - ZONE_SIZE, MAX_COORD), # Top-Right
    'BR': (MAX_COORD - ZONE_SIZE, MAX_COORD, 0.0, ZONE_SIZE),  # Bottom-Right
}

# The sequence for start states
START_ZONE_SEQUENCE = ['BL', 'TL', 'TR', 'BR']

# The mirrored sequence for goal states
GOAL_ZONE_SEQUENCE = ['TR', 'BR', 'BL', 'TL']

# Center 3x3 area for obstacles
OBS_MIN = (MAX_COORD / 2.0) - (ZONE_SIZE / 2.0)  # 7.5 - 1.5 = 6.0
OBS_MAX = (MAX_COORD / 2.0) + (ZONE_SIZE / 2.0)  # 7.5 + 1.5 = 9.0
OBS_BOUNDARY = (OBS_MIN, OBS_MAX, OBS_MIN, OBS_MAX) # (6.0, 9.0, 6.0, 9.0)

def generate_zoned_states(num_states):
    """
    Generates JAX arrays for start, goal, and obstacle states 
    based on the specified 4-zone structure.
    """
    global key
    
    start_list = []
    goal_list = []
    obstacle_list = []

    for zone_index in range(4):
        s_zone = START_ZONE_SEQUENCE[zone_index]
        g_zone = GOAL_ZONE_SEQUENCE[zone_index]
        
        # Get (min_x, max_x, min_y, max_y) for start and goal zones
        s_min_x, s_max_x, s_min_y, s_max_y = ZONE_BOUNDARIES[s_zone]
        g_min_x, g_max_x, g_min_y, g_max_y = ZONE_BOUNDARIES[g_zone]
        
        # Get (min_x, max_x, min_y, max_y) for obstacle center zone
        o_min_x, o_max_x, o_min_y, o_max_y = OBS_BOUNDARY

        for i in range(N_PER_ZONE):
            key, s_key, g_key, o_key, jitter_key = jax.random.split(key, 5)

            # 1. Generate Start State (x, y)
            start_x = jax.random.uniform(s_key, shape=(), minval=s_min_x, maxval=s_max_x)
            start_y = jax.random.uniform(s_key, shape=(), minval=s_min_y, maxval=s_max_y)
            start_state_base = jnp.array([start_x, start_y])

            # 2. Generate Goal State (x, y)
            goal_x = jax.random.uniform(g_key, shape=(), minval=g_min_x, maxval=g_max_x)
            goal_y = jax.random.uniform(g_key, shape=(), minval=g_min_y, maxval=g_max_y)
            goal_state_base = jnp.array([goal_x, goal_y])
            
            # 3. Generate Obstacle State (x, y)
            obs_x = jax.random.uniform(o_key, shape=(), minval=o_min_x, maxval=o_max_x)
            obs_y = jax.random.uniform(o_key, shape=(), minval=o_min_y, maxval=o_max_y)
            obstacle_state_base = jnp.array([obs_x, obs_y])

            # 4. Apply Jitter (Random digit in the thousandth place)
            jitter_digits = jax.random.randint(jitter_key, shape=(6,), minval=1, maxval=10)
            jitter_array = jitter_digits / 1000.0 # Convert to [0.001, 0.009]

            start_state = start_state_base + jitter_array[:2]
            goal_state = goal_state_base + jitter_array[2:4]
            obstacle_state = obstacle_state_base + jitter_array[4:6]
            
            # Store the generated states
            start_list.append(start_state)
            goal_list.append(goal_state)
            obstacle_list.append(obstacle_state)
    
    return start_list, goal_list, obstacle_list

def main():
    # 1. GENERATE STATES
    start_states, goal_states, obstacle_states = generate_zoned_states(200)
    
    PLOT_INDICES = {40, 80, 120, 160, 199}

    # 2. INITIALIZE MASTER LISTS FOR AGGREGATION
    all_augmented_trajectories = []  
    all_slacks = []        
    all_labels = []        
    
    # 3. RUN SIMULATIONS AND AGGREGATE DATA
    for i, (start_state, goal_state, obstacle_state) in enumerate(zip(start_states, goal_states, obstacle_states)):
        
        # Run simulation and get 5 results
        augmented_trajectory_run, slack_values_run, labels_run, active_steps_run, plotting_trajectory_run = run_sim(
            start_state, 
            goal_state, 
            obstacle_state, 
            i
        )

        if i in PLOT_INDICES:
            print(f"Plotting and saving Run {i}...")
            
            # Use the plotting trajectory which contains [x, y, u_x, u_y]
            plot_trajectory(
                trajectory=plotting_trajectory_run, 
                slack_values=slack_values_run, 
                obstacle_state=obstacle_state, 
                radius=radius, 
                time_steps=time_steps,
                goal_state=goal_state
            )
            
            # Save the figure to a file
            filename = f"output/run_{i}_trajectory.png"
            plt.savefig(filename)
            plt.close() 
            print(f"   Saved plot to: {filename}")
        
        # --- FILTERING ---
        filtered_augmented_trajectory = augmented_trajectory_run[active_steps_run]
        filtered_slacks = slack_values_run[active_steps_run]
        filtered_labels = labels_run[active_steps_run]
        
        # Only aggregate if the run contributed meaningful data
        if filtered_augmented_trajectory.shape[0] > 0:
            print(f"\n--- Simulation Run {i + 1} / {len(start_states)} ---")
            print(f"   Saved {filtered_augmented_trajectory.shape[0]} out of {time_steps} steps.")
            
            # Aggregate the full augmented data set
            all_augmented_trajectories.append(filtered_augmented_trajectory)
            all_slacks.append(filtered_slacks)
            all_labels.append(filtered_labels)
            
        else:
            print(f"\n--- Simulation Run {i + 1} / {len(start_states)} ---")
            print(f"   ⚠️ Run skipped: No active steps found (Control < {CONTROL_THRESHOLD}).")


    # --- 4. CONCATENATION (Flattening for Variable Lengths) ---
    X_data_flat = jnp.concatenate(all_augmented_trajectories, axis=0) 
    Y_labels_flat = jnp.concatenate(all_labels, axis=0)
    slacks_flat = jnp.concatenate(all_slacks, axis=0)
    
    total_samples = Y_labels_flat.shape[0]
    critical_count = jnp.sum(Y_labels_flat) 
    non_critical_count = total_samples - critical_count
    critical_percentage = (critical_count / total_samples) * 100
    
    
    print("\n--- Data Balance Summary ---")
    print(f"Total Samples Saved:          {total_samples}")
    print(f"Feature Dimension:            {X_data_flat.shape[1]}") 
    print(f"Safety-Critical (Label 1) Count: {critical_count}")
    print(f"Non-Critical (Label 0) Count:  {non_critical_count}")
    print(f"Proportion of Critical Data:   {critical_percentage:.2f}%")

    # 5. SAVE TO MASTER NPZ FILE
    master_filename = "data/master_sim_data_filtered_augmented.npz" 
    print(f"\nSaving aggregated augmented data to: {master_filename}")
    
    np.savez_compressed(
        master_filename,
        
        # PRIMARY TRAINING DATA (Flattened, Filtered)
        X_data_flat=X_data_flat,       # CONTAINS [x, u, dx_goal, dx_obs]
        Y_labels_flat=Y_labels_flat,   
        slacks_flat=slacks_flat,       

        # Constants
        time_steps_max=time_steps, 
        dt=dt
    )
    print("Master NPZ file saved successfully with augmented data.")

if __name__=='__main__':
    main()