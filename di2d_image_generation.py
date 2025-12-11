import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
# Use the 'Agg' backend for generating images without a display (headless)
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os 
import random
import math
from dynamaxsys.integrators import DoubleIntegrator2D
from cbfax.cbf import *
import qpax
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def generate_sim_diagram(trajectory, obstacle_states, radius, output_filename):
    """
    Generates and saves a clean diagram of the agent and obstacles.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    for obs_state in obstacle_states:
        # Extract all state components
        center_x = obs_state[0].item() 
        center_y = obs_state[1].item()
        vel_x = obs_state[2].item() 
        vel_y = obs_state[3].item()  
        
        # Plot the obstacle's body (circle)
        obstacle_circle = Circle(
            (center_x, center_y), 
            radius, 
            color='red', 
            alpha=0.4,
            zorder=0 
        )
        ax.add_patch(obstacle_circle)
        
        # vel vector
        ax.quiver(
            center_x, center_y, 
            vel_x, vel_y, 
            color='red',  
            angles='xy', 
            scale_units='xy', 
            scale=1.0,
            width=0.005, 
            headwidth=3,
            headlength=5,
            zorder=1     
        )

    current_state_x = trajectory[-1, 0]
    current_state_y = trajectory[-1, 1]
    current_control_vx = trajectory[-1, 2] 
    current_control_vy = trajectory[-1, 3]
    
    ax.quiver(
        current_state_x, current_state_y, 
        current_control_vx, current_control_vy, 
        color='blue', 
        angles='xy', 
        scale_units='xy', 
        scale=1.0, 
        width=0.005, 
        headwidth=3,
        headlength=5,
        zorder=2
    )
    

    ax.plot(
        current_state_x, current_state_y, 
        'o', 
        color='blue', 
        markersize=6, 
        zorder=3
    )
    
    ax.set_aspect('equal')
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off') 
    
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig) # free up memory 
    
    return output_filename

@jax.jit
def policy(state: jax.Array, goal: jax.Array, scale: float = 1.0) -> jax.Array:
    """Simple proportional controller.

    Args:
        state (jax.Array): Current state of the agent.
        goal (jax.Array): Desired goal state.
        scale (float, optional): Scaling factor for the control input. Defaults to 1.

    Returns:
        jax.Array: Control input to be applied.
    """
    control = scale*(goal - state)
    return control

def calculate_LQR_gains(Q, R, A, B, time_steps) :
    N = time_steps
    P = []
    K = []

    for i in range(N + 1):
        P.append(jnp.zeros((4, 4)))

    for i in range(N):
        K.append(jnp.zeros((2, 4)))

    P[N] = Q

    for i in range(N - 1, -1, -1):
        # calculate K_i = (R + B.T * P_{i+1} * B)^-1 * (B.T * P_{i+1} * A)
        # split equation into two parts:
        #   (R + B.T * P_{i+1} * B)^-1
        #   (B.T * P_{i+1} * A)
        eq_first = R + B.T @ P[i + 1] @ B
        eq_second = B.T @ P[i + 1] @ A
        # jnp.linalg.solve takes the inverse of the first eq and multiplies it with second
        K[i] = jnp.linalg.solve(eq_first, eq_second) 

        # calculate DARE so you have the P value for next pass
        # note: we could substitute K[i] with (R + B.T * P_{i+1} * B)^-1 * (B.T * P_{i+1} * A)
        P[i] = Q + A.T @ P[i + 1] @ A - A.T @ P[i + 1] @ B @ K[i]
    
    return K

def LQR_policy(state: jax.Array, goal: jax.Array, K) -> jax.Array:
    # TODO: implement LQR
    state_error = state - goal
    control = -1 * K @ state_error
    return control

@jax.jit 
def form_cbf_qp(state: jax.Array, nominal_control: jax.Array, obstacle_states: jax.Array, radius: float, alpha: float) -> tuple:
    """Formulate CBF-QP problem for DoubleIntegrator2D and return its parameters for qpax library.

    Args:
        state (jax.Array): DoubleIntegrator2D state.
        nominal_control (jax.Array): Nominal control input.
        obstacle_state (jax.Array): Obstacle state.
        radius (float): Safety radius.
        alpha (float): class Kappa parameter that represents aggressivness

    Returns:
        tuple: CBF-QP parameters (Q, q, A, b, G, h)
    """
    # Split agent state
    p = state[:2]
    v = state[2:]

    # Split obstacle states
    p_obs = obstacle_states[:, :2]
    v_obs = obstacle_states[:, 2:]

    # Relative quantities
    p_rel = p - p_obs       
    v_rel = v - v_obs     

    # Compute CBF terms per obstacle
    h_x = jnp.sum(p_rel**2, axis=1) - radius**2       
    h_dot = 2.0 * jnp.sum(p_rel * v_rel, axis=1)      
    Lf2_h = 2.0 * jnp.sum(v_rel**2, axis=1)          
    LgLf_h = 2.0 * p_rel                             
    rhs = Lf2_h + 2.0 * alpha * h_dot + (alpha**2) * h_x  

    Q = jnp.eye(2)
    q = -2.0 * nominal_control
    A = jnp.empty((0, 2))
    b = jnp.empty((0,))
    G = -LgLf_h  # each row is a constraint
    h = rhs      # vector of constraints

    return Q, q, A, b, G, h


@jax.jit
def apply_CBF(state: jax.Array, control: jax.Array, obstacle_state: jax.Array, radius: float, alpha: float) -> tuple:
    """Apply CBF-QP to control input for safety.

    Args:
        state (jax.Array): SingleIntegrator2D state.
        control (jax.Array): Control input to be augmented.
        obstacle_state (jax.Array): Obstacle state.
        radius (float): Safety radius.
        alpha (float): class Kappa parameter that represents aggressivness

    Returns:
        tuple: Safe control and the slack on the safety constraint
    """
    # TODO: apply CBF-QP to control
    Q, q, A, b, G, h = form_cbf_qp(state, control, obstacle_state, radius, alpha)
    safe_control, slack_vector, _, _, _, _ = qpax.solve_qp(Q, q, A, b, G, h, solver_tol=1e-3)
    slack = slack_vector[0]
    return safe_control, slack

def generate_labels(deviation_values: jax.Array) -> jax.Array:
    """
    Labels state action pairs based on the control deviation.

    - 1 (safety-critical): Deviation > threshold (CBF significantly changed nominal control).
    - 0 (nominal): Deviation <= threshold (Nominal control was already safe or needed minor adjustment).
    """
    # A small threshold to detect meaningful deviation
    threshold = 1.0 
    # Label is 1 (critical) if the deviation is GREATER than the threshold
    labels = (deviation_values > threshold).astype(jnp.int32)
    return labels

def generate_zoned_states():
    """
    Generates JAX arrays for start, goal, and (multiple) obstacle states
    with a non-overlapping placement guarantee.
    """
    global key
    
    start_list = []
    goal_list = []
    obstacle_list = [] # This will be a list of [ (N_obs, 4), ... ] arrays

    for zone_index in range(4):
        s_zone = START_ZONE_SEQUENCE[zone_index]
        g_zone = GOAL_ZONE_SEQUENCE[zone_index]
        
        s_min_x, s_max_x, s_min_y, s_max_y = ZONE_BOUNDARIES[s_zone]
        g_min_x, g_max_x, g_min_y, g_max_y = ZONE_BOUNDARIES[g_zone]
        o_min_x, o_max_x, o_min_y, o_max_y = OBS_BOUNDARY
        
        for i in range(N_PER_ZONE):
            # Get a unique key for this sample
            key, sample_key, s_key, g_key, o_key, random_key = jax.random.split(key, 6)

            start_x = jax.random.uniform(s_key, shape=(), minval=s_min_x, maxval=s_max_x)
            start_y = jax.random.uniform(s_key, shape=(), minval=s_min_y, maxval=s_max_y)
            goal_x = jax.random.uniform(g_key, shape=(), minval=g_min_x, maxval=g_max_x)
            goal_y = jax.random.uniform(g_key, shape=(), minval=g_min_y, maxval=g_max_y)
            
            # --- 2. Iteratively Generate Obstacles ---
            
            # This list will hold the *base* (x, y) positions of all
            # objects in this sample for collision checking.
            occupied_positions_base = [
                (start_x, start_y),
                (goal_x, goal_y)
            ]
            
            # This list will hold the full [x, y, vx, vy] states
            sample_obstacle_list_base = [] # A standard Python list
            
            # Decide how many obstacles for this sample
            key, o_num_key = jax.random.split(key)
            num_obstacles = jax.random.randint(
                o_num_key, 
                shape=(), 
                minval=1, 
                maxval=MAX_OBSTACLES_PER_SAMPLE + 1
            )
            
            for _ in range(num_obstacles):
                
                for attempt in range(MAX_PLACEMENT_ATTEMPTS):
                    # Generate a *single* candidate position
                    key, o_pos_key = jax.random.split(key)
                    cand_x = jax.random.uniform(o_pos_key, shape=(), minval=o_min_x, maxval=o_max_x)
                    cand_y = jax.random.uniform(o_pos_key, shape=(), minval=o_min_y, maxval=o_max_y)
                    
                    # Check for overlap with all previously placed objects
                    is_overlapping = False
                    for (ox, oy) in occupied_positions_base:
                        dist = jnp.sqrt((cand_x - ox)**2 + (cand_y - oy)**2)
                        if dist < MIN_SEPARATION:
                            is_overlapping = True
                            break # This spot is bad, try a new one
                    
                    if not is_overlapping:
                        key, o_vel_key, o_type_key = jax.random.split(key, 3)
                        
                        angle = jax.random.uniform(o_vel_key, shape=(), minval=0.0, maxval=2*jnp.pi)
                        speed = jax.random.uniform(o_vel_key, shape=(), minval=0.5, maxval=MAX_OBS_SPEED)
                        
                        is_static = jax.random.bernoulli(o_type_key, p=(1.0 - P_MOVING_OBSTACLE))
                        
                        obs_vx = jnp.where(is_static, 0.0, speed * jnp.cos(angle))
                        obs_vy = jnp.where(is_static, 0.0, speed * jnp.sin(angle))
                        
                        occupied_positions_base.append((cand_x, cand_y))
                        sample_obstacle_list_base.append(jnp.array([cand_x, cand_y, obs_vx, obs_vy]))
                        
                        break # Exit the "attempt" loop

            if sample_obstacle_list_base:
                obstacle_states_base = jnp.stack(sample_obstacle_list_base)

            num_obs = obstacle_states_base.shape[0]
            num_random_digits = 2 + 2 + (num_obs * 4)
            random_digits = jax.random.randint(
                random_key, 
                shape=(num_random_digits,), 
                minval=1, 
                maxval=10
            )
            random_array = random_digits / 1000.0 

            start_state = jnp.array([
                start_x + random_array[0], 
                start_y + random_array[1], 
                0.0, 0.0
            ])
            goal_state = jnp.array([
                goal_x + random_array[2], 
                goal_y + random_array[3], 
                0.0, 0.0
            ])
            
            if num_obs > 0:
                obs_jitter = random_array[4:].reshape((num_obs, 4))
                obstacle_states = obstacle_states_base + obs_jitter

            start_list.append(start_state)
            goal_list.append(goal_state)
            obstacle_list.append(obstacle_states)
    
    return start_list, goal_list, obstacle_list


# CONSTANTS
time_steps = 50
dt = 0.25
dynamics = DoubleIntegrator2D()

# safety profile
radius = 2
alpha = 2

# LQR params
Q = jnp.eye(4) # 4D state
R = jnp.eye(2) # 2D control
A = jnp.array([
    [1.0, 0.0, dt, 0.0],
    [0.0, 1.0, 0.0, dt],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])
B = jnp.array([
    [0.5 * dt**2, 0.0],
    [0.0, 0.5 * dt**2],
    [dt, 0.0],
    [0.0, dt]
])
K_list = calculate_LQR_gains(Q, R, A, B, time_steps)
K = jnp.array(K_list)

key = jax.random.PRNGKey(42)

GOAL_PROXIMITY_THRESHOLD = 0.3

ARENA_SIZE = 15.0
RADIUS = 0.5  # Radius for *all* objects (agent and obstacles)
PADDING = 1.5 # How far from the wall to spawn
MIN_SEPARATION = 2 * RADIUS + 0.5 # Min dist between any two objects at spawn
MAX_OBS_SPEED = 2.0 # Max speed for a moving obstacle

N_TOTAL = 240 # total number of simulation runs
N_PER_ZONE = 60 # number of points per zone
ZONE_SIZE = 3.0 # the 3x3 area size
MAX_COORD = 15.0 # max coordinate bound

ZONE_BOUNDARIES = {
    'BL': (0.0, ZONE_SIZE, 0.0, ZONE_SIZE),             # Bottom-Left
    'TL': (0.0, ZONE_SIZE, MAX_COORD - ZONE_SIZE, MAX_COORD), # Top-Left
    'TR': (MAX_COORD - ZONE_SIZE, MAX_COORD, MAX_COORD - ZONE_SIZE, MAX_COORD), # Top-Right
    'BR': (MAX_COORD - ZONE_SIZE, MAX_COORD, 0.0, ZONE_SIZE),  # Bottom-Right
}

# sequence for start states
START_ZONE_SEQUENCE = ['BL', 'TL', 'TR', 'BR']

# opposing the sequences for goal states
GOAL_ZONE_SEQUENCE = ['TR', 'BR', 'BL', 'TL']

OBS_MIN = (MAX_COORD / 2.0) - (ZONE_SIZE / 2.0)  # 7.5 - 1.5 = 6.0
OBS_MAX = (MAX_COORD / 2.0) + (ZONE_SIZE / 2.0)  # 7.5 + 1.5 = 9.0
OBS_BOUNDARY = (OBS_MIN, 3.0, OBS_MIN, 12.0) # (6.0, 3.0, 6.0, 12.0)

MAX_OBSTACLES_PER_SAMPLE = 3
MAX_OBS_SPEED = 2.0
P_MOVING_OBSTACLE = 0.5
AGENT_RADIUS = 0.5
OBSTACLE_RADIUS = 0.5
MIN_BUFFER = 0.2
MIN_SEPARATION = AGENT_RADIUS + OBSTACLE_RADIUS + MIN_BUFFER
MAX_PLACEMENT_ATTEMPTS = 50

# this function doesn't really work bc we save at each time step, not each sim
#
#  def run_sim(initial_state, goal_state, initial_obstacle_states):
#     """
#     Runs a single simulation from start to goal.
    
#     Returns:
#         tuple: (
#             trajectory: jnp.array (time_steps, 6), 
#             deviation_values: jnp.array (time_steps), 
#             final_state: jnp.array (4)
#         )
#     """
#     state = initial_state
#     obstacle_states = initial_obstacle_states

#     trajectory = []
#     deviation_values = []
    
#     for k in range(time_steps):
#         control_nominal = LQR_policy(state, goal_state, K[k])
#         control_safe, _ = apply_CBF(state, control_nominal, obstacle_states, radius, alpha)

#         deviation = jnp.linalg.norm(control_safe - control_nominal)
        
#         trajectory.append(jnp.concat([state, control_safe]).squeeze())
#         deviation_values.append(deviation)

#         state = state + dt * dynamics(state, control_safe, dt*k)

#         p_obs = obstacle_states[:, :2]
#         v_obs = obstacle_states[:, 2:]
#         p_obs_new = p_obs + dt * v_obs
#         obstacle_states = jnp.concatenate([p_obs_new, v_obs], axis=1)

#         distance_to_goal = jnp.linalg.norm(state[:2] - goal_state[:2])
#         if distance_to_goal < GOAL_PROXIMITY_THRESHOLD:
#             # If agent reaches goal, stop the simulation early
#             # Pad the rest of the trajectory with the final state/control
#             for _ in range(k + 1, time_steps):
#                 trajectory.append(jnp.concat([state, jnp.array([0.0, 0.0])]).squeeze())
#                 deviation_values.append(0.0)
#             break
            
#     return jnp.array(trajectory), jnp.array(deviation_values), state, jnp.array(obstacle_states)

def main():
    
    BASE_OUTPUT_DIR = "output/di2d_sim_images_hierarchical"
    CRITICAL_DIR = os.path.join(BASE_OUTPUT_DIR, "critical_unsafe")
    NOMINAL_DIR = os.path.join(BASE_OUTPUT_DIR, "nominal_safe")
    DATASET_DIR = "output"

    for d in [BASE_OUTPUT_DIR, CRITICAL_DIR, NOMINAL_DIR, DATASET_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    start_states, goal_states, all_obstacle_states = generate_zoned_states()
    
    N_SIMULATIONS = len(start_states)
    
    print(f"Starting {N_SIMULATIONS} simulation runs.")
    
    # Trackers for the final dataset
    image_filenames = []      # The paths for ONLY the saved images
    saved_labels = []         # The label (0/1) for ONLY the saved images
    
    total_images_saved = 0
    
    # Loop over all generated initial conditions
    for sim_run_count, (start_state, goal_state, initial_obstacle_states) in enumerate(zip(start_states, goal_states, all_obstacle_states), 1):
        state = start_state
        obstacle_states_k = initial_obstacle_states

        trajectory = []
        deviation_values = []
        
        for k in range(time_steps):
            control_nominal = LQR_policy(state, goal_state, K[k])
            control_safe, _ = apply_CBF(state, control_nominal, obstacle_states_k, radius, alpha)

            current_deviation = jnp.linalg.norm(control_safe - control_nominal)
            deviation_values.append(current_deviation)
            
            # Determine folder based on safety-criticality
            if current_deviation > 1.0:
                output_folder = CRITICAL_DIR
                label = 1 
            else:
                output_folder = NOMINAL_DIR
                label = 0 
            
            current_trajectory_slice = jnp.array(trajectory + [jnp.concat([state, control_safe]).squeeze()])

            image_filename = os.path.join(
                output_folder, 
                f"sim_{sim_run_count:04d}_frame_{k:03d}.png"
            )
            
            # Generate and save the image
            generate_sim_diagram(
                current_trajectory_slice, 
                obstacle_states_k, # The obstacle state at time k
                radius, 
                image_filename
            )
            
            image_filenames.append(image_filename)
            saved_labels.append(label)
            total_images_saved += 1

            state = state + dt * dynamics(state, control_safe, dt*k)
            p_obs = obstacle_states_k[:, :2]
            v_obs = obstacle_states_k[:, 2:]
            p_obs_new = p_obs + dt * v_obs
            obstacle_states_k = jnp.concatenate([p_obs_new, v_obs], axis=1)

            trajectory.append(jnp.concat([state, control_safe]).squeeze())

            distance_to_goal = jnp.linalg.norm(state[:2] - goal_state[:2])
            if distance_to_goal < GOAL_PROXIMITY_THRESHOLD and k < time_steps - 1:
                # If agent reaches goal, stop the simulation loop
                break

        # Simple progress update
        if sim_run_count % 20 == 0:
            print(f"Simulations run: {sim_run_count}/{N_SIMULATIONS}. Images saved: {total_images_saved}.")

    
    print("\n--- Simulation Complete ---")
    print(f"Total simulations run: {sim_run_count}")
    print(f"Total images saved: {total_images_saved}")
    print(f"Images in '{os.path.basename(CRITICAL_DIR)}': {sum(saved_labels)}")
    print(f"Images in '{os.path.basename(NOMINAL_DIR)}': {len(saved_labels) - sum(saved_labels)}")
    print(f"Dataset manifest saved to {os.path.join(DATASET_DIR, 'di2d_hierarchical_image_data.npz')}")


if __name__=='__main__':
    main()