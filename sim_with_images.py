import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
# Use the 'Agg' backend for generating images without a display (headless)
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os 

from dynamaxsys.integrators import SingleIntegrator2D
from cbfax.cbf import *
import qpax

# --- (All necessary helper functions: generate_sim_diagram, policy, calculate_LQR_gains, 
# LQR_policy, form_cbf_qp, apply_CBF, generate_labels should be defined here) ---

def generate_sim_diagram(trajectory, obstacle_states, radius, output_filename):
    """
    Generates and saves a clean diagram of the agent and obstacles.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # 1. Plot Obstacle(s)
    for i, obs_state in enumerate(obstacle_states):
        # Correct indexing and conversion
        center_x = obs_state[0].item() 
        center_y = obs_state[1].item()
        
        obstacle_circle = Circle(
            (center_x, center_y), 
            radius, 
            color='red', 
            alpha=0.4,
            zorder=0
        )
        ax.add_patch(obstacle_circle)

    # 2. Plot Agent Marker (Position, Direction, Magnitude)
    current_state_x = trajectory[-1, 0]
    current_state_y = trajectory[-1, 1]
    current_control_vx = trajectory[-1, 2] 
    current_control_vy = trajectory[-1, 3]
    
    # Plot the agent as a directional arrow (quiver)
    ax.quiver(
        current_state_x, current_state_y, 
        current_control_vx, current_control_vy, 
        color='blue', 
        angles='xy', 
        scale_units='xy', 
        scale=3.0, 
        width=0.005, 
        headwidth=3,
        headlength=5,
        zorder=2
    )
    
    # Plot a simple position dot
    ax.plot(
        current_state_x, current_state_y, 
        'o', 
        color='blue', 
        markersize=6, 
        zorder=3
    )
    
    # 3. Set Plotting Parameters
    ax.set_aspect('equal')
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off') 
    
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig) 
    
    return output_filename

@jax.jit
def policy(state: jax.Array, goal: jax.Array, scale: float = 1.0) -> jax.Array:
    """Simple proportional controller."""
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
        # calculate K_i = (R + B.T * P_{i+1} * B)^-1 * (B.T * P_{i+1} * A)
        eq_first = R + B.T @ P[i + 1] @ B
        eq_second = B.T @ P[i + 1] @ A
        K[i] = jnp.linalg.solve(eq_first, eq_second) 

        # calculate DARE so you have the P value for next pass
        P[i] = Q + A.T @ P[i + 1] @ A - A.T @ P[i + 1] @ B @ K[i]
    
    return K

def LQR_policy(state: jax.Array, goal: jax.Array, K) -> jax.Array:
    state_error = state - goal
    control = -1 * K @ state_error
    return control

@jax.jit
def form_cbf_qp(state: jax.Array, nominal_control: jax.Array, obstacle_states: jax.Array, radius: float, alpha: float) -> tuple:
    p = state[:2]

    # Obstacles: only positions
    p_obs = obstacle_states[:, :2]

    # Relative position
    p_rel = p - p_obs              
    h_x = jnp.sum(p_rel**2, axis=1) - radius**2   
    Lg_h = 2.0 * p_rel              
    rhs = alpha * h_x       

    # Set up QP
    Q = jnp.eye(2)
    q = -2.0 * nominal_control
    A = jnp.empty((0, 2))
    b = jnp.empty((0,))
    G = -Lg_h                       
    h = rhs

    return Q, q, A, b, G, h

@jax.jit
def apply_CBF(state: jax.Array, control: jax.Array, obstacle_state: jax.Array, radius: float, alpha: float) -> tuple:
    Q, q, A, b, G, h = form_cbf_qp(state, control, obstacle_state, radius, alpha)
    safe_control, slack_vector, _, _, _, _ = qpax.solve_qp(Q, q, A, b, G, h, solver_tol=1e-3)
    slack = slack_vector[0]
    return safe_control, slack

def generate_labels(slack_values: jax.Array) -> jax.Array:
    threshold = 1.0
    labels = (slack_values <= threshold).astype(jnp.int32)
    return labels

def save_dataset(image_filenames, full_labels, saved_indices, output_dir="data"):
    """
    Saves the list of image filenames and the corresponding, filtered labels.
    
    Args:
        image_filenames (list[str]): List of file paths for the saved images.
        full_labels (jax.Array): The labels for the FULL simulation (length=time_steps).
        saved_indices (list[int]): The time step indices where an image was actually saved.
        output_dir (str): Directory to save the final .npz file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 1. Filter the labels to match ONLY the saved images
    # Use jnp.take to select only the labels corresponding to the saved indices.
    filtered_labels = jnp.take(full_labels, jnp.array(saved_indices))
    
    # 2. Save the dataset
    # Storing both the image filenames (as a NumPy array of strings) and their labels.
    np.savez(
        os.path.join(output_dir, "binary_classification_image_data"), 
        image_filenames=np.array(image_filenames),
        labels=filtered_labels
    )
    
    print(f"--- Dataset Saved ---")
    print(f"Saved {len(filtered_labels)} image/label pairs.")
    print(f"File: {os.path.join(output_dir, 'binary_classification_image_data.npz')}")
    print(f"Labels:  {filtered_labels}")

def main():
    # simulation agent set up
    time_steps = 50
    dt = 0.1
    dynamics = SingleIntegrator2D()

    # agent state and goal init
    state = jnp.zeros(dynamics.state_dim)
    goal_state = jnp.array([10, 14])

    # OBSTACLE 
    obstacle_states = jnp.array([[10, 8]])
    
    # SAFETY PROFILE
    radius = 2
    alpha = 2
    
    # FILTER THRESHOLD
    GOAL_PROXIMITY_THRESHOLD = 0.3 

    # LQR params
    Q = jnp.eye(2) 
    R = jnp.eye(2)
    A = jnp.eye(2)
    B = jnp.eye(2) * dt
    K_list = calculate_LQR_gains(Q, R, A, B, time_steps)
    K = jnp.array(K_list)

    # simulation loop
    trajectory = []
    slack_values = []
    image_filenames = []
    saved_indices = [] # time_steps that pass the filter
    
    if not os.path.exists("output/sim_images"):
        os.makedirs("output/sim_images")

    for k in range(time_steps):
        # nominal
        control = LQR_policy(state, goal_state, K[k])
        # control is the safe control applied after CBF
        control, slack = apply_CBF(state, control, obstacle_states, radius, alpha) 

        trajectory.append(jnp.concat([state, control]).squeeze()) 
        slack_values.append(slack)
        current_trajectory = jnp.array(trajectory)
        
        # only significant points are saved (get rid of ones very close to goal)
        distance_to_goal = jnp.linalg.norm(state - goal_state)
        if (distance_to_goal >= GOAL_PROXIMITY_THRESHOLD):
            image_filename = f"output/sim_images/sim_frame_{k:04d}.png"
            generate_sim_diagram(current_trajectory, obstacle_states, radius, image_filename)
            image_filenames.append(image_filename)
            saved_indices.append(k)

        state = state + dt*dynamics(state, control, dt*k)


    # final_trajectory = jnp.array(trajectory)
    slack_values = jnp.array(slack_values).squeeze()

    
    # Generate and save labels for the FULL trajectory
    labels = generate_labels(slack_values)

    save_dataset(image_filenames, labels, saved_indices)
    
    print(f"Generated {len(image_filenames)} images, of {time_steps} possible time steps.")

if __name__=='__main__':
    main()