import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
# Use the 'Agg' backend for generating images without a display (headless)
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os 

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

    # 1. Plot Obstacle(s)
    for i, obs_state in enumerate(obstacle_states):
        # Extract all state components
        center_x = obs_state[0].item() 
        center_y = obs_state[1].item()
        vel_x = obs_state[2].item()  # <-- NEW: Get obstacle velocity x
        vel_y = obs_state[3].item()  # <-- NEW: Get obstacle velocity y
        
        # Plot the obstacle's body (circle)
        obstacle_circle = Circle(
            (center_x, center_y), 
            radius, 
            color='red', 
            alpha=0.4,
            zorder=0  # Base layer
        )
        ax.add_patch(obstacle_circle)
        
        # <-- NEW: Plot the obstacle's velocity vector (arrow) -->
        ax.quiver(
            center_x, center_y, 
            vel_x, vel_y, 
            color='red',  # Match the circle color
            angles='xy', 
            scale_units='xy', 
            scale=1.0,    # Use same scale as agent for consistency
            width=0.005, 
            headwidth=3,
            headlength=5,
            zorder=1      # Draw arrow on top of the circle
        )
        # <-- End of new section -->

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
        scale=1.0, 
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
        os.path.join(output_dir, "di2d_binary_classification_image_data"), 
        image_filenames=np.array(image_filenames),
        labels=filtered_labels
    )
    
    print(f"--- Dataset Saved ---")
    print(f"Saved {len(filtered_labels)} image/label pairs.")
    print(f"File: {os.path.join(output_dir, 'di2d_binary_classification_image_data.npz')}")
    print(f"Labels:  {filtered_labels}")

def main():
    # simulation agent set up
    time_steps = 50
    dt = 0.25
    dynamics = DoubleIntegrator2D()

    # agent state and goal init
    # 4D now, add start velocity of 0,0
    state = jnp.array([0,0,0,0])
    goal_state = jnp.array([10, 14, 0, 0])

    # obstacle 
    obstacle_states = jnp.array([[3,5, 0.0, 0], [7, 7, 0, 1.0]])
    
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

    GOAL_PROXIMITY_THRESHOLD = 0.3

    if not os.path.exists("output/di2d_sim_images"):
        os.makedirs("output/di2d_sim_images")

    # simulation loop
    trajectory = []
    deviation_values = []
    image_filenames = []
    saved_indices = [] # time_steps that pass the filter
    
    for k in range(time_steps):
        # --- MODIFIED ---
        # 1. Calculate nominal control
        control_nominal = LQR_policy(state, goal_state, K[k])
        
        # 2. Augment control for safety
        #    We still get 'slack' back, but we won't use it as our primary indicator
        control_safe, slack = apply_CBF(state, control_nominal, obstacle_states, radius, alpha)

        deviation = jnp.linalg.norm(control_safe - control_nominal)
        trajectory.append(jnp.concat([state, control_safe]).squeeze())
        deviation_values.append(deviation)

        state = state + dt*dynamics(state, control_safe, dt*k)

        p_obs = obstacle_states[:, :2]
        v_obs = obstacle_states[:, 2:]
        p_obs_new = p_obs + dt * v_obs
        obstacle_states = jnp.concatenate([p_obs_new, v_obs], axis=1)

        current_trajectory = jnp.array(trajectory)
        current_deviation = jnp.array(deviation_values)

        distance_to_goal = jnp.linalg.norm(state - goal_state)
        if (distance_to_goal >= GOAL_PROXIMITY_THRESHOLD):
            image_filename = f"output/di2d_sim_images/sim_frame_{k:04d}.png"
            generate_sim_diagram(current_trajectory, obstacle_states, radius, image_filename)
            image_filenames.append(image_filename)
            saved_indices.append(k)



    deviation_values = jnp.array(deviation_values).squeeze()

    labels = generate_labels(deviation_values)

    save_dataset(image_filenames, labels, saved_indices)

    print(f"Generated {len(image_filenames)} images, of {time_steps} possible time steps.")
    print(f"Total safety-critical states: {sum(labels)}")



if __name__=='__main__':
    main()