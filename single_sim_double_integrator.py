import jax
import jax.numpy as jnp

from dynamaxsys.integrators import DoubleIntegrator2D
from cbfax.cbf import *

import qpax

import numpy as np

import matplotlib  
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# to make gifs for the sim
from PIL import Image
import os
import glob

def plot_trajectory(trajectory, deviation_values, obstacle_states, radius, time_steps):
    # Define a gridspec layout where the first subplot takes up more rows
    fig = plt.figure(figsize=(8, 10))
    gs = fig.add_gridspec(6, 1)  # total of 6 rows
    ax0 = fig.add_subplot(gs[0:3, 0])  # first plot takes 3 rows
    ax1 = fig.add_subplot(gs[3, 0])
    ax2 = fig.add_subplot(gs[4, 0])
    ax3 = fig.add_subplot(gs[5:, 0])

    # --- Big trajectory plot (Modified for specific coloring) ---
    
    # Define a default color and the special color
    default_color = 'blue'
    special_color = 'purple'
    
    # Placeholders for the legend
    # We plot dummy points outside the loop to ensure they appear in the legend correctly
    ax0.plot([], [], marker='.', linestyle='', color=default_color, label='Regular Steps')
    ax0.plot([], [], marker='o', linestyle='', color=special_color, label='Every 10th Step')


    # Plot dots for the trajectory, changing color every 10 time steps
    for i in range(time_steps):
        # Check if the index is a multiple of 10 (i.e., 0, 10, 20, 30, etc.)
        if i % 10 == 0:
            color = special_color
            marker = 'o'  # Use a distinct marker for the special points
            size = 6
        else:
            color = default_color
            marker = '.'
            size = 3
            
        # Plot a single point for the current time step
        ax0.plot(
            trajectory[i, 0], 
            trajectory[i, 1], 
            marker=marker, 
            linestyle='', 
            color=color, 
            markersize=size
        )

    # Iterate over all obstacle states to plot a circle for each
    for i, obs_state in enumerate(obstacle_states):
        center_x = obs_state[0].item()
        center_y = obs_state[1].item()
        
        obstacle_circle = Circle(
            (center_x, center_y), 
            radius, 
            color='red', 
            alpha=0.3
        )
        ax0.add_patch(obstacle_circle)
        
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')
    ax0.set_aspect('equal')
    ax0.set_xlim(-1, 16)
    ax0.set_ylim(-1, 16)

    ax1.plot(jnp.arange(time_steps), trajectory[:, 2])
    ax1.set_xlabel('time step')
    ax1.set_ylabel('x control')

    ax2.plot(jnp.arange(time_steps), trajectory[:, 3])
    ax2.set_xlabel('time step')
    ax2.set_ylabel('y control')

    # --- MODIFIED ---
    # Plot the deviation values instead of slack
    ax3.plot(jnp.arange(time_steps), deviation_values)
    ax3.set_xlabel('time step')
    # Update the label
    ax3.set_ylabel('Control Deviation (Norm)')

def create_frame(ax, k, trajectory_history, deviation_history, obstacle_states, radius, goal_state, time_steps):
    """Updates the main trajectory plot for a single frame (step k)."""
    
    # 0. Clear and set up the plot
    ax.clear()
    
    # Data up to step k
    current_trajectory = trajectory_history[:k+1] 
    # --- MODIFIED ---
    current_deviation = deviation_history[k]

    # --- Plot 1: Agent Path and Position ---
    
    # 1. Plot Full Trajectory History (Position only: [:, 0] and [:, 1])
    ax.plot(current_trajectory[:, 0], current_trajectory[:, 1], 'b-', alpha=0.6, linewidth=1.5)
    
    # 2. Plot Current Agent Position (Last point in history)
    current_pos = current_trajectory[-1, :2]
    ax.plot(current_pos[0], current_pos[1], 'go', markersize=7, label='Agent Position', zorder=10)
    
    # 3. Plot Goal State
    ax.plot(
        goal_state[0], 
        goal_state[1], 
        marker='*', 
        markersize=12, 
        color='red', 
        label='Goal',
        zorder=5 
    )

    # 4. Plot Obstacles
    for obs_state in obstacle_states:
        center_x, center_y = obs_state[0].item(), obs_state[1].item()
        
        obstacle_circle = Circle(
            (center_x, center_y), 
            radius, 
            color='red', 
            alpha=0.3
        )
        ax.add_patch(obstacle_circle)
        
    # --- MODIFIED ---
    # Update title to show deviation
    ax.set_title(f"Simulation Step {k+1}/{time_steps} | Deviation: {current_deviation:.2f}", fontsize=12)
    ax.set_xlim(-1, 16)
    ax.set_ylim(-1, 16)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')

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
    obstacle_states = jnp.array([[3,5, 0.4, 0], [7, 7, 0, 0.6]])
    
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


    png_fig, png_ax = plt.subplots(figsize=(6, 6))
    # Create a separate folder for the many frames
    frame_dir = "output/frames_dynamic_obs"
    print(f"Starting simulation. Saving {time_steps} image files to {frame_dir}.")
    os.makedirs(frame_dir, exist_ok=True)
    
    # simulation loop
    trajectory = []
    # --- MODIFIED ---
    # Rename slack_values to deviation_values
    deviation_values = []
    
    for k in range(time_steps):
        # --- MODIFIED ---
        # 1. Calculate nominal control
        control_nominal = LQR_policy(state, goal_state, K[k])
        
        # 2. Augment control for safety
        #    We still get 'slack' back, but we won't use it as our primary indicator
        control_safe, slack = apply_CBF(state, control_nominal, obstacle_states, radius, alpha)

        # 3. Calculate the deviation (our new indicator)
        deviation = jnp.linalg.norm(control_safe - control_nominal)

        # 4. Update trajectory history with the *safe* control
        trajectory.append(jnp.concat([state, control_safe]).squeeze())
        
        # 5. Store the deviation
        deviation_values.append(deviation)

        # 6. Apply *safe* control to dynamics
        state = state + dt*dynamics(state, control_safe, dt*k)
        # --- END MODIFIED ---

        # change obs dynamics
        p_obs = obstacle_states[:, :2]
        v_obs = obstacle_states[:, 2:]
        p_obs_new = p_obs + dt * v_obs
        obstacle_states = jnp.concatenate([p_obs_new, v_obs], axis=1)

        current_trajectory_history = jnp.array(trajectory)
        # --- MODIFIED ---
        # Pass the history of deviation values
        current_deviation_history = jnp.array(deviation_values)
        
        create_frame(
            png_ax, 
            k, 
            current_trajectory_history, 
            current_deviation_history, # <-- changed
            obstacle_states,
            radius, 
            goal_state, 
            time_steps
        )
        
        # Save frame with zero-padding for correct numerical sorting
        frame_filename = os.path.join(frame_dir, f"frame_{k:03d}.png")
        plt.savefig(frame_filename, dpi=100)


    trajectory = jnp.array(trajectory)
    
    # --- MODIFIED ---
    # Convert deviation_values to a jax array
    deviation_values = jnp.array(deviation_values).squeeze()
    print("Deviation values:")
    print(deviation_values)
    
    # Generate labels based on deviation
    labels = generate_labels(deviation_values)

    print(labels)
    
    # Save the deviation values in the .npz file
    np.savez("data/safe_profile", 
             trajectory=trajectory, 
             deviation=deviation_values, # <-- changed key
             obstacle=obstacle_states, 
             radius=radius, 
             alpha=alpha, 
             labels=labels
            )
    # --- END MODIFIED ---
    
    plt.close(png_fig)
    print(f"{time_steps} frame images saved to {frame_dir}/")
    
    # --- MODIFIED ---
    # Pass deviation_values to the plotting function
    plot_trajectory(trajectory, deviation_values, obstacle_states, radius, time_steps)
    # --- END MODIFIED ---
    
    plt.show()
    # plt.savefig("output/cbf.png")

if __name__=='__main__':
    main()