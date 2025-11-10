import jax
import jax.numpy as jnp

from dynamaxsys.integrators import SingleIntegrator2D
from cbfax.cbf import *

import qpax

import numpy as np

import matplotlib
matplotlib.use("Qt5Agg")  
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def plot_trajectory(trajectory, slack_values, obstacle_states, radius, time_steps):
    # Define a gridspec layout where the first subplot takes up more rows
    fig = plt.figure(figsize=(8, 10))
    gs = fig.add_gridspec(6, 1)  # total of 6 rows
    ax0 = fig.add_subplot(gs[0:3, 0])  # first plot takes 3 rows
    ax1 = fig.add_subplot(gs[3, 0])
    ax2 = fig.add_subplot(gs[4, 0])
    ax3 = fig.add_subplot(gs[5:, 0])

    # --- Big trajectory plot (Modified for multiple obstacles) ---
    ax0.plot(trajectory[:, 0], trajectory[:, 1], label='Trajectory')
    
    # Iterate over all obstacle states to plot a circle for each
    for i, obs_state in enumerate(obstacle_states):
        # CRITICAL FIX: Use .item() to convert JAX array elements to Python scalars 
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

    ax3.plot(jnp.arange(time_steps), slack_values)
    ax3.set_xlabel('time step')
    ax3.set_ylabel('slack variable')

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
        P.append(jnp.zeros((2, 2)))

    for i in range(N):
        K.append(jnp.zeros((2, 2)))

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
    """Formulate CBF-QP problem for SingleIntegrator2D and return its parameters for qpax library.

    Args:
        state (jax.Array): SingleIntegrator2D state.
        nominal_control (jax.Array): Nominal control input.
        obstacle_state (jax.Array): Obstacle state.
        radius (float): Safety radius.
        alpha (float): class Kappa parameter that represents aggressivness

    Returns:
        tuple: CBF-QP parameters (Q, q, A, b, G, h)
    """
    # TODO: implement CBF-QP
    p = state - obstacle_states 
    h_x = jnp.sum(p * p, axis=1) - radius**2  

    # basically, we want to just focus on the closest obstacle to prevent the QP from not finding a control
    critical_index = jnp.argmin(h_x) 

    # now we work only wth the critical_index or the closest obstacle
    h_critical = h_x[critical_index]
    Lg_h_critical = 2.0 * p[critical_index] 
    Lf_h_critical = 0.0 # Lf_h is always 0 for SingleIntegrator2D

    Q = jnp.eye(2)
    q = -2.0 * nominal_control
    A = jnp.empty((0, 2))
    b = jnp.empty((0,))
    G = -jnp.expand_dims(Lg_h_critical, axis=0) # Must be (1, 2)
    h = jnp.expand_dims(Lf_h_critical + alpha * h_critical, axis=0) # Must be (1,)

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

def generate_labels(slack_values: jax.Array) -> jax.Array:
    """
    Labels state action pairs based on the slack variable, assuming LOW SLACK indicates DANGER/CRITICALITY.

    - 1 (safety-critical): Slack <= threshold (Control needs deviation because nominal path is dangerous).
    - 0 (nominal): Slack > threshold (High slack indicates the nominal control is far from the danger boundary).
    """
    # Compare slack values to the threshold
    # Now, slack_values <= threshold (low slack) results in True, which becomes 1 (safety-critical).
    threshold = 1.0
    labels = (slack_values <= threshold).astype(jnp.int32)
    return labels


def main():
    # simulation agent set up
    time_steps = 50
    dt = 0.1
    dynamics = SingleIntegrator2D()

    # agent state and goal init
    state = jnp.zeros(dynamics.state_dim)
    goal_state = jnp.array([10, 14])

    # obstacle 
    obstacle_states = jnp.array([[3,5], [5, 10]])
    
    # safety profile
    radius = 2
    alpha = 2

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
    for k in range(time_steps):
        # nominal control
        # control = policy(state, goal_state)
        control = LQR_policy(state, goal_state, K[k])
        # TODO: uncomment this when CBF code is finished
        # # augment control for safety
        control, slack = apply_CBF(state, control, obstacle_states, radius, alpha)


        # upate trajectory history
        trajectory.append(jnp.concat([state, control]).squeeze())
        slack_values.append(slack)

        # apply dynamics
        state = state + dt*dynamics(state, control, dt*k)


    trajectory = jnp.array(trajectory)
    # TODO: uncomment this when CBF code is finished
    slack_values = jnp.array(slack_values).squeeze()
    labels = generate_labels(slack_values)
    np.savez("data/safe_profile", trajectory=trajectory, slack=slack_values, obstacle=obstacle_states, radius=radius, alpha=alpha, labels=labels)

    plot_trajectory(trajectory, slack_values, obstacle_states, radius, time_steps)
    plt.show()
    # plt.savefig("output/cbf.png")

if __name__=='__main__':
    main()