import jax
import jax.numpy as jnp

from dynamaxsys.integrators import SingleIntegrator2D
from cbfax.cbf import *

import qpax

import numpy as np

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

def plot_trajectory(trajectory, slack_values, obstacle_state, radius, time_steps):
    _, axs = plt.subplots(4)
    axs[0].plot(trajectory[:,0], trajectory[:,1])
    from matplotlib.patches import Circle
    obstacle_circle = Circle((obstacle_state[0], obstacle_state[1]), radius, color='red', alpha=0.3)
    axs[0].add_patch(obstacle_circle)
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_aspect('equal') 

    axs[1].plot(jnp.arange(time_steps), trajectory[:,2])
    axs[1].set_xlabel('time step')
    axs[1].set_ylabel('x control')

    axs[2].plot(jnp.arange(time_steps), trajectory[:,3])
    axs[2].set_xlabel('time step')
    axs[2].set_ylabel('y control')

    # TODO: uncomment this when CBF code is finished
    # axs[3].plot(jnp.arange(time_steps), slack_values)
    # axs[3].set_xlabel('time step')
    # axs[3].set_ylabel('slack variable')

    plt.savefig("output/trajectory.png")

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


# state is basically is just x, y
# same with goal, but it's where we wanna go
def LQR_policy(state: jax.Array, goal: jax.Array) -> jax.Array:
    # TODO: implement LQR
    
    control = -K @ (goal - state)
    return control

@jax.jit
def form_cbf_qp(state: jax.Array, nominal_control: jax.Array, obstacle_state: jax.Array, radius: float, alpha: float) -> tuple:
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
    return safe_control, slack


def main():
    # simulation agent set up
    time_steps = 50
    dt = 0.1
    dynamics = SingleIntegrator2D()

    # agent state and goal init
    state = jnp.zeros(dynamics.state_dim)
    goal_state = jnp.array([10, 10])
    print(state)
    print(goal_state)

    # obstacle 
    obstacle_state = jnp.array([5.5,5])
    
    # safety profile
    radius = 3
    alpha = 5

    # simulation loop
    trajectory = []
    slack_values = []
    for k in range(time_steps):
        # nominal control
        control = policy(state, goal_state)
        # TODO: uncomment this when CBF code is finished
        # # augment control for safety
        # control, slack = apply_CBF(state, control, obstacle_state, radius, alpha)

        # upate trajectory history
        trajectory.append(jnp.concat([state, control]).squeeze())
        # slack_values.append(slack)

        # apply dynamics
        # state = state + dt* (state, control, dt*k)
        


    trajectory = jnp.array(trajectory)
    # TODO: uncomment this when CBF code is finished
    # slack_values = jnp.array(slack_values).squeeze()
    # np.savez("data/safe_profile", trajectory=trajectory, slack=slack_values, obstacle=obstacle_state, radius=radius, alpha=alpha)

    slack_values = None # change this later
    plot_trajectory(trajectory, slack_values, obstacle_state, radius, time_steps)

if __name__=='__main__':
    main()
