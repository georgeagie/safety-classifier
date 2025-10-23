import numpy as np
import jax.numpy as jnp

# looks at the npz file yielded from our data from sim
data = np.load('data/safe_profile.npz')

# Print the names of the arrays stored inside
print("Arrays contained in the file:", data.files) 


# Convert the trajectory data to a JAX array
# The trajectory contains: [state_x, state_y, control_x, control_y]
jnp_trajectory = jnp.array(data['trajectory'])

# Access and convert the labels to a JAX array
jnp_labels = jnp.array(data['labels'])

# Access and convert the slack values to a JAX array
jnp_slack = jnp.array(data['slack'])

# Analyze the labels using jax.numpy functions
# jnp.sum() works on the boolean/integer array to count '1's
safety_critical_count = jnp.sum(jnp_labels) 
nominal_count = len(jnp_labels) - safety_critical_count

print("Total time steps:", len(jnp_labels))
print(f"Safety-Critical steps (Label 1): {safety_critical_count}")
print(f"Nominal steps (Label 0): {nominal_count}")

data.close()