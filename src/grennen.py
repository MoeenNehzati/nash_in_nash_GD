#===============================
# Setting up JAX environment variables for performance tuning
#===============================`
import os
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=true "
    "intra_op_parallelism_threads=12 "
    "inter_op_parallelism_threads=3 "
    "--xla_cpu_use_thunk_runtime=false "
)
# os.environ["XLA_FLAGS"] += (
#     "--xla_cpu_enable_fast_math=true "
#     "--xla_cpu_fast_math_honor_infs=false "
#     "--xla_cpu_fast_math_honor_nans=false "
# )
os.environ["JAX_ENABLE_X64"] = "0"  # float32 mode for speed


#===============================
# Main script to run the Grennan (2013) stent market equilibrium
#===============================
from jax import numpy as jnp
from grennen_specification import static_params, dynamic_params, prices0_i
from nash.contract import find_equilibrium, best_response_map, residual
from jax import jacrev

p_eq, info = find_equilibrium(static_params, dynamic_params, prices0_i)

# Calculate best response at equilibrium
br_residual = lambda p: residual(static_params, dynamic_params, p)
p_eq_br = best_response_map(static_params, dynamic_params, p_eq)
J = jacrev(br_residual)(p_eq)
detJ = jnp.linalg.det(J)
sign, logabs = jnp.linalg.slogdet(J)

print("Determinant of Jacobian of r(p) = br(p)-p at p_eq:", detJ)
print("slogdet (sign, logabs):", sign, logabs)
print("If p is the equilibrium and br(p) the best response")
print("[p, br(p)] is")
print(jnp.vstack((p_eq, p_eq_br)).T)
print("Difference between equilibrium prices and best responses at equilibrium:")
print((p_eq - p_eq_br).T)

# i=0
# calculate_nash_product(params, prices0, i)
# print(jacobian(lambda p:calculate_nash_product(params,p,i))(prices0))

#get the sign of jacobian
#pagas proof