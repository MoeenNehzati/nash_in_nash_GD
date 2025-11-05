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
from grennen_specification import static_params, dynamic_params
from nash.contract import find_equilibrium, best_response_map
p_eq, info = find_equilibrium(static_params, dynamic_params)

#update price
dynamic_params["prices_i"] = p_eq
best_response_to_peq = best_response_map(static_params, dynamic_params)
print("Last two prices are")
print(jnp.vstack((p_eq, best_response_to_peq)))
print("Difference between equilibrium prices and best responses at equilibrium:")
print(p_eq - best_response_to_peq)

# i=0
# calculate_nash_product(params, prices0, i)
# print(jacobian(lambda p:calculate_nash_product(params,p,i))(prices0))