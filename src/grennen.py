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



from nash.initializer import init_simple_params, init_grennan_params
from nash.contract import find_equilibrium, best_response_map, calculate_nash_product
static_params, dynamic_params = init_grennan_params()

import jax
p_eq, info = find_equilibrium(static_params, dynamic_params)
p_eqq = best_response_map(static_params, {**dynamic_params, "prices_i": p_eq})
print(p_eq, p_eqq)
print(p_eq - p_eqq)

# i=0
# calculate_nash_product(params, prices0, i)
# print(jacobian(lambda p:calculate_nash_product(params,p,i))(prices0))