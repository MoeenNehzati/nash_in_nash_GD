#===============================
# Setting up JAX environment variables for performance tuning
#===============================
import os
import argparse

# CLI overrides (must be handled before importing JAX)
parser = argparse.ArgumentParser(description="Grennan equilibrium solver")
parser.add_argument("--x64", action="store_true",
                    help="Enable 64-bit precision (default: 32-bit)")
parser.add_argument("--tol-inf", type=float, default=1e-5,
                    help="Override solver infinity-norm tolerance (tol_inf)")
parser.add_argument("--tol-l2", type=float, default=None,
                    help="Override solver L2-norm tolerance (tol_l2)")

args, _unknown = parser.parse_known_args()

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

# Default to float32; allow CLI override before importing JAX
os.environ["JAX_ENABLE_X64"] = "1" if args.x64 else "0"


#===============================
# Main script to run the Grennan (2013) stent market equilibrium
#===============================
from jax import numpy as jnp
from grennen_specification import static_params, dynamic_params, prices0_i
from nash.contract import find_equilibrium, best_response_map, residual
from jax import jacrev
from flax.core import FrozenDict

# Optionally override solver tolerances (create a new FrozenDict)
static_params_local = dict(static_params)
static_params_local["tol_inf"] = args.tol_inf
if args.tol_l2 is not None:
    static_params_local["tol_l2"] = args.tol_l2
static_params_local = FrozenDict(static_params_local)

print(f"\nRunning Grennan equilibrium solver...")
print(f"Using JAX_ENABLE_X64={os.environ.get('JAX_ENABLE_X64')}")
print(f"Solver tolerances: tol_inf={static_params_local['tol_inf']}, tol_l2={static_params_local['tol_l2']}")
print("=" * 70)

p_eq, info = find_equilibrium(static_params_local, dynamic_params, prices0_i)

# Calculate best response at equilibrium
br_residual = lambda p: residual(static_params_local, dynamic_params, p)
p_eq_br = best_response_map(static_params_local, dynamic_params, p_eq)
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

# Jacobian analysis for uniqueness verification
# Using sign of Jacobian determinant (Pakes-McGuire approach for checking local uniqueness)
