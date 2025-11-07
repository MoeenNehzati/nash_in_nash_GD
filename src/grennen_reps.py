#===============================
# Setting up JAX environment variables for performance tuning
#===============================
import os
import argparse

# CLI overrides (must be handled before importing JAX)
parser = argparse.ArgumentParser(description="Grennan replications runner")
parser.add_argument("--x64", action="store_true",
                    help="Enable 64-bit precision (default: 32-bit)")
parser.add_argument("--tol-inf", type=float, default=1e-5,
                    help="Override solver infinity-norm tolerance (tol_inf)")
parser.add_argument("--tol-l2", type=float, default=None,
                    help="Override solver L2-norm tolerance (tol_l2)")
parser.add_argument("--cmp-tol", type=float, default=5e-3,
                    help="Override comparison tolerance across replications")
parser.add_argument("--reps", type=int, default=100,
                    help="Override number of replications")
parser.add_argument("--jobs", type=int, default=8,
                    help="Override number of parallel workers")

args, _unknown = parser.parse_known_args()

os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=true "
    "intra_op_parallelism_threads=1 "
    "inter_op_parallelism_threads=1 "
    "--xla_cpu_use_thunk_runtime=false "
)

# Default to float32; allow CLI override before importing JAX
os.environ["JAX_ENABLE_X64"] = "1" if args.x64 else "0"

# Suppress specific JAX dtype truncation warning


#===============================
# Replication test: Check equilibrium uniqueness across random initial prices
#===============================
import warnings
from jax import numpy as jnp
from jax import random
import numpy as np
import time
from joblib import Parallel, delayed
import json
from typing import Dict, Tuple, Optional
import glob
from nash.contract import find_equilibrium


def load_replication_results(filepath: Optional[str] = None) -> Tuple[Dict, np.ndarray]:
    """Load replication results from saved output files.
    
    Parameters
    ----------
    filepath : str, optional
        Path to the .npz file to load. If None, loads the most recent file
        from the outputs/ directory.
    
    Returns
    -------
    config : dict
        Configuration metadata (from JSON file)
    data : dict
        Dictionary containing all numpy arrays:
        - equilibria: (n_replications, n_stents) array of equilibrium prices
        - mean_eq: (n_stents,) mean equilibrium across replications
        - losses: (n_replications,) final loss values
        - iterations: (n_replications,) iteration counts
        - lower_bounds, upper_bounds: price bounds
        - sigma_t, sigma_stent, theta_p, theta_i, bp_i, phi_l: model parameters
        - hastype_i_t: type indicator matrix
    
    Examples
    --------
    >>> config, data = load_replication_results()
    >>> print(f"Loaded {config['n_replications']} replications")
    >>> print(f"Mean equilibrium: {data['mean_eq']}")
    """
    if filepath is None:
        # Find most recent file in outputs/ directory
        npz_files = glob.glob("outputs/grennen_reps_*.npz")
        if not npz_files:
            raise FileNotFoundError("No replication output files found in outputs/ directory")
        filepath = max(npz_files, key=os.path.getctime)
        print(f"Loading most recent results: {filepath}")
    
    # Load the npz file
    data = np.load(filepath)
    
    # Load corresponding JSON config
    json_path = filepath.replace('.npz', '.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            config = json.load(f)
    else:
        print(f"Warning: Config file {json_path} not found")
        config = {}
    
    return config, dict(data)


# Helper function to run find_equilibrium with suppressed output
def find_equilibrium_quiet(static_params, dynamic_params, prices_init):
    """Run find_equilibrium with stdout and warnings suppressed."""
    warnings.filterwarnings("ignore", message=".*Explicitly requested dtype.*will be truncated.*")
    result = find_equilibrium(static_params, dynamic_params, prices_init, quiet=True)
    return result


def run_replications():
    """Main function to run replication tests. Only called when script is run directly."""
    # Import here to avoid executing at module import time
    from grennen_specification import static_params, dynamic_params, prices0_i
    from flax.core import FrozenDict
    
    # Configuration (allow CLI overrides)
    n_replications = args.reps
    seed = 42
    tolerance = args.cmp_tol
    n_jobs = args.jobs

    # Get price bounds
    lower_bounds = dynamic_params["lower_bounds_i"]
    upper_bounds = dynamic_params["upper_bounds_i"]
    n_stents = static_params["n_stents"]

    # Optionally override solver tolerances (create a new FrozenDict)
    static_params_local = dict(static_params)
    static_params_local["tol_inf"] = args.tol_inf
    static_params_local["tol_l2"] = args.tol_l2
    static_params_local = FrozenDict(static_params_local)

    # Storage for results
    equilibria = []
    losses = []
    iterations = []

    # Generate all random initial prices at once
    key = random.PRNGKey(seed)
    # Sample uniform [0, 1] with shape (n_stents, n_replications)
    uniform_samples = random.uniform(key, shape=(n_stents, n_replications))
    # Scale to [lower_bound, upper_bound] for each product
    # Broadcasting: lower_bounds[:, None] is (n_stents, 1), uniform_samples is (n_stents, n_replications)
    all_initial_prices = lower_bounds[:, None] + (upper_bounds[:, None] - lower_bounds[:, None]) * uniform_samples
    # Transpose so shape is (n_replications, n_stents)
    all_initial_prices = all_initial_prices.T

    print(f"\nRunning {n_replications} replications with random initial prices...")
    print(f"Using JAX_ENABLE_X64={os.environ.get('JAX_ENABLE_X64')}")
    print(f"Solver tolerances: tol_inf={static_params_local['tol_inf']}, tol_l2={static_params_local['tol_l2']}")
    print(f"Comparison tolerance: {tolerance}")
    print(f"Lower price bounds: {lower_bounds}")
    print(f"Upper price bounds: {upper_bounds}")
    print(f"Parallel workers: {n_jobs}")
    print("=" * 70)

    # Start timing
    start_time = time.time()

    # Run replications in parallel with tqdm progress bar
    # Threading backend: all threads share compiled JAX code (compile once, not per worker)
    # JAX releases GIL during computation, so threading works well for JAX despite Python's GIL
    print("Starting parallel execution...")
    results = Parallel(
        n_jobs=n_jobs, 
        backend='threading',
        verbose=10  # Show progress updates from joblib
    )(
        delayed(find_equilibrium_quiet)(static_params_local, dynamic_params, prices_init)
        for prices_init in all_initial_prices
    )

    # Clear any stray output and print on fresh line
    print("\r" + " " * 100 + "\r", end="", flush=True)  # Clear current line
    print(f"\nParallel execution completed. Processing {len(results)} results...", flush=True)

    # Unpack results
    for p_eq, info in results:
        equilibria.append(p_eq)
        losses.append(info["loss"])
        iterations.append(info["n_iter"])

    print(f"Unpacked {len(equilibria)} equilibria", flush=True)

    # End timing
    total_time = time.time() - start_time

    print("\n" + "=" * 70, flush=True)
    print("SUMMARY OF RESULTS", flush=True)
    print("=" * 70, flush=True)

    # Convert to array for easier analysis
    equilibria = jnp.array(equilibria)  # Shape: (n_replications, n_stents)

    # Check if all equilibria are the same
    print(f"\nTotal time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Average time per replication: {total_time/len(equilibria):.2f} seconds")
    print(f"\nConvergence statistics:")
    print(f"  Mean iterations: {np.mean(iterations):.1f} ± {np.std(iterations):.1f}")
    print(f"  Mean final loss: {np.mean(losses):.3e}")
    print(f"  Max final loss: {np.max(losses):.3e}")

    # Compare all equilibria to the mean equilibrium across replications
    mean_eq = jnp.mean(equilibria, axis=0)
    max_diff = jnp.max(jnp.abs(equilibria - mean_eq), axis=1)

    print(f"\nEquilibrium comparison (relative to mean across replications):")
    for i in range(len(equilibria)):
        status = "✓ SAME" if max_diff[i] < tolerance else "✗ DIFFERENT"
        print(f"  Replication {i+1}: max diff = {max_diff[i]:.6f}  {status}")

    # Overall assessment
    all_same = jnp.all(max_diff < tolerance)
    print("\n" + "=" * 70)
    if all_same:
        print("✓ SUCCESS: All replications are within tolerance of the MEAN equilibrium!")
        print(f"  (tolerance {tolerance})")
    else:
        print("✗ WARNING: Different equilibria found!")
        print(f"  Max difference across replications: {max_diff.max():.6f}")
        
    print("=" * 70)

    # Print the mean equilibrium
    print(f"\nMean equilibrium prices (across replications):")
    print(mean_eq)

    # Compute statistics across all equilibria
    print(f"\nEquilibrium price statistics across replications:")
    print(f"  Mean: {equilibria.mean(axis=0)}")
    print(f"  Std:  {equilibria.std(axis=0)}")
    print(f"  Min:  {equilibria.min(axis=0)}")
    print(f"  Max:  {equilibria.max(axis=0)}")

    # ===============================
    # Persist results for future analysis
    # ===============================
    os.makedirs("outputs", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    npz_path = os.path.join("outputs", f"grennen_reps_{timestamp}.npz")
    json_path = os.path.join("outputs", f"grennen_reps_{timestamp}.json")

    # Save numeric artifacts (arrays) in compressed NPZ
    np.savez_compressed(
        npz_path,
        equilibria=np.asarray(equilibria),
        mean_eq=np.asarray(mean_eq),
        losses=np.asarray(losses),
        iterations=np.asarray(iterations),
        lower_bounds=np.asarray(lower_bounds),
        upper_bounds=np.asarray(upper_bounds),
        sigma_t=np.asarray(dynamic_params["σ_t"]),
        sigma_stent=np.asarray(dynamic_params["σ_stent"]),
        theta_p=np.asarray(dynamic_params["θ_p"]),
        theta_i=np.asarray(dynamic_params["θ_i"]),
        bp_i=np.asarray(dynamic_params["bp_i"]),
        phi_l=np.asarray(dynamic_params["φ_l"]),
        hastype_i_t=np.asarray(dynamic_params["hastype_i_t"]),
    )

    # Save configuration/specification metadata in JSON
    config = {
        "n_replications": n_replications,
        "seed": seed,
        "tolerance": tolerance,
        "n_jobs": n_jobs,
        "n_stents": n_stents,
        "optimizer_params": {k: v for k, v in static_params_local["optimizer_params"].items()},
        "max_iter": static_params_local["max_iter"],
        "tol_inf": static_params_local["tol_inf"],
        "tol_l2": static_params_local["tol_l2"],
        "refresh_rate": static_params_local["refresh_rate"],
        "jax_enable_x64": os.environ.get("JAX_ENABLE_X64"),
        "paths": {"npz": npz_path},
    }

    with open(json_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved arrays to: {npz_path}")
    print(f"Saved config to: {json_path}")


if __name__ == "__main__":
    run_replications()
