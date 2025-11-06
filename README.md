# Fornow - Nash Bargaining with Nested Logit Demand

A Python package for simulating Nash bargaining equilibria in nested logit demand systems, following the approach of Grennan (2013) for medical device markets. This implementation uses JAX for automatic differentiation and JIT compilation, with gradient-based optimization (Adam) to find equilibrium prices.

## Overview

This package implements a Nash-in-Nash bargaining model for a medical device market (coronary stents) with:
- **Nested logit demand structure** with 4 hierarchical levels
- **Nash bargaining** between suppliers and hospitals
- **Gradient-based equilibrium solver** using Adam optimizer with bound projection
- **Parallel replication testing** for equilibrium uniqueness verification

## Installation

### 1. Create a Virtual Environment

```bash
# Create a new virtual environment with Python 3.10+
python3 -m venv .venv
```

### 2. Activate the Virtual Environment

**On Linux:**
```bash
source .venv/bin/activate
```

**On macOS:**
```bash
source .venv/bin/activate
```

**On Windows:**
```bash
.venv\Scripts\activate
```

### 3. Install the Package

```bash
# Install in editable mode with all dependencies
pip install -e .

# Or install with test dependencies
pip install -e ".[test]"
```

## Running Tests

Once the virtual environment is activated and the package is installed:

```bash
pytest
```

This will run all tests in the `tests/` directory.

## Package Structure

### Core Modules

#### `nash.contract`

The core module implementing nested logit demand and Nash bargaining calculations:

**Demand System Functions:**
- `compute_δ_i()` - Computes mean utilities from prices
- `compute_D_i_l()` - Computes exponentiated utilities and type-level aggregates
- `compute_s_h_l()` - Top-level choice probabilities (has stent vs no stent)
- `compute_s_t_h_l()` - Type-level choice probabilities (DES vs BMS)
- `compute_s_i_t_l()` - Product-level choice probabilities within type
- `compute_s_i_l()` - Full nested share computation
- `compute_market_shares()` - Aggregates shares over loyalty types
- `compute_shares_from_prices()` - Main demand function (prices → market shares)

**Profit Functions:**
- `compute_hospital_profit()` - Hospital profit from stent procedures
- `compute_supplier_profit()` - Supplier profit for a given stent

**Nash Bargaining:**
- `calculate_nash_product()` - Computes Nash bargaining product for a supplier-hospital pair
- `best_response_map()` - Computes best response prices for all products
- `residual()` - Computes residual r(p) = BR(p) - p for convergence checking
- `find_equilibrium()` - Finds Nash bargaining equilibrium prices via gradient descent

**Equilibrium Solver Details:**
- Uses **Adam optimizer** with projection onto price bounds
- Convergence criterion: `||r||_∞ ≤ tol_inf` (infinity norm of residual)
- Real-time progress bar showing both `||r||_2` and `||r||_∞`
- JIT-compiled loss and gradient computation for performance
- Handles numerical issues (NaN detection, gradient clipping)

**Nested Logit Structure:**
The model has four hierarchical levels:
- `i ∈ I`: Individual stent (product)
- `t ∈ T`: Stent type (DES or BMS)
- `h ∈ H`: Top-level decision (has stent or not)
- `l ∈ L`: Loyalty type (doctor-patient brand preference)

#### `nash.initializer`

Parameter initialization module for testing scenarios:

**Functions:**
- `init_simple_params(mode)` - Initializes a simple 2-product model for testing
  - Modes: `"symmetric"`, `"asymmetric"`, `"high_loyalty"`, `"high_elasticity"`
  - Returns: `(static_params, dynamic_params)` tuple

**Parameter Structure:**
- `static_params`: FrozenDict with fixed model structure (n_stents, n_grid, optimization settings)
- `dynamic_params`: Dict with economic parameters (prices, costs, revenues, utilities, etc.)

#### `grennen_specification`

Full 8-product Grennan model specification calibrated to empirical data:

**Model Structure:**
- 6 BMS (Bare Metal Stent) products
- 2 DES (Drug-Eluting Stent) products
- Calibrated to empirical parameters from Grennan (2013)

**Components:**
- Optimization parameters (learning rate, convergence tolerance, max iterations)
- Dimensionality and type indicators
- Preference and nesting parameters (σ, λ values for nested logit)
- Economic primitives (revenues, marginal costs, bargaining parameters)
- Loyalty structure (loyalty weights φ_l, intensity λ_i)
- Price domain (bounds, grids, initial prices)
- Nest-scaling denominators for demand calculations

**Exports:**
- `static_params`: FrozenDict with model dimensions and optimization settings
- `dynamic_params`: Dict with all economic and structural parameters
- `prices0_i`: Initial price vector (midpoint of bounds)

### Simulation Scripts

#### `grennen.py`

Main equilibrium solver for the Grennan model:

```bash
python src/grennen.py
```

**Features:**
- Sets JAX environment variables for CPU optimization
- Runs in **32-bit precision** (float32) by default for speed
- Finds Nash equilibrium prices using gradient descent
- Computes best response prices at equilibrium
- Calculates Jacobian determinant of residual map
- Prints convergence diagnostics and equilibrium results

**Output:**
- Equilibrium prices `p_eq`
- Best response prices at equilibrium `br(p_eq)`
- Residual `r = br(p_eq) - p_eq`
- Jacobian determinant and sign information
- Convergence metrics (iterations, loss, norms)

#### `grennen_reps.py`

Parallel replication testing for equilibrium uniqueness:

```bash
python src/grennen_reps.py [OPTIONS]
```

**Command Line Options:**
- `--x64` - Enable 64-bit precision (default: 32-bit float32)
- `--reps N` - Number of random initial prices to test (default: 100)
- `--jobs N` - Number of parallel workers (default: 8)
- `--tol-inf FLOAT` - Override infinity-norm tolerance
- `--tol-l2 FLOAT` - Override L2-norm tolerance (for display only)
- `--cmp-tol FLOAT` - Comparison tolerance across replications (default: 5e-3)

**Example:**
```bash
# Run with 64-bit precision and 200 replications
python src/grennen_reps.py --x64 --reps 200 --jobs 12

# Run with custom solver tolerance
python src/grennen_reps.py --tol-inf 1e-6 --reps 50
```

**Features:**
- Tests equilibrium uniqueness by solving from random initial prices
- Parallel execution using joblib for speed
- Suppresses individual solver output (quiet mode)
- Compares all equilibria to check for uniqueness
- Saves results to timestamped NPZ and JSON files in `outputs/` directory
- Reports convergence failures and tolerance violations
- **Can be imported as a module** without running computations

**Output Files:**
- `outputs/grennen_reps_YYYYMMDD_HHMMSS.npz` - All numeric results (compressed)
- `outputs/grennen_reps_YYYYMMDD_HHMMSS.json` - Configuration metadata

**Loading Results Programmatically:**

The module can be imported to load previously computed results without re-running:

```python
from src.grennen_reps import load_replication_results

# Load most recent results
config, data = load_replication_results()

# Or load specific file
config, data = load_replication_results("outputs/grennen_reps_20251106_120000.npz")

# Access results
print(f"Loaded {config['n_replications']} replications")
print(f"Mean equilibrium: {data['mean_eq']}")
print(f"All equilibria shape: {data['equilibria'].shape}")
print(f"Convergence iterations: {data['iterations']}")
```

## Dependencies

**Core Dependencies:**
- **jax** (0.8.0) - Automatic differentiation and JIT compilation
- **jaxlib** (0.8.0) - JAX backend
- **numpy** (2.3.4) - Numerical arrays
- **optax** (0.2.6) - Gradient-based optimization (Adam optimizer)
- **flax** (0.12.0) - Neural network library (used for FrozenDict)
- **rich** (14.2.0) - Terminal progress bars and formatting
- **joblib** (1.5.2) - Parallel processing for replications

**Test Dependencies:**
- **pytest** (8.4.2) - Testing framework

## Configuration

### JAX Environment Variables

The package configures JAX for optimal CPU performance:

```python
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=true "
    "intra_op_parallelism_threads=12 "
    "inter_op_parallelism_threads=3 "
    "--xla_cpu_use_thunk_runtime=false"
)
os.environ["JAX_ENABLE_X64"] = "0"  # float32 by default
```

**Precision Modes:**
- **32-bit (default)**: Faster computation, suitable for most cases
- **64-bit (with --x64)**: Higher precision, useful for sensitive analyses

### Solver Parameters

Default convergence settings in `static_params`:

```python
{
    "tol_inf": 1e-5,        # Infinity norm threshold (stopping criterion)
    "tol_l2": 1e-8,         # L2 norm threshold (display only)
    "max_iter": 10000,      # Maximum iterations
    "refresh_rate": 10,     # Progress bar update frequency
    "optimizer_params": {
        "learning_rate": 0.01,
        "b1": 0.9,
        "b2": 0.999,
        "eps": 1e-8
    }
}
```

**Note:** Only `tol_inf` is used for the stopping criterion. The solver stops when `||r||_∞ ≤ tol_inf`, where `r = BR(p) - p` is the residual.

## Project Structure

```
fornow/
├── src/
│   ├── __init__.py
│   ├── grennen.py                 # Main equilibrium solver
│   ├── grennen_reps.py            # Parallel replication testing
│   ├── grennen_specification.py   # Grennan model parameters (8 products)
│   └── nash/
│       ├── __init__.py
│       ├── contract.py            # Nested logit demand & Nash bargaining
│       └── initializer.py         # Test parameter initialization
├── tests/
│   ├── test_br.py                 # Best response tests
│   └── test_contract.py           # Demand & bargaining tests
├── setup.py                       # Package configuration
├── pytest.ini                     # Pytest configuration
└── README.md                      # This file
```

## Usage Examples

### Basic Equilibrium Computation

```python
from grennen_specification import static_params, dynamic_params, prices0_i
from nash.contract import find_equilibrium

# Find equilibrium
p_eq, info = find_equilibrium(static_params, dynamic_params, prices0_i)

print(f"Equilibrium found in {info['n_iter']} iterations")
print(f"Final residual: ||r||_∞ = {info['r_inf']:.2e}")
print(f"Equilibrium prices: {p_eq}")
```

### Testing with Simple Parameters

```python
from nash.initializer import init_simple_params
from nash.contract import find_equilibrium

# Initialize symmetric 2-product test case
static_params, dynamic_params = init_simple_params("symmetric")
prices0 = dynamic_params["prices0_i"]

# Solve
p_eq, info = find_equilibrium(static_params, dynamic_params, prices0)
```

### Loading Previous Replication Results

```python
from src.grennen_reps import load_replication_results
import numpy as np

# Load most recent replication test results
config, data = load_replication_results()

# Analyze equilibrium variation across replications
equilibria = data['equilibria']  # shape: (n_replications, n_stents)
std_across_reps = np.std(equilibria, axis=0)

print(f"Configuration: {config['n_replications']} reps, {config['n_stents']} products")
print(f"Standard deviation across replications: {std_across_reps}")
print(f"Mean iterations: {np.mean(data['iterations']):.1f}")
print(f"All converged to same equilibrium: {np.max(std_across_reps) < config['tolerance']}")
```

# Solve
p_eq, info = find_equilibrium(static_params, dynamic_params, prices0)
```

### Command-Line Replication Testing

```bash
# Standard test with default settings
python src/grennen_reps.py

# High-precision test with more replications
python src/grennen_reps.py --x64 --replications 500 --jobs 16

# Custom tolerance for tight convergence
python src/grennen_reps.py --tol-inf 1e-7 --replications 100
```

## Algorithm Details

### Nash-in-Nash Equilibrium

The model solves for prices where each supplier-hospital pair maximizes the Nash bargaining product:

$$\max_{p_i} \left[\Pi^s_i(p) - \Pi^s_i(p^{-i})\right]^{\tau_i} \left[\Pi^h(p) - \Pi^h(p^{-i})\right]^{1-\tau_i}$$

where:
- $\Pi^s_i(p)$: Supplier profit with product i at prices p
- $\Pi^h(p)$: Hospital profit at prices p
- $p^{-i}$: Outside option prices (without product i)
- $\tau_i$: Bargaining weight for product i

### Gradient-Based Solution

Instead of traditional fixed-point iteration, we minimize:

$$L(p) = \|BR(p) - p\|_2^2$$

using Adam with:
1. **Gradient computation**: JIT-compiled automatic differentiation
2. **Bound projection**: Clip prices to $[p^{min}_i, p^{max}_i]$ after each step
3. **Convergence check**: Stop when $\|BR(p) - p\|_\infty \leq \epsilon$

This approach is more robust than pure fixed-point iteration for complex demand systems.

## References

Grennan, M. (2013). "Price Discrimination and Bargaining: Empirical Evidence from Medical Devices." 
*American Economic Review*, 103(1): 145-177.

## License

This project is for research and educational purposes.

## Contributing

When contributing, please:
1. Run tests with `pytest` before submitting
2. Ensure code follows the existing style
3. Update documentation for new features
4. Test with both 32-bit and 64-bit precision modes
