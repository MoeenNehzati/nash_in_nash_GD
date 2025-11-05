# Fornow - Nash Bargaining with Nested Logit Demand

A Python package for simulating Nash bargaining equilibria in nested logit demand systems, following the approach of Grennan (2013) for medical device markets.

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

### `nash.contract`

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
- `find_equilibrium()` - Finds Nash bargaining equilibrium prices

**Nested Logit Structure:**
The model has four hierarchical levels:
- `i ∈ I`: Individual stent (product)
- `t ∈ T`: Stent type (DES or BMS)
- `h ∈ H`: Top-level decision (has stent or not)
- `l ∈ L`: Loyalty type (doctor-patient brand preference)

### `nash.initializer`

Parameter initialization module for testing scenarios:

**Functions:**
- `init_simple_params(mode)` - Initializes a simple 2-product model for testing
  - Modes: `"symmetric"`, `"asymmetric"`, `"high_loyalty"`, `"high_elasticity"`
  - Returns: `(static_params, dynamic_params)` tuple

**Parameter Structure:**
- `static_params`: Fixed model structure (n_stents, n_grid, optimization settings)
- `dynamic_params`: Economic parameters (prices, costs, revenues, utilities, etc.)

### `grennen_specification`

Full 8-product Grennan model specification:

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

## Running the Main Script

Execute the main simulation:

```bash
python src/grennen.py
```

This script:
1. Imports Grennan model parameters from `grennen_specification.py`
2. Finds Nash bargaining equilibrium prices
3. Computes best response prices
4. Prints equilibrium results and convergence information

**Note:** The script sets JAX environment variables for CPU performance optimization.

## Dependencies

- **jax** (0.8.0) - Automatic differentiation and JIT compilation
- **jaxlib** (0.8.0) - JAX backend
- **numpy** (2.3.4) - Numerical arrays
- **optax** (0.2.6) - Gradient-based optimization
- **flax** (0.12.0) - Neural network library (used for FrozenDict)
- **rich** (14.2.0) - Terminal progress bars and formatting

Test dependencies:
- **pytest** (8.4.2) - Testing framework

## Project Structure

```
fornow/
├── src/
│   ├── __init__.py
│   ├── grennen.py                # Main simulation script
│   ├── grennen_specification.py  # Grennan model parameters (8 products)
│   └── nash/
│       ├── __init__.py
│       ├── contract.py           # Nested logit & Nash bargaining
│       └── initializer.py        # Test parameter initialization
├── tests/
│   ├── test_br.py                # Best response tests
│   └── test_contract.py          # Demand & bargaining tests
├── setup.py                      # Package configuration
├── pytest.ini                    # Pytest configuration
└── README.md                     # This file
```

## References

Grennan, M. (2013). "Price Discrimination and Bargaining: Empirical Evidence from Medical Devices." 
*American Economic Review*, 103(1): 145-177.
