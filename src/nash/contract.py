"""
Nested Logit Notation and Hierarchical Structure
================================================

This module implements the nested-logit demand structure used in Grennan-style
models of stent choice, with the following four index levels:

    i ∈ I   : individual stent (product)
    t ∈ T   : stent type (DES or BMS)
    h ∈ H   : top-level decision (has stent or not)
    l ∈ L   : loyalty type (doctor–patient brand preference)

The hierarchy of choices is:

    (i | t, l)  →  (t | h, l)  →  (h | l)

so that total share for product i (conditional on loyalty type l) is:

    s_{i l} = s_{i|t,l} · s_{t|h,l} · s_{h|l}

----------------------------------------------------------------------
Notation summary
----------------------------------------------------------------------

CAPITAL-LETTER objects represent *inclusive values* (unnormalized sums),
while lowercase *s* variables represent *shares* (probabilities).

  ┌───────────────────────┬────────┬─────────────────────────────────────┐
  │ Symbol                │ Shape  │ Definition / Interpretation        │
  ├───────────────────────┼────────┼─────────────────────────────────────┤
  │ δ_i                   │ (I,)   │ Mean utility of stent i                                    │
  │ λ_i                   │ (I,)   │ Loyalty bonus for stent i                                  │
  │ σ_t                   │ (T,)   │ Within-type dissimilarity                                  │
  │ σ_stent               │ ()     │ Top-level dissimilarity (stent vs                          │
  │                       │        │   no-stent)                                                │
  │ D_{i l}               │ (I,L)  │ exp((δ_i + λ_i·1[i=l]) / denom_i);                         │
  │                       │        │ product-level *exponentiated util*                         │
  │ S_{t l}               │ (T,L)  │ ∑_{i∈t} D_{i l}  (inclusive value)                        │
  │ V_{h l}               │ (1,L)  │ ∑_t S_{t l}^{1−σ_t}                                        │
  │ s_{i|t,l}             │ (I,L)  │ D_{i l} / ∑_{i'∈t} D_{i' l}                                │
  │ s_{t|h,l}             │ (T,L)  │ S_{t l}^{1−σ_t} / ∑_{t'}S_{t' l}^{1−σ_t'}                  │
  │ s_{h|l}               │ (L,)   │ V_{h l}^{1−σ_stent} / [1+V_{h l}^{1−σ_stent}]               │
  │ s_{i l}               │ (I,L)  │ Total share = s_{i|t,l}·s_{t|h,l}·s_{h|l}                   │
  └───────────────────────┴────────┴─────────────────────────────────────┘

----------------------------------------------------------------------
Matrix notation (Einstein convention)
----------------------------------------------------------------------

Let `hastype_i_t` be a one-hot matrix of shape (I, T)
indicating each stent's membership in a type.

Key operations (in Einstein form):

    # 1. Type-level inclusive value
    S_{t l} = ∑_i hastype_{i t} D_{i l}
        → jnp.einsum('it,il->tl', hastype_i_t, D_i_l)

    # 2. Top-level inclusive value (all stents)
    V_{h l} = ∑_t S_{t l}^{1−σ_t}

    # 3. Within-type share (i | t, l)
    s_{i t l} = D_{i l} / ∑_{i'∈t} D_{i' l}

    # 4. Share of each type (t | h, l)
    s_{t h l} = S_{t l}^{1−σ_t} / ∑_{t'} S_{t' l}^{1−σ_t'}

    # 5. Share of having a stent (h | l)
    s_{h l} = V_{h l}^{1−σ_stent} / (1 + V_{h l}^{1−σ_stent})

Final probability decomposition:

    s_{i l} = s_{i t l} · s_{t h l} · s_{h l}

----------------------------------------------------------------------
Implementation notes
----------------------------------------------------------------------

- Use `einsum('it,il->tl', ...)` for all within-type aggregations.
- Use per-type dissimilarities σ_t and a top-level σ_stent.
- `D_i_l` can be constructed once via:
      D_i_l = exp((δ_i + λ_i·1[i=l]) / D_denom_i_l)
- Use log-space computations if overflow/underflow becomes an issue.

----------------------------------------------------------------------
Shapes
----------------------------------------------------------------------

    D_i_l          : (I, L)
    S_t_l          : (T, L)
    V_h_l          : (1, L)
    s_i_t_l        : (I, L)
    s_t_h_l        : (T, L)
    s_h_l          : (L,)
    s_i_l          : (I, L)

All columns (over i or t) sum to 1 at each conditioning level.
"""

import jax
import jax.numpy as jnp
from jax import value_and_grad, jit, lax, vmap
from functools import partial
import optax
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn

# Simple no-op progress object matching Rich's API used here
class DummyProgress:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False  # don't suppress exceptions

    def add_task(self, *args, **kwargs):
        return 0

    def update(self, *args, **kwargs):
        pass

@partial(jit, static_argnames=("static_params", "nash_objective_fn"))
def argmax_f(static_params, dynamic_params, prices_i: jnp.ndarray, i, nash_objective_fn):
    """
    Find the optimal price for product i using differentiable grid search.
    
    Uses JAX's custom_root for differentiability:
    - Forward pass: Grid search over precomputed price grid
    - Backward pass: Implicit differentiation through first-order condition
    
    Parameters
    ----------
    static_params : FrozenDict
        Static parameters (n_stents, n_grid, etc.)
    dynamic_params : dict
        Dynamic parameters including price_grids_i
    prices_i : Array, shape (I,)
        Current prices for all products
    i : int
        Index of product to optimize
    nash_objective_fn : callable
        Objective function to maximize (Nash bargaining product)
        Should have signature: nash_objective_fn(static_params, dynamic_params, prices_i, price_i, i)
        
    Returns
    -------
    x_star : Array
        Optimal price for product i that maximizes nash_objective_fn
    """
    # Get precomputed price grid for this product
    grid = dynamic_params["price_grids_i"][i]  # Shape: (n_grid,)

    # First-order condition: derivative of objective w.r.t. price
    def FOC(x):
        return jax.grad(nash_objective_fn, argnums=3)(static_params, dynamic_params, prices_i, x, i)

    # Forward pass: Grid search to find best price
    def solve(F, x0):
        # Evaluate objective at all grid points
        vals = vmap(lambda x: nash_objective_fn(static_params, dynamic_params, prices_i, x, i))(grid)
        # Return price with highest objective value
        x_star = grid[jnp.argmax(vals)]
        return x_star

    # Backward pass: Implicit differentiation for gradients
    def tangent_solve(matvec, rhs):
        # Compute Jacobian and regularize to avoid division by zero
        J = matvec(jnp.array(1.0))
        J = jnp.where(jnp.abs(J) < 1e-6, 1e-6, J)
        return rhs / J

    # Use custom_root for differentiable optimization
    x0 = jnp.array(0.0)  # Initial guess (not used in forward pass)
    return lax.custom_root(FOC, x0, solve, tangent_solve)

# ============================================================================
# Demand system primitives
# ============================================================================

def compute_δ_i(static_params, dynamic_params, prices_i: jnp.ndarray) -> jnp.ndarray:
    """
    Compute mean utility for each product.
    
    Mean utility combines base utility and price sensitivity:
        δᵢ = θᵢ + θ_p × priceᵢ
    
    Parameters
    ----------
    static_params : FrozenDict
        Static parameters (unused here)
    dynamic_params : dict
        Contains θ_i (base utility) and θ_p (price coefficient)
    prices_i : Array, shape (I,)
        Current prices for each product
        
    Returns
    -------
    δ : Array, shape (I,)
        Mean utility for each product
        
    Notes
    -----
    θ_p is typically negative (higher price reduces utility)
    Prices are in $1000 units for numerical stability
    """
    return dynamic_params["θ_i"] + dynamic_params["θ_p"] * prices_i

# ============================================================================
# Nested logit demand system
# ============================================================================
# Index notation:
#   i ∈ I : individual stent (product)
#   l ∈ L : loyalty type (brand preference)
#   t ∈ T : stent type (DES or BMS)
#   h ∈ H : has stent or not (top-level choice)

def compute_D_i_l(static_params, dynamic_params, δ: jnp.ndarray):
    """
    Compute exponentiated scaled utilities and type-level inclusive values.
    
    This is the first step in the nested logit demand system:
        D_{i,l} = exp((δᵢ + loyalty_bonusᵢ,ₗ) / denomᵢ,ₗ)
    
    where denom depends on the nesting structure (see init_params).
    
    Parameters
    ----------
    static_params : FrozenDict
        Static parameters (unused here)
    dynamic_params : dict
        Contains D_loyalty_i_l (loyalty bonuses), D_denom_i_l (denominators),
        and hastype_i_t (product-type mapping)
    δ : Array, shape (I,)
        Mean utilities for each product
        
    Returns
    -------
    D_i_l : Array, shape (I, L)
        Exponentiated scaled utilities for each product-loyalty pair
    S_t_l : Array, shape (T, L)
        Type-level inclusive values: sum of D_i_l within each type
        
    Notes
    -----
    S_t_l is computed here for efficiency since it's used in multiple
    downstream calculations (shares, probabilities, etc.)
    """
    # Scale utilities by nest-specific denominators and add loyalty bonuses
    log_d = (δ[:, None] + dynamic_params["D_loyalty_i_l"]) / dynamic_params["D_denom_i_l"]
    D_i_l = jnp.exp(log_d)
    
    # Aggregate to type level: S_{t,l} = Σᵢ∈t D_{i,l}
    hastype_i_t = dynamic_params["hastype_i_t"]  # One-hot: (I, T)
    S_t_l = hastype_i_t.T @ D_i_l  # Matrix multiply: (T, I) @ (I, L) = (T, L)
    
    return D_i_l, S_t_l

def safe_pow(x, alpha, eps=1e-6):
    # ensure finite derivatives when x → 0
    x_safe = jnp.maximum(x, eps)
    return x_safe ** alpha


def compute_s_h_l(static_params, dynamic_params, S_t_l):
    """
    Compute top-level choice probabilities: stent vs. no-stent.
    
    Probability of choosing any stent (vs. outside option):
        s_{h,l} = V_{h,l}^(1-σ_stent) / (1 + V_{h,l}^(1-σ_stent))
    
    where V_{h,l} = Σₜ S_{t,l}^(1-σₜ) is the top-level inclusive value.
    
    Parameters
    ----------
    static_params : FrozenDict
        Static parameters (unused here)
    dynamic_params : dict
        Contains σ_t (within-type dissimilarity) and σ_stent (top-level)
    S_t_l : Array, shape (T, L)
        Type-level inclusive values
        
    Returns
    -------
    s_h_l : Array, shape (L,)
        Probability of choosing any stent for each loyalty type
        
    Notes
    -----
    This is the top level of the nested logit: choose stent vs. outside option.
    The dissimilarity parameter σ_stent controls substitution at this level.
    """
    σ_t = dynamic_params["σ_t"]          # Shape: (T,)
    σ_stent = dynamic_params["σ_stent"]  # Scalar
    
    # Compute top-level inclusive value: V_{h,l} = Σₜ S_{t,l}^(1-σₜ)
    V_h_l = jnp.sum(S_t_l ** (1 - σ_t)[:, None], axis=0)  # Shape: (L,)
    
    # Logit probability: fraction choosing any stent vs. outside option
    s_h_l = (V_h_l ** (1 - σ_stent)) / (1 + V_h_l ** (1 - σ_stent))
    
    return s_h_l


def compute_s_t_h_l(static_params, dynamic_params, S_t_l):
    """
    Compute type choice probabilities conditional on choosing a stent.
    
    Probability of choosing type t given stent choice and loyalty:
        s_{t|h,l} = S_{t,l}^(1-σₜ) / Σₜ′ S_{t′,l}^(1-σₜ′)
    
    Parameters
    ----------
    static_params : FrozenDict
        Static parameters (unused here)
    dynamic_params : dict
        Contains σ_t (within-type dissimilarity parameters)
    S_t_l : Array, shape (T, L)
        Type-level inclusive values
        
    Returns
    -------
    s_t_h_l : Array, shape (T, L)
        Conditional probability of each type given stent choice
        
    Notes
    -----
    This is the middle level of the nested logit: choose DES vs. BMS.
    Each type has its own dissimilarity parameter σₜ controlling
    substitution among products within that type.
    """
    σ_t = dynamic_params["σ_t"]  # Shape: (T,)
    
    # Raise inclusive values to (1-σₜ) power
    S_t_l_pow = S_t_l ** (1 - σ_t)[:, None]  # Shape: (T, L)
    
    # Normalize to get probabilities (sum to 1 over types)
    denom_l = S_t_l_pow.sum(axis=0, keepdims=True)  # Shape: (1, L)
    s_t_h_l = S_t_l_pow / denom_l                   # Shape: (T, L)
    
    return s_t_h_l


def compute_s_i_t_l(static_params, dynamic_params, D_i_l, S_t_l):
    """
    Compute product choice probabilities conditional on type and loyalty.
    
    Probability of choosing product i given type t and loyalty l:
        s_{i|t,l} = D_{i,l} / Σᵢ′∈t D_{i′,l}
    
    Parameters
    ----------
    static_params : FrozenDict
        Static parameters (unused here)
    dynamic_params : dict
        Contains hastype_i_t (product-type membership matrix)
    D_i_l : Array, shape (I, L)
        Exponentiated scaled utilities
    S_t_l : Array, shape (T, L)
        Type-level inclusive values (denominators)
        
    Returns
    -------
    s_i_t_l : Array, shape (I, L)
        Conditional probability of each product given its type
        
    Notes
    -----
    This is the bottom level of the nested logit: choose specific product
    within chosen type (e.g., which BMS stent to use).
    """
    hastype_i_t = dynamic_params["hastype_i_t"]  # Shape: (I, T)
    
    # Map type-level denominators back to product level using Einstein notation
    # denom_{i,l} = Σₜ hastype_{i,t} × S_{t,l}
    denom_i_l = jnp.einsum("it,tl->il", hastype_i_t, S_t_l)  # Shape: (I, L)
    
    # Normalize: probability = numerator / denominator
    s_i_t_l = D_i_l / denom_i_l
    
    return s_i_t_l



def compute_s_t_l(static_params, dynamic_params, D_i_l: jnp.ndarray) -> jnp.ndarray:
    """
    Compute stent type choice probabilities by loyalty type.
    
    This function calculates the probability of choosing each stent type
    (DES vs BMS) conditional on loyalty type, accounting for the type-level
    nesting with dissimilarity parameter σ_t.
    
    Formula:
        1. Aggregate products to types: S_t_l = Σ_{i∈t} D_{i,l}
        2. Apply dissimilarity power: S̃_t_l = S_t_l^(1 - σ_t)
        3. Normalize: s_{t,l} = S̃_t_l / Σ_t S̃_t_l
    
    Parameters
    ----------
    static_params : FrozenDict
        Static parameters (unused here)
    dynamic_params : dict
        Contains hastype_i_t (product-type mapping) and σ_t (dissimilarity parameters)
    D_i_l : Array, shape (I, L)
        Exponentiated scaled utilities for each product-loyalty pair
        
    Returns
    -------
    s_t_l : Array, shape (T, L)
        Type choice probabilities conditional on loyalty
        For each loyalty type: Σ_t s_{t,l} = 1.0
        
    Notes
    -----
    The σ_t parameter captures within-type similarity:
    - σ_t = 0: products within type are independent (reduces to standard logit)
    - σ_t > 0: products within type are substitutes (nested logit structure)
    - σ_t must be in [0, 1) for consistency with random utility theory
    """
    hastype_i_t = dynamic_params["hastype_i_t"]  # (I, T)
    σ_t = dynamic_params["σ_t"]                  # (T,)
    S_t_l = jnp.einsum("it,il->tl", hastype_i_t, D_i_l)  # (T, L)
    # Step 2: apply the (1 - σ_t) power for each type
    one_minus_σ_t = 1.0 - σ_t                             # (T,)
    S_t_l_powered = jnp.einsum("tl,t->tl", S_t_l ** one_minus_σ_t[:, None], jnp.ones_like(one_minus_σ_t))
    # Step 3: normalize across t so that ∑_t s_{t,l} = 1
    denom_l = S_t_l_powered.sum(axis=0, keepdims=True)    # (1, L)
    s_t_l = S_t_l_powered / denom_l                       # (T, L)
    return s_t_l

def compute_s_i_l(static_params, dynamic_params, D_i_l, S_t_l):
    """
    Compute total product choice probabilities by loyalty type.
    
    Combines all three levels of the nested logit:
        s_{i,l} = s_{i|t,l} × s_{t|h,l} × s_{h|l}
    
    This gives the probability that a consumer of loyalty type l
    chooses product i, accounting for:
    1. Choice of stent vs. no-stent (h)
    2. Choice of type within stents (t|h)  
    3. Choice of product within type (i|t)
    
    Parameters
    ----------
    static_params : FrozenDict
        Static parameters (unused here)
    dynamic_params : dict
        Contains hastype_i_t, σ_t, σ_stent
    D_i_l : Array, shape (I, L)
        Exponentiated scaled utilities
    S_t_l : Array, shape (T, L)
        Type-level inclusive values
        
    Returns
    -------
    s_i_l : Array, shape (I, L)
        Probability of choosing each product by loyalty type
        
    Notes
    -----
    Uses Einstein notation for efficient computation:
        s_{i,l} = s_{i,l}^prod × s_{t,l}^type × hastype_{i,t} × s_l^top
    where hastype_{i,t} ensures we only multiply by the relevant type share.
    """
    hastype_i_t = dynamic_params["hastype_i_t"]
    
    # Compute conditional probabilities at each level
    s_h_l = compute_s_h_l(static_params, dynamic_params, S_t_l)      # Top: stent vs. none
    s_t_h_l = compute_s_t_h_l(static_params, dynamic_params, S_t_l)  # Middle: type choice
    s_i_t_l = compute_s_i_t_l(static_params, dynamic_params, D_i_l, S_t_l)  # Bottom: product
    
    # Combine all levels using Einstein summation
    # "il,tl,it,l->il" means: multiply and sum appropriately to get (I,L) result
    s_i_l = jnp.einsum("il,tl,it,l->il", s_i_t_l, s_t_h_l, hastype_i_t, s_h_l)
    
    return s_i_l

def compute_market_shares(static_params, dynamic_params, s_i_l):
    """
    Aggregate product probabilities across loyalty types to get market shares.
    
    Market share = weighted average across loyalty types:
        sᵢ = Σₗ φₗ × s_{i,l}
    
    where φₗ is the population fraction with loyalty type l.
    
    Parameters
    ----------
    static_params : FrozenDict
        Static parameters (unused here)
    dynamic_params : dict
        Contains φ_l (loyalty type distribution)
    s_i_l : Array, shape (I, L)
        Product choice probabilities by loyalty type
        
    Returns
    -------
    s_i : Array, shape (I,)
        Market share for each product (sums to < 1, remainder is outside option)
    """
    φ_l = dynamic_params["φ_l"]  # Shape: (L,) - population weights
    return jnp.einsum("il,l->i", s_i_l, φ_l)

# ============================================================================
# Main demand function
# ============================================================================

def compute_shares_from_prices(static_params, dynamic_params, prices_i: jnp.ndarray) -> jnp.ndarray:
    """
    Compute market shares from prices using nested logit demand system.
    
    This is the main demand function that chains together:
    1. Mean utilities from prices
    2. Exponentiated utilities and type aggregates
    3. Conditional choice probabilities at each nest level
    4. Market shares by aggregating over loyalty types
    
    Parameters
    ----------
    static_params : FrozenDict
        Static parameters (n_stents, etc.)
    dynamic_params : dict
        All demand parameters (utilities, dissimilarities, loyalty, etc.)
    prices_i : Array, shape (I,)
        Current prices for each product
        
    Returns
    -------
    s_i : Array, shape (I,)
        Market share for each product
        
    Notes
    -----
    This function composes the full nested logit demand system.
    It's called repeatedly during equilibrium computation and profit calculations.
    """
    # Step 1: Compute mean utilities from prices
    δ_i = compute_δ_i(static_params, dynamic_params, prices_i)
    
    # Step 2: Compute exponentiated utilities and type-level aggregates
    D_i_l, S_t_l = compute_D_i_l(static_params, dynamic_params, δ_i)
    
    # Step 3: Compute choice probabilities conditional on loyalty type
    s_i_l = compute_s_i_l(static_params, dynamic_params, D_i_l, S_t_l)
    
    # Step 4: Aggregate over loyalty types to get market shares
    return compute_market_shares(static_params, dynamic_params, s_i_l)


# ============================================================================
# Profit functions
# ============================================================================

def compute_hospital_profit(
    static_params, dynamic_params, prices_i: jnp.ndarray, excluded_stent: int | None = None
) -> jnp.ndarray:
    """
    Compute hospital profit from stent procedures.
    
    Hospital profit = Σ_i (Revenue_i - Price_i) × s_i(prices)
    
    The hospital receives a fixed reimbursement per procedure and pays the
    negotiated stent price. Profit is the markup times market share.
    
    Parameters
    ----------
    static_params : FrozenDict
        Static parameters
    dynamic_params : dict
        Contains hospital_revenues_i (reimbursement per procedure)
    prices_i : Array, shape (I,)
        Current prices for each product
    excluded_stent : int or None
        If provided, simulates a disagreement scenario where this stent is
        unavailable. Its price is set prohibitively low and share forced to 0.
        
    Returns
    -------
    profit : scalar
        Total hospital profit across all stent procedures
        
    Notes
    -----
    When excluded_stent is used, this computes the hospital's disagreement payoff
    in bargaining - what they would earn if negotiations fail with that supplier.
    """
    if excluded_stent is not None:
        # Temporarily modify prices for disagreement scenario
        prices_mod = prices_i.at[excluded_stent].set(1e-10)
        shares = compute_shares_from_prices(static_params, dynamic_params, prices_mod)
        shares = shares.at[excluded_stent].set(0.)
        prices_used = prices_i  # Use original prices for profit calculation
    else:
        shares = compute_shares_from_prices(static_params, dynamic_params, prices_i)
        prices_used = prices_i

    hospital_revenues = dynamic_params["hospital_revenues_i"]
    return jnp.sum((hospital_revenues - prices_used) * shares)


# -----------------------------------------------------------------------------
# Supplier profit
# -----------------------------------------------------------------------------
def compute_supplier_profit(
    static_params, dynamic_params, prices_i: jnp.ndarray, i: int, excluded: bool = False
) -> jnp.ndarray:
    """
    Compute profit for supplier i from selling stents.
    
    Supplier profit = (Price_i - Cost_i) × s_i(prices)
    
    Each supplier earns the negotiated markup times their market share.
    
    Parameters
    ----------
    static_params : FrozenDict
        Static parameters
    dynamic_params : dict
        Contains marginal_costs_i (production costs per stent)
    prices_i : Array, shape (I,)
        Current prices for each product
    i : int
        Index of the supplier whose profit to compute
    excluded : bool, default=False
        If True, return 0 (disagreement payoff when negotiations fail)
        
    Returns
    -------
    profit : scalar
        Supplier i's profit from stent sales
        
    Notes
    -----
    When excluded=True, this represents the supplier's outside option value
    in Nash bargaining - zero profit if they don't reach an agreement.
    """
    if excluded:
        return jnp.array(0, dtype=jnp.float64)

    shares = compute_shares_from_prices(static_params, dynamic_params, prices_i)
    markup = prices_i[i] - dynamic_params["marginal_costs_i"][i]
    return markup * shares[i]


# ============================================================================
# Nash bargaining
# ============================================================================

def calculate_nash_product(static_params, dynamic_params, prices_i: jnp.ndarray, i: int) -> jnp.ndarray:
    """
    Compute Nash bargaining product for supplier i and the hospital.
    
    The Nash product represents the joint surplus from agreement weighted
    by bargaining power:
        
        N_i = [Supplier_Gain_i]^(b_i) × [Hospital_Gain_i]^(1 - b_i)
    
    For numerical stability, we compute the log Nash product:
        log N_i = b_i × log(Supplier_Gain_i) + (1 - b_i) × log(Hospital_Gain_i)
    
    Gains are measured relative to disagreement payoffs:
    - Supplier_Gain_i = Profit_i(with i) - Profit_i(without i)
    - Hospital_Gain_i = Profit_H(with i) - Profit_H(without i)
    
    Parameters
    ----------
    static_params : FrozenDict
        Static parameters
    dynamic_params : dict
        Contains bp_i (bargaining power weights)
    prices_i : Array, shape (I,)
        Current prices for each product
    i : int
        Index of the supplier bargaining with the hospital
        
    Returns
    -------
    log_nash : scalar
        Logarithm of the Nash product (for numerical stability)
        
    Notes
    -----
    - Floors at 1e-10 prevent log(0) when gains are negative or zero
    - b_i ∈ [0, 1] represents supplier i's bargaining power
    - When b_i = 0.5, bargaining power is symmetric
    - The Nash solution maximizes this product
    """
    # Supplier side: profit with and without agreement
    supplier_on = compute_supplier_profit(static_params, dynamic_params, prices_i, i, excluded=False)
    supplier_off = compute_supplier_profit(static_params, dynamic_params, prices_i, i, excluded=True)
    supplier_gains = jnp.maximum(supplier_on - supplier_off, 1e-10)

    # Hospital side: profit with and without this supplier
    hospital_on = compute_hospital_profit(static_params, dynamic_params, prices_i, excluded_stent=None)
    hospital_off = compute_hospital_profit(static_params, dynamic_params, prices_i, excluded_stent=i)
    hospital_gains = jnp.maximum(hospital_on - hospital_off, 1e-10)

    # Combine using log for numerical stability
    b_i = dynamic_params["bp_i"][i]  # bargaining weight for product i
    log_nash = b_i * jnp.log(supplier_gains) + (1.0 - b_i) * jnp.log(hospital_gains)
    return log_nash

def nash_objective(static_params, dynamic_params, prices_i: jnp.ndarray, price_i, i):
    """
    Objective function for Nash bargaining: log Nash product at given price.
    
    This function evaluates the Nash product when supplier i sets price_i,
    holding all other prices fixed. Used as the objective in argmax_f.
    
    Parameters
    ----------
    static_params : FrozenDict
        Static parameters
    dynamic_params : dict
        Model parameters
    prices_i : Array, shape (I,)
        Current prices for all products
    price_i : scalar
        Candidate price for supplier i
    i : int
        Index of the supplier setting their price
        
    Returns
    -------
    log_nash : scalar
        Log Nash product evaluated at this price configuration
        
    Notes
    -----
    This function is passed to argmax_f to find the price that maximizes
    the Nash product for supplier i.
    Temporarily updates prices_i[i] to evaluate objective at price_i.
    """
    # Temporarily update price for supplier i
    prices_new = prices_i.at[i].set(price_i)
    return calculate_nash_product(static_params, dynamic_params, prices_new, i)

@partial(jit, static_argnames="static_params")
def best_response_map(static_params, dynamic_params, prices_i: jnp.ndarray):
    """
    Compute best response prices for all suppliers simultaneously.
    
    For each supplier i, finds the price that maximizes their Nash product
    with the hospital, holding other prices fixed. This is the key mapping
    for the Nash-in-Nash equilibrium.
    
    Implementation uses a simple Python loop instead of vmap because:
    - Loop is 1.35x faster for small problems (8 products)
    - vmap has overhead that dominates for this problem size
    
    Parameters
    ----------
    static_params : FrozenDict
        Contains n_stents (number of products)
    dynamic_params : dict
        All model parameters
    prices_i : Array, shape (I,)
        Current prices for all products
        
    Returns
    -------
    br_prices : Array, shape (I,)
        Best response price for each supplier
        
    Notes
    -----
    This function is JIT-compiled for efficiency. The vmap version is
    commented out below but kept for reference.
    
    A Nash equilibrium satisfies BR(p*) = p* (fixed point condition).
    """
    I = static_params["n_stents"]
    indices = jnp.arange(I)
    return vmap(argmax_f, in_axes=(None, None, None, 0, None))(
        static_params, dynamic_params, prices_i, indices, nash_objective
    )

def residual(static_params, dynamic_params, prices_i):
    return best_response_map(static_params, dynamic_params, prices_i) - prices_i

def loss_fn(static_params, dynamic_params, prices_i):
    """
    Fixed-point residual loss for Nash equilibrium.
    
    Measures how far current prices are from a Nash equilibrium:
        L(p) = Σ_i [BR_i(p) - p_i]²
    
    Where BR_i(p) is supplier i's best response to prices p.
    At equilibrium, BR(p*) = p* so L(p*) = 0.
    
    Parameters
    ----------
    static_params : FrozenDict
        Static parameters
    dynamic_params : dict
        Model parameters
    prices_i : Array, shape (I,)
        Current candidate prices
        
    Returns
    -------
    loss : scalar
        Sum of squared residuals
        
    Notes
    -----
    Minimizing this loss function finds the Nash equilibrium.
    Gradient descent on L(p) converges to a fixed point of BR(·).
    """
    r = residual(static_params, dynamic_params, prices_i)
    return jnp.sum(r ** 2)

# Precompile loss, gradient, and residual norms for efficiency
def loss_with_aux(static_params, dynamic_params, prices_i):
    """
    Compute squared 2-norm loss and auxiliary residual norms.

    Returns
    -------
    loss : scalar
        ||r||_2^2 where r = BR(p) - p
    aux : (r_inf, r_l2)
        r_inf = ||r||_inf, r_l2 = ||r||_2
    """
    r = residual(static_params, dynamic_params, prices_i)
    loss = jnp.sum(r ** 2)
    r_inf = jnp.max(jnp.abs(r))
    r_l2 = jnp.sqrt(loss)
    return loss, (r_inf, r_l2)

loss_grad_with_aux = jit(
    value_and_grad(loss_with_aux, argnums=2, has_aux=True),
    static_argnames=("static_params",)
)


# ============================================================================
# Nash equilibrium solver
# ============================================================================

def find_equilibrium(
    static_params,
    dynamic_params,
    prices0_i: jnp.ndarray,
    quiet: bool = False,
):
    """Find Nash-in-Nash equilibrium prices via gradient-based fixed-point search.

    We seek prices p* satisfying the fixed-point condition BR(p*) = p*, where
    BR(·) is the vector of best-response prices. We minimize the squared
    residual L(p) = || BR(p) - p ||_2^2 using Adam with projection onto
    feasible price bounds.

    The solver uses the infinity norm as the convergence criterion:
      ||r||_∞ ≤ tol_inf

    A real-time Rich progress bar displays the current residual norms:
      ‖r‖₂ (l2_loss column) and ‖r‖∞ (linf_loss column) every `refresh_rate` steps.

    Parameters
    ----------
    static_params : FrozenDict
        Must contain:
          - optimizer_params: dict with Adam hyperparameters
              {learning_rate, b1, b2, eps}
          - max_iter       : int, maximum iterations
          - tol_inf        : float, infinity-norm stopping threshold
          - tol_l2         : float, 2-norm stopping threshold
          - refresh_rate   : int, UI update interval (in iterations)
          - n_stents       : int, problem dimension (used for shapes)
    dynamic_params : dict
        Model primitives (bounds, costs, revenues, demand parameters, grids).
    prices0_i : Array, shape (I,)
        Initial price vector (e.g., midpoint of bounds).
    quiet : bool, default False
        If True, suppresses the visual progress bar (uses a dummy no-op).

    Returns
    -------
    p_eq : Array, shape (I,)
        Estimated equilibrium prices.
    info : dict with keys
        - loss     : final squared residual (||r||_2^2)
        - n_iter   : iterations executed
        - history  : 1D array of loss values (per iteration)
        - r_inf    : final infinity norm ||r||_∞
        - r_l2     : final 2-norm ||r||_2
        - tol_inf  : stopping threshold used for infinity norm
        - tol_l2   : stopping threshold used for 2-norm

    Notes
    -----
    - Residual r(p) is computed in a single JIT pass together with gradients.
    - The displayed norms are derived directly from r, not back-propagated.
    - For fp32 arithmetic, default tolerances (1e-5, 1e-8) are conservative.
    - If convergence stalls near the bounds, consider relaxing tol_inf slightly.
    """
    optimizer_params = static_params["optimizer_params"]
    max_iter = static_params["max_iter"]
    refresh_rate = static_params["refresh_rate"]
    tol_inf = static_params["tol_inf"]
    tol_l2 = static_params["tol_l2"]

    
    # Get price bounds for projection
    lower_bounds = dynamic_params["lower_bounds_i"]
    upper_bounds = dynamic_params["upper_bounds_i"]
    
    p = prices0_i
    # Chain optimizer with bound projection
    opt = optax.chain(
        optax.adam(**optimizer_params),
        optax.clip_by_global_norm(1.0),  # Prevent exploding gradients
    )
    opt_state = opt.init(p)
    history = []

    # Choose progress implementation (real Rich progress or dummy no-op)
    progress_cm = (
        DummyProgress()
        if quiet
        else Progress(
            TextColumn("[bold blue]Finding equilibrium...[/bold blue]"),
            BarColumn(),
            TextColumn("[green]{task.completed}/{task.total}[/green] steps"),
            TextColumn("‖r‖₂={task.fields[l2_loss]:.2e}"),
            TextColumn("‖r‖∞={task.fields[linf_loss]:.2e}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
    )

    # Use the progress context manager so the bar renders
    with progress_cm as progress:
        task = progress.add_task(
            "solve",
            total=max_iter,
            l2_loss=float("inf"),
            linf_loss=float("inf"),
        )
        loss_val = float("inf")
        t = 0
        for t in range(max_iter):
            # Compute loss, residual norms, and gradient in one compiled pass
            (loss, (r_inf, r_l2)), grad_p = loss_grad_with_aux(static_params, dynamic_params, p)

            # Update prices using Adam
            updates, opt_state = opt.update(grad_p, opt_state)
            p = optax.apply_updates(p, updates)

            # Project prices back to valid bounds
            p = jnp.clip(p, lower_bounds, upper_bounds)

            history.append(float(loss))

            # Check for numerical issues (convert to Python float first)
            loss_val = float(loss)
            if jnp.isnan(loss_val) or jnp.isnan(p).any():
                print(f"NaN detected at step {t}")
                print("Prices:", p)
                print("Gradients:", grad_p)
                print("Loss:", loss_val)
                break

            # Update progress display
            if t % refresh_rate == 0:
                progress.update(task, completed=t + 1, l2_loss=float(r_l2), linf_loss=float(r_inf))

            # Convergence check: only r_inf must be within threshold
            if r_inf <= tol_inf:
                break

        # Final update to mark completion
        progress.update(task, completed=t + 1, l2_loss=float(r_l2), linf_loss=float(r_inf))

    return p, {
        "loss": loss_val,
        "n_iter": t + 1,
        "history": history,
        "r_inf": r_inf,
        "r_l2": r_l2,
        "tol_inf": tol_inf,
        "tol_l2": tol_l2,
    }
