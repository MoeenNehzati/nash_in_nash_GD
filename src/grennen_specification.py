import jax
from jax import numpy as jnp
from flax.core import FrozenDict
from numpy import random as rng
# ===============================================================
# --- optimization params ---
# ===============================================================
lr=1e-2
max_iter=5000
tol=1e-6
refresh_rate=10

# ===============================================================
# --- dimensionality of the problem ---
# ===============================================================
n_stents = 8      # total products
n_bms = 6
n_des = 2
n_types = 2       # BMS and DES
n_loyalty = n_stents

# --- Indicators for stent type ---
is_bms = jnp.arange(n_stents) < n_bms # (I,) boolean
is_des = ~is_bms                    # (I,) boolean
hastype_i_t = jnp.stack([is_des, is_bms], axis=1)                # (I, T)

# ===============================================================
# --- Preference and nesting parameters (Grennan 2013) ---
# ===============================================================
σ_des, σ_bms = 0.29, 0.29
σ_t = jnp.array([σ_des, σ_bms])            # (T,)
σ_stent = 0.38                             # top-level (h)
θ_p = -0.27                                # price disutility (utils / $1000)

λ_des, λ_bms = 2.0, 3.3
λ_t = jnp.array([λ_des, λ_bms])            # (T,)
λ_stent = 3.3                              # top-level loyalty (h)

# ===============================================================
# --- Economic primitives ---
# ===============================================================
revenue_bms, revenue_des = 15.65698, 17.59703   # expressed in thousands of dollars
marginal_cost_bms, marginal_cost_des = .034, 1.103   # expressed in thousands of dollars

# Marginal cost and revenue per product (I,)
marginal_costs_i = marginal_cost_bms * is_bms + marginal_cost_des * is_des
hospital_revenues_i = revenue_bms * is_bms + revenue_des * is_des

# Bargaining parameters (mean=0.33, sd=0.07, clipped [0.01,0.99])
bp_i = jnp.clip(jnp.array(rng.normal(0.33, 0.07, n_stents)), 0.01, 0.99)  # (I,)

# Product fixed effects (taste shocks)
θ_i = jnp.array(rng.uniform(-0.1, 0.1, n_stents))  # (I,) # in utilities

# ===============================================================
# --- Loyalty structure ---
# ===============================================================
bms_share = 0.91
des_share = 1.0 - bms_share

# Loyalty weights φ_l (each l loyal to one stent i)
φ_l = bms_share * is_bms / is_bms.sum() + des_share * is_des / is_des.sum()  # (L=I,)

# Loyalty intensity λ_i (by product type)
λ_i = λ_bms * is_bms + λ_des * is_des                                        # (I,)
D_loyalty_i_l = jnp.diag(λ_i)                                                # (I, L)

# ===============================================================
# --- Price domain ---
# ===============================================================
lower_bounds_i = marginal_costs_i                                            # (I,)
upper_bounds_i = jnp.full(n_stents, max(revenue_bms, revenue_des) * 1.1)     # (I,)

n_grid = 1000

price_grids_i = jnp.array([
    jnp.linspace(lower_bounds_i[i], upper_bounds_i[i], n_grid)
    for i in range(n_stents)
])                                                                           # (I, n_grid)
    
# Initial guess for prices (midpoint of bounds)
prices0_i = (lower_bounds_i + upper_bounds_i) / 2                            # (I,)

# ===============================================================
# --- Nest-scaling denominators (used in D_i_l construction) ---
# α_i = (1 − σ_stent)(1 − σ_t(i))
# ===============================================================
nest_scale_i = (1-σ_stent) * jnp.einsum('it,t->i', hastype_i_t, (1-σ_t))      # (I,)
D_denom_i_l = jnp.repeat(nest_scale_i[:, None], n_stents, axis=1)      # (I, L)


# ===============================================================
# ---Putting things together---
# ===============================================================
meta_params = dict(
    lr=1e-2,
    max_iter=5000,
    tol=1e-6,
    refresh_rate=10,
)

static_params = dict(
    n_stents = n_stents,
    n_bms = n_bms,
    n_des = n_des,
    n_types = n_types,
    n_loyalty = n_loyalty,    
)

static_params.update(meta_params)
static_params = FrozenDict(static_params)

dynamic_params = dict(
    # indicators
    is_bms=is_bms,                 # (I,)
    is_des=is_des,                 # (I,)
    hastype_i_t=hastype_i_t,       # (I, T)

    # dissimilarity / loyalty
    σ_stent=σ_stent,
    σ_t=σ_t,
    λ_t=λ_t,
    λ_stent=λ_stent,

    # economic primitives
    θ_p=θ_p,
    θ_i=θ_i,
    bp_i=bp_i,
    revenue_bms=revenue_bms,
    revenue_des=revenue_des,
    marginal_costs_i=marginal_costs_i,
    hospital_revenues_i=hospital_revenues_i,

    # loyalty structure
    φ_l=φ_l,                       # (L,)
    λ_i=λ_i,                       # (I,)
    nest_scale_i=nest_scale_i,   # (I,)
    D_loyalty_i_l=D_loyalty_i_l,   # (I, L)
    D_denom_i_l=D_denom_i_l,       # (I, L)

    # price domain
    lower_bounds_i=lower_bounds_i,   # (I,)
    upper_bounds_i=upper_bounds_i,   # (I,)
    price_grids_i=price_grids_i,     # (I, n_grid)
    prices_i=prices0_i,                # (I,) - current/initial prices
)