import jax
import numpy as np
jnp = jax.numpy
from flax.core import FrozenDict

meta_params = dict(
        lr=1e-2,
        max_iter=5000,
        tol=1e-6,
        refresh_rate=1
)

#todo add the change this step to the bar
def init_simple_params(mode="symmetric"):
    """
    Initialize small deterministic 2-product model for best-response tests.
    """
    n_stents = 2
    lower_bounds_i = jnp.array([.4, .4])
    upper_bounds_i = jnp.array([1.2, 1.2])

    if mode == "symmetric":
        θ_i = jnp.array([0.0, 0.0])
        λ_i = jnp.array([4.0, 4.0])
        σ_t = jnp.array([0.25, 0.25])
        σ_stent = 0.25
        marginal_costs_i = jnp.array([.2, .2])
        hospital_revenues_i = jnp.array([1.0, 1.0])
        bp_i = jnp.array([0.33, 0.33])

    elif mode == "asymmetric":
        θ_i = jnp.array([0.0, 0.0])
        λ_i = jnp.array([2.0, 2.0])
        σ_t = jnp.array([0.3, 0.3])
        σ_stent = 0.4
        marginal_costs_i = jnp.array([.2, .2])
        hospital_revenues_i = jnp.array([1.2, .9])  # product 0 more profitable
        bp_i = jnp.array([0.33, 0.33])

    elif mode == "high_loyalty":
        θ_i = jnp.array([0.0, 0.0])
        λ_i = jnp.array([8.0, 8.0])
        σ_t = jnp.array([0.1, 0.1])    # within-type very correlated
        σ_stent = 0.1                  # weak substitution across types
        marginal_costs_i = jnp.array([.2, .2])
        hospital_revenues_i = jnp.array([1.0, 1.0])
        bp_i = jnp.array([0.33, 0.33])

    elif mode == "high_elasticity":
        θ_i = jnp.array([0.0, 0.0])
        λ_i = jnp.array([0.5, 0.5])
        σ_t = jnp.array([0.8, 0.8])
        σ_stent = 0.8
        marginal_costs_i = jnp.array([.2, .2])
        hospital_revenues_i = jnp.array([1.0, 1.0])
        bp_i = jnp.array([0.33, 0.33])

    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    static_params =  dict(
        n_stents = n_stents,
        n_grid = 1000,
    )
    static_params.update(meta_params)
    
    # Precompute price grids
    n_grid = static_params["n_grid"]
    price_grids_i = jnp.array([
        jnp.linspace(lower_bounds_i[i], upper_bounds_i[i], n_grid)
        for i in range(n_stents)
    ])
    
    # Initial guess for prices (midpoint of bounds)
    prices_i = (lower_bounds_i + upper_bounds_i) / 2
    
    dynamic_params = dict(
        lower_bounds_i=lower_bounds_i,
        upper_bounds_i=upper_bounds_i,
        price_grids_i=price_grids_i,
        prices_i=prices_i,
        marginal_costs_i=marginal_costs_i,
        hospital_revenues_i=hospital_revenues_i,
        θ_i=θ_i,
        λ_i=λ_i,
        σ_t=σ_t,
        σ_stent=σ_stent,
        bp_i=bp_i,
        θ_p=-0.3,
        D_denom_i_l=jnp.ones((n_stents, n_stents)),
        D_loyalty_i_l=jnp.eye(n_stents) * λ_i,
        φ_l=jnp.array([0.5, 0.5]),
        hastype_i_t=jnp.eye(n_stents),
    )
    return FrozenDict(static_params), dynamic_params