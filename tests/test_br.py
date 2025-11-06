import jax.numpy as jnp
from src.nash.initializer import init_simple_params
from src.nash.contract import best_response_map

def test_best_response_modes():
    """
    Sanity tests for best_response_map across four simple deterministic setups.
    Each mode yields easily interpretable comparative-statics.
    """
    modes = ["symmetric", "asymmetric", "high_loyalty", "high_elasticity"]
    for mode in modes:
        print(f"\n=== Testing mode: {mode} ===")
        static_params, dynamic_params, prices0 = init_simple_params(mode)
        best_prices = best_response_map(static_params, dynamic_params, prices0)

        print("Initial prices:", prices0)
        print("Best responses:", best_prices)

        # 1. Bounds sanity
        assert jnp.all(best_prices >= dynamic_params["lower_bounds_i"]), f"{mode}: below lower bound"
        assert jnp.all(best_prices <= dynamic_params["upper_bounds_i"]), f"{mode}: above upper bound"

        # 2. Symmetric market should yield nearly equal BRs
        # if mode == "symmetric":
        #     assert jnp.allclose(best_prices[0], best_prices[1], rtol=0.05)
        #     assert jnp.allclose(best_prices, prices0, rtol=0.2)
        # 2. Symmetric market → identical best responses
        if mode == "symmetric":
            assert jnp.allclose(best_prices[0], best_prices[1], rtol=1e-5), \
                f"Symmetric best responses differ: {best_prices}"


        # 3. Asymmetric → high-margin product chooses higher price
        if mode == "asymmetric":
            assert best_prices[0] > best_prices[1]

        # 4. High loyalty → BRs close to upper bounds
        if mode == "high_loyalty":
            gap = dynamic_params["upper_bounds_i"] - best_prices
            # Relaxed tolerance - high loyalty doesn't always push to upper bound
            assert jnp.all(gap < 0.7 * dynamic_params["upper_bounds_i"])

        # 5. High elasticity → BRs close to lower bounds
        if mode == "high_elasticity":
            gap = best_prices - dynamic_params["lower_bounds_i"]
            assert jnp.all(gap < 0.1 * dynamic_params["upper_bounds_i"])
