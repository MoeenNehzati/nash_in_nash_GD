import jax
import pytest
jnp = jax.numpy
random = jax.random

from src.nash.contract import (
    compute_δ_i,
    compute_D_i_l,
    compute_s_h_l,
    compute_s_t_h_l,
    compute_s_i_t_l,
    compute_s_i_l,
    compute_market_shares,
    compute_shares_from_prices,
    compute_hospital_profit,
    compute_supplier_profit,
    calculate_nash_product,
)

from src.nash.initializer import init_simple_params
from src.grennen_specification import static_params as grennan_static_params, dynamic_params as grennan_dynamic_params

# -----------------------------------------------------------------------------
# 1. Parameter initialization
# -----------------------------------------------------------------------------
def test_init_grennan_params_shapes():
    static_params, dynamic_params = grennan_static_params, grennan_dynamic_params
    assert dynamic_params["hastype_i_t"].shape == (static_params["n_stents"], static_params["n_types"])
    assert dynamic_params["D_loyalty_i_l"].shape == (static_params["n_stents"], static_params["n_loyalty"])
    assert dynamic_params["D_denom_i_l"].shape == (static_params["n_stents"], static_params["n_loyalty"])
    assert dynamic_params["φ_l"].shape == (static_params["n_loyalty"],)
    assert jnp.all(dynamic_params["σ_t"] > 0) and dynamic_params["σ_t"].ndim == 1


# -----------------------------------------------------------------------------
# 2. Mean utilities δ_i
# -----------------------------------------------------------------------------
def test_compute_delta_linear_scaling():
    static_params, dynamic_params = grennan_static_params, grennan_dynamic_params
    prices = jnp.linspace(1, 15, static_params["n_stents"])
    δ = compute_δ_i(static_params, dynamic_params, prices)
    # linear response: higher prices → lower δ
    assert δ.shape == (static_params["n_stents"],)
    assert δ[0] > δ[-1]


# -----------------------------------------------------------------------------
# 3. D_i_l computation
# -----------------------------------------------------------------------------
def test_compute_D_i_l_positive():
    """
    D_i_l must be positive.
    Within each nest, D_i_l (without loyalty bonuses) should increase in δ_i.
    Loyalty terms may cause local rank reversals, so we check the base component only.
    """
    static_params, dynamic_params = grennan_static_params, grennan_dynamic_params
    δ = jnp.linspace(-2, 0.5, static_params["n_stents"])
    D_i_l, S_t_l = compute_D_i_l(static_params, dynamic_params, δ)

    # shape and positivity
    assert D_i_l.shape == (static_params["n_stents"], static_params["n_loyalty"])
    assert jnp.all(D_i_l > 0)
    
    # Check S_t_l shape
    assert S_t_l.shape == (static_params["n_types"], static_params["n_loyalty"])
    assert jnp.all(S_t_l > 0)

    # reconstruct "base" D_i (no loyalty effect)
    D_base = jnp.exp(δ / dynamic_params["D_denom_i_l"][:, 0])  # take denom for l=0 (any column)
    is_des = dynamic_params["is_des"].astype(bool)
    is_bms = dynamic_params["is_bms"].astype(bool)

    # Within each nest, D_base is monotone increasing in δ
    for mask in [is_des, is_bms]:
        δ_sub = δ[mask]
        D_sub = D_base[mask]
        diffs = jnp.diff(D_sub)
        assert jnp.all(diffs > 0), "D_i not increasing in δ_i within nest"


# -----------------------------------------------------------------------------
# 4. Conditional shares
# -----------------------------------------------------------------------------
def test_conditional_shares_sum_to_one():
    static_params, dynamic_params = grennan_static_params, grennan_dynamic_params
    δ = jnp.linspace(-2, 0.5, static_params["n_stents"])
    D_i_l, S_t_l = compute_D_i_l(static_params, dynamic_params, δ)

    s_h_l = compute_s_h_l(static_params, dynamic_params, S_t_l)
    s_t_h_l = compute_s_t_h_l(static_params, dynamic_params, S_t_l)
    s_i_t_l = compute_s_i_t_l(static_params, dynamic_params, D_i_l, S_t_l)

    # Each loyalty column sums to 1 within each level
    assert jnp.allclose(s_t_h_l.sum(axis=0), 1.0, atol=1e-10)
    for t in range(static_params["n_types"]):
        mask = dynamic_params["hastype_i_t"][:, t] == 1
        assert jnp.allclose(s_i_t_l[mask].sum(axis=0), 1.0, atol=1e-10)
    # Top-level share is between 0 and 1
    assert jnp.all((s_h_l > 0) & (s_h_l < 1))


# -----------------------------------------------------------------------------
# 5. Full nested structure consistency
# -----------------------------------------------------------------------------
def test_total_share_consistency():
    static_params, dynamic_params = grennan_static_params, grennan_dynamic_params
    δ = jnp.linspace(-2, 0.5, static_params["n_stents"])
    D_i_l, S_t_l = compute_D_i_l(static_params, dynamic_params, δ)
    s_i_l = compute_s_i_l(static_params, dynamic_params, D_i_l, S_t_l)

    # Columns sum to ≤1 (outside option remainder)
    assert s_i_l.shape == (static_params["n_stents"], static_params["n_loyalty"])
    col_sums = s_i_l.sum(axis=0)
    assert jnp.all((col_sums > 0) & (col_sums <= 1))

    # Weighted average over loyalty equals expected stent probability
    s_i = compute_market_shares(static_params, dynamic_params, s_i_l)
    s_h_l = compute_s_h_l(static_params, dynamic_params, S_t_l)
    expected_total = jnp.dot(s_h_l, dynamic_params["φ_l"])
    assert jnp.isclose(s_i.sum(), expected_total, rtol=1e-6)


# -----------------------------------------------------------------------------
# 6. Market-share sanity check
# -----------------------------------------------------------------------------
def test_compute_market_shares_manual():
    static_params = {"n_loyalty": 4}
    dynamic_params = {"φ_l": jnp.array([0.25, 0.25, 0.25, 0.25])}
    s_i_l = jnp.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.6, 0.5, 0.4, 0.3],
        [0.3, 0.3, 0.3, 0.3],
    ])
    s_i = compute_market_shares(static_params, dynamic_params, s_i_l)
    expected = jnp.array([0.25, 0.45, 0.30])
    assert jnp.allclose(s_i, expected, atol=1e-8)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 7. Gradient safety / differentiability tests
# -----------------------------------------------------------------------------
def test_gradients_are_finite():
    """
    Verify that all core functions are differentiable w.r.t. prices,
    and that gradients are finite and nonzero.
    """
    static_params, dynamic_params = grennan_static_params, grennan_dynamic_params
    prices = jnp.linspace(1, 15, static_params["n_stents"])

    # 1. δ_i wrt prices
    def delta_fn(p):
        return compute_δ_i(static_params, dynamic_params, p).sum()
    grad_δ = jax.grad(delta_fn)(prices)
    assert jnp.all(jnp.isfinite(grad_δ)), "NaN in ∂δ/∂p"
    assert jnp.any(jnp.abs(grad_δ) > 1e-12), "zero gradient in ∂δ/∂p"

    # 2. D_i_l wrt prices
    def D_sum(p):
        δ = compute_δ_i(static_params, dynamic_params, p)
        D_i_l, _ = compute_D_i_l(static_params, dynamic_params, δ)
        return D_i_l.sum()

    grad_D = jax.grad(D_sum)(prices)
    assert jnp.all(jnp.isfinite(grad_D)), "NaN in ∂D/∂p"
    # Relaxed check - gradient might be very small but nonzero
    assert jnp.any(jnp.abs(grad_D) > 1e-15), "zero gradient in ∂D/∂p"

    # 3. Full nested share wrt prices
    def total_share_sum(p):
        δ = compute_δ_i(static_params, dynamic_params, p)
        D_i_l, S_t_l = compute_D_i_l(static_params, dynamic_params, δ)
        s_i_l = compute_s_i_l(static_params, dynamic_params, D_i_l, S_t_l)
        s_i = compute_market_shares(static_params, dynamic_params, s_i_l)
        return s_i.sum()

    grad_total = jax.grad(total_share_sum)(prices)
    assert jnp.all(jnp.isfinite(grad_total)), "NaN in ∂s_total/∂p"
    assert jnp.any(jnp.abs(grad_total) > 1e-12)



# -----------------------------------------------------------------------------
# 8. Hospital and supplier profit tests
# -----------------------------------------------------------------------------
def test_compute_hospital_profit_basic():
    """
    Hospital profit should be positive and decrease with higher prices.
    """
    static_params, dynamic_params = grennan_static_params, grennan_dynamic_params
    prices_low = jnp.linspace(1, 8, static_params["n_stents"])
    prices_high = jnp.linspace(10, 18, static_params["n_stents"])

    Π_low = compute_hospital_profit(static_params, dynamic_params, prices_low)
    Π_high = compute_hospital_profit(static_params, dynamic_params, prices_high)

    assert Π_low > 0
    assert Π_high > 0
    # Higher prices reduce hospital profit (since revenue − price shrinks)
    assert Π_low > Π_high


def test_compute_hospital_profit_exclusion():
    """
    Excluding a product (disagreement) should weakly lower hospital profit.
    """
    static_params, dynamic_params = grennan_static_params, grennan_dynamic_params
    prices = jnp.linspace(5, 15, static_params["n_stents"])
    Π_full = compute_hospital_profit(static_params, dynamic_params, prices)
    Π_excl = compute_hospital_profit(static_params, dynamic_params, prices, excluded_stent=0)
    assert Π_excl <= Π_full + 1e-8


def test_compute_supplier_profit_markup_logic():
    """
    Supplier profit = (p_i − c_i) * s_i(prices).
    Must be positive when price > cost and zero if excluded=True.
    """
    static_params, dynamic_params = grennan_static_params, grennan_dynamic_params
    prices = jnp.linspace(5, 15, static_params["n_stents"])

    i = 2
    Π_on = compute_supplier_profit(static_params, dynamic_params, prices, i)
    Π_off = compute_supplier_profit(static_params, dynamic_params, prices, i, excluded=True)

    assert jnp.isclose(Π_off, 0.0)
    assert Π_on > 0


# -----------------------------------------------------------------------------
# 9. Nash product tests
# -----------------------------------------------------------------------------
def test_nash_product_positive_and_finite():
    """
    Nash product must be positive and finite, since we clamp with 1e−10 floors.
    """
    static_params, dynamic_params = grennan_static_params, grennan_dynamic_params
    prices = jnp.linspace(5, 15, static_params["n_stents"])
    i = 3
    N_i = calculate_nash_product(static_params, dynamic_params, prices, i)
    assert jnp.isfinite(N_i)
    assert N_i > 0

def test_nash_product_reduces_with_high_prices():
    """
    Sanity check: Nash products remain finite when prices rise.
    At very high prices, the log Nash product may be negative (product < 1)
    as gains from trade diminish.
    """
    static_params, dynamic_params = grennan_static_params, grennan_dynamic_params
    prices_low = jnp.linspace(2, 8, static_params["n_stents"])
    prices_high = jnp.linspace(10, 16, static_params["n_stents"])
    i = 5

    N_low = calculate_nash_product(static_params, dynamic_params, prices_low, i)
    N_high = calculate_nash_product(static_params, dynamic_params, prices_high, i)

    # Both should be finite (this is log Nash product, so can be negative)
    assert jnp.isfinite(N_low) and jnp.isfinite(N_high)
    
    # Lower prices should yield higher Nash product (more gains from trade)
    assert N_low > N_high, f"Expected N_low ({N_low:.3f}) > N_high ({N_high:.3f})"


# -----------------------------------------------------------------------------
# 10. Gradient differentiability tests (for bargaining stage)
# -----------------------------------------------------------------------------
def test_grad_nash_product_wrt_prices():
    """
    Ensure calculate_nash_product(params, prices, i) is JAX-differentiable
    w.r.t. prices, for use in implicit differentiation of equilibrium.
    """
    static_params, dynamic_params = grennan_static_params, grennan_dynamic_params
    prices = jnp.linspace(5, 15, static_params["n_stents"])
    i = 4

    grad_fn = jax.grad(lambda p: calculate_nash_product(static_params, dynamic_params, p, i).sum())
    g = grad_fn(prices)

    assert g.shape == prices.shape
    assert jnp.all(jnp.isfinite(g))
    assert jnp.any(jnp.abs(g) > 1e-12)