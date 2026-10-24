import pandas as pd
import pytest

from conftest import LONG_SETUP, T0, make_bars, mirror
from sniper.signals import detect
from sniper.simulate import simulate

W_START = T0 + pd.Timedelta(minutes=10)
W_END = T0 + pd.Timedelta(minutes=190)


def test_long_setup_detected_with_expected_order(params, const_atr):
    bars = make_bars(LONG_SETUP)
    o = detect(bars, "long", "AL", 4000.0, W_START, W_END, *const_atr, params, obstacles=[4012.0])
    assert o is not None
    assert o.extreme == pytest.approx(3998.5)
    assert o.trigger_time == T0 + pd.Timedelta(minutes=20)        # clôture de la bougie 19
    assert o.fvg_proximal == pytest.approx(4000.5)
    assert o.fvg_distal == pytest.approx(3999.6)
    assert o.entry == pytest.approx(4000.275)                     # premier quart du FVG
    assert o.sl == pytest.approx(3998.5 - 0.6)                    # tampon = 2 x spread
    assert o.risk == pytest.approx(4000.275 - 3997.9)
    assert o.tp == pytest.approx(4000.275 + 2.5 * o.risk)


def test_long_trade_hits_tp(params, const_atr):
    bars = make_bars(LONG_SETUP)
    o = detect(bars, "long", "AL", 4000.0, W_START, W_END, *const_atr, params, obstacles=[4012.0])
    res = simulate(bars, o, W_END, params)
    assert res.status == "filled"
    assert res.fill_time == T0 + pd.Timedelta(minutes=21)
    assert res.exit_reason == "tp"
    assert res.r == pytest.approx(2.5)


def test_short_is_exact_mirror(params, const_atr):
    bars = mirror(make_bars(LONG_SETUP))
    o = detect(bars, "short", "AH", 4000.0, W_START, W_END, *const_atr, params, obstacles=[3988.0])
    assert o is not None
    assert o.entry == pytest.approx(8000 - 4000.275)
    assert o.sl == pytest.approx(8000 - 3997.9)
    res = simulate(bars, o, W_END, params)
    assert res.exit_reason == "tp"
    assert res.r == pytest.approx(2.5)


def test_no_trade_without_reclaim(params, const_atr):
    rows = [list(r) for r in LONG_SETUP]
    for i in range(15, len(rows)):  # le prix reste sous le niveau : pas de réintégration
        rows[i] = [x - 3.0 for x in rows[i]]
    bars = make_bars([tuple(r) for r in rows])
    assert detect(bars, "long", "AL", 4000.0, W_START, W_END, *const_atr, params) is None


def test_room_filter_rejects_close_obstacle(params, const_atr):
    bars = make_bars(LONG_SETUP)
    # obstacle à moins de 2,5R au-dessus de l'entrée
    assert detect(bars, "long", "AL", 4000.0, W_START, W_END, *const_atr, params, obstacles=[4003.0]) is None


def test_deep_sweep_is_a_real_breakout(params, const_atr):
    a1, a5, _ = const_atr
    bars = make_bars(LONG_SETUP)
    small_h1 = lambda t: 2.0  # excès max = 1,0 : le sweep de 1,5 sous le niveau l'annule
    assert detect(bars, "long", "AL", 4000.0, W_START, W_END, a1, a5, small_h1, params) is None


def test_same_bar_sl_and_tp_counts_as_loss(params, const_atr):
    rows = list(LONG_SETUP)
    rows[22] = (4000.6, 4007.0, 3997.0, 4003.0)  # touche TP et SL dans la même minute
    bars = make_bars(rows)
    o = detect(bars, "long", "AL", 4000.0, W_START, W_END, *const_atr, params, obstacles=[4012.0])
    res = simulate(bars, o, W_END, params)
    assert res.exit_reason == "sl"
    assert res.r == pytest.approx(-1 - params.sl_slippage / o.risk)


def test_order_cancelled_if_target_reached_before_fill(params, const_atr):
    rows = list(LONG_SETUP)
    rows[20] = (4004.0, 4007.0, 4003.9, 4006.8)   # part à l'objectif sans revenir
    rows[21] = (4006.8, 4007.2, 4006.5, 4007.0)
    bars = make_bars(rows)
    o = detect(bars, "long", "AL", 4000.0, W_START, W_END, *const_atr, params, obstacles=[4012.0])
    assert simulate(bars, o, W_END, params).status == "cancelled_runaway"
