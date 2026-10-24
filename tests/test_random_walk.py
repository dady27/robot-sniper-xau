from sniper.backtest import run
from sniper.synthetic import random_walk_m1


def test_no_edge_on_random_walk():
    # Garde-fou contre la lecture du futur : sans avantage, le TP à 2,5R reste vers 28 %.
    setups = run(random_walk_m1("2025-01-06", 400, seed=1))
    filled = setups[setups["status"] == "filled"]
    assert len(filled) >= 20
    assert (filled["exit_reason"] == "tp").mean() < 0.45
    assert filled["r"].mean() < 0.5
