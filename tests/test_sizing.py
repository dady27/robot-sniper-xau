import pytest

from sniper.sizing import net_payoff, solve


def test_net_payoff_includes_spread():
    assert net_payoff(2.5, 3.0, 0.30) == pytest.approx((7.5 - 0.3) / 3.3)


def test_no_edge_cannot_beat_balance_over_goal():
    # Jeu équitable : aucune mise ne dépasse solde / objectif (Dubins et Savage).
    r = solve(10, 100, 40, p=0.5, rr=1.0, sl=1.0, spread=0.0, account="cent", resolution=0.05)
    assert r.probability <= 0.10 + 1e-6


def test_more_edge_more_chance():
    lo = solve(10, 1000, 44, p=0.45, account="standard").probability
    hi = solve(10, 1000, 44, p=0.55, account="standard").probability
    assert hi > lo


def test_standard_account_bets_whole_lots():
    r = solve(10, 1000, 44, p=0.50, account="standard")
    assert r.lots >= 0.01
    assert r.risk == pytest.approx(round(r.lots / 0.01) * 3.3, abs=0.11)


def test_goal_reached_needs_no_trade():
    r = solve(1000, 1000, 10, p=0.5)
    assert r.probability == 1.0 and r.risk == 0.0
