import numpy as np

from sniper.events import follow, outcome_r, run_events, summarize_events
from sniper.synthetic import random_walk_m1


def test_follow_counts_stop_first_on_ambiguous_bar():
    # Achat à 100, stop à 99 : la bougie 1 touche +3R et le stop, on compte le stop.
    h = np.array([100.0, 103.0, 104.0])
    l = np.array([100.0, 98.9, 103.0])
    c = np.array([100.0, 99.0, 103.5])
    mfe, stopped, r_end, _ = follow(h, l, c, np.zeros(3), 1, 3, 100.0, 99.0, {})
    assert stopped and mfe == 0.0 and r_end == -1.0


def test_follow_exit_cost_and_levels():
    # Vente vue en prix retournés : la sortie à l'ask coûte le spread.
    h = np.array([100.0, 101.5, 102.4])
    l = np.array([100.0, 100.2, 101.0])
    c = np.array([100.0, 101.0, 102.0])
    mfe, stopped, r_end, hit = follow(h, l, c, np.full(3, 0.4), 1, 3, 100.0, 99.0, {"opp": 102.0})
    assert not stopped
    assert np.isclose(mfe, 2.0) and np.isclose(r_end, 1.6) and hit["opp"]


def test_outcome_r():
    assert outcome_r(2.6, True, -1.0, 2.5) == 2.5    # TP touché avant le stop
    assert outcome_r(1.2, True, -1.0, 2.5) == -1.0   # stoppé avant le TP
    assert outcome_r(1.2, False, 0.7, 2.5) == 0.7    # sortie de temps


def test_no_edge_on_random_walk():
    # Garde-fou : sur une marche aléatoire, l'événement ne bat pas le contrôle neutre
    # et le TP à 2,5R reste sous 1 / 3,5 (le spread coûte un peu).
    ev = run_events(random_walk_m1("2025-01-06", 300, seed=1))
    s = summarize_events(ev).set_index("cible_r")
    assert len(ev) >= 150
    assert s.loc["2.5", "taux_tp"] < 1 / 3.5 + 0.03
    assert abs(s.loc["2.5", "t"]) < 2.5
    assert s.loc["2.5", "esperance_r"] < 0.0
