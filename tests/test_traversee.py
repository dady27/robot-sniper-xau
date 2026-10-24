from sniper.events import summarize_events
from sniper.synthetic import random_walk_m1
from sniper.traversee import run_cross


def test_no_edge_on_random_walk():
    # Garde-fou : sur une marche aléatoire, traverser un niveau n'apporte rien.
    # Le TP à 2,5R reste vers 1 / 3,5 (moins le spread) et l'écart au contrôle reste du bruit.
    ev = run_cross(random_walk_m1("2025-01-06", 300, seed=1))
    s = summarize_events(ev).set_index("cible_r")
    assert len(ev) >= 120
    assert set(ev["famille"]) == {"ronds", "veille", "semaine"}
    assert not ev.duplicated(["time", "side"]).any()
    assert s.loc["2.5", "taux_tp"] < 1 / 3.5 + 0.03
    assert ev["r2.5"].mean() < 0.1
    assert abs(s.loc["2.5", "t"]) < 2.5
