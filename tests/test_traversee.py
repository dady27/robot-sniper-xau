import numpy as np

from sniper.events import summarize_events
from sniper.synthetic import random_walk_m1
from sniper.traversee import CrossParams, run_cross


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


def test_step_100_market_entry():
    # Multiples de 100 $ et entrée au marché : achat à l'ask, vente au bid, à la clôture de la minute du contact.
    m1 = random_walk_m1("2025-01-06", 300, seed=1)
    ev = run_cross(m1, CrossParams(families=("ronds",), step=100.0, stop_atr_h1=1.0, entry="close"))
    assert len(ev) >= 30
    assert (ev["level"] % 100 == 0).all()
    bar = m1.loc[ev["time"]]
    expected = np.where(ev["side"] == "long", bar["close"] + bar["spread"], bar["close"])
    assert np.allclose(ev["entry"], expected)
