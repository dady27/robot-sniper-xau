"""Paramètres de la stratégie Sweep · Reclaim · CISD.

Toutes les distances sont exprimées en multiples d'ATR ou en dollars de prix de l'or,
jamais en pips (la définition du pip varie selon les brokers).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyParams:
    # Sweep
    sweep_min_atr_m5: float = 0.10      # dépassement minimal du niveau, en ATR(14) M5
    max_excursion_atr_h1: float = 0.50  # au-delà, c'est une vraie expansion : setup annulé
    reclaim_max_m5_bars: int = 3        # clôture M5 de retour dans le range, au plus 3 bougies M5 après l'extrême
    trigger_max_minutes: int = 30       # CISD + réintégration au plus 30 min après l'extrême

    # FVG et entrée
    fvg_min_atr_m1: float = 0.15
    entry_depth: float = 0.25           # ordre limite dans le premier quart du FVG

    # Stop-loss
    sl_buffer_atr_m5: float = 0.10
    sl_buffer_spread_mult: float = 2.0
    sl_min_atr_m1: float = 0.80
    sl_max_atr_h1: float = 0.35

    # Objectif
    rr: float = 2.5                     # TP unique à 2,5R
    room_min_r: float = 2.5             # premier obstacle opposé à au moins 2,5R

    # Ordre limite
    order_valid_m1_bars: int = 20
    cancel_at_r: float = 2.5            # annulé si le prix atteint l'objectif avant le remplissage

    # Range asiatique exploitable (largeur en ATR(14) H1)
    asia_width_min_atr_h1: float = 0.4
    asia_width_max_atr_h1: float = 2.0

    # Sortie
    use_crt_h1_exit: bool = True        # sortie si la H1 du sweep clôture hors du range
    exit_after_window_minutes: int = 90 # fermeture forcée 90 min après la fin de la killzone
    sl_slippage: float = 0.10           # glissement ajouté à chaque SL, en dollars

    # Volatilité ATR
    atr_period: int = 14


DEFAULT_PARAMS = StrategyParams()
