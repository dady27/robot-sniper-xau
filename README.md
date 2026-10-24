# Robot Sniper XAU

Robot de trading sur l'or (XAU/USD) : sweep de liquidité, réintégration, CISD et FVG, avec une mise calculée pour maximiser la chance d'atteindre un objectif de capital (10 $ vers 1 000 $).

Ce dépôt contient d'abord le laboratoire Python (phases 0 et 1). Il sert à mesurer, sur des années d'historique, le taux de TP réel des setups avant d'écrire une seule ligne de l'Expert Advisor MQL5.

## La règle de base

| Élément | Règle (valeurs dans `sniper/config.py`) |
|---|---|
| Niveaux | haut et bas de l'Asie (00h-06h Londres), de la veille (jour de trading 18h-17h New York) et de la session de Londres |
| Sweep | la mèche dépasse le niveau d'au moins 0,1 ATR M5, sans s'en éloigner de plus de 0,5 ATR H1 |
| Réintégration | une clôture M5 revient de l'autre côté du niveau dans les 3 bougies M5 |
| CISD | une clôture dépasse l'ouverture de la série de bougies qui a fait l'extrême |
| FVG | un gap M1 d'au moins 0,15 ATR M1 laissé par le déplacement |
| Entrée | ordre limite dans le premier quart du FVG |
| SL | extrême moins max(2 × spread, 0,1 ATR M5), au moins 0,8 ATR M1, au plus 0,35 ATR H1 |
| TP | 2,5R, seulement si le premier niveau opposé est à 2,5R ou plus |
| Sorties | SL, TP, juge CRT H1 (clôture de l'heure de l'extrême repassée sous le niveau), ou 90 min après la killzone |
| Modèles | A : reversal de la killzone de Londres (07h-10h Londres). B : reversal de la killzone de New York (08h-11h New York). |

Les heures suivent Londres et New York, changement d'heure compris. Un seul trade par killzone. Quand une bougie M1 touche le SL et le TP, le backtest compte une perte.

## Structure

```
sniper/
  config.py      paramètres de la stratégie
  sessions.py    fenêtres horaires (Asie, killzones, jour de trading)
  data.py        ticks Dukascopy (.bi5), export MT5, bougies, ATR connu à la clôture
  signals.py     détection sweep → réintégration → CISD → FVG → ordre
  simulate.py    remplissage de l'ordre limite et sortie (bid/ask séparés)
  backtest.py    niveaux du jour et rejeu des modèles A et B
  report.py      synthèse par modèle, intervalle de confiance de Wilson
  sizing.py      mise optimale vers un objectif (programmation dynamique)
  synthetic.py   marché aléatoire de contrôle
scripts/         téléchargement, backtest, contrôle sur hasard, calcul de mise
tests/           tests sur bougies construites à la main
```

## Lancer

```bash
pip install -r requirements.txt
python -m pytest -q
```

Données Dukascopy (2015 à aujourd'hui, environ 12 ans de ticks) :

```bash
python scripts/download_dukascopy.py --start 2015-01-01 --end 2026-10-01
python scripts/run_backtest.py --m1-dir data/m1
```

Ou un export MT5 d'Exness. Dans MetaTrader 5 : Affichage → Symboles → onglet Barres, choisir le symbole or du compte (souvent `XAUUSDm` sur un compte Standard), M1, la période la plus longue possible, Demander, puis Exporter les barres.

```bash
python scripts/run_backtest.py --mt5 XAUUSDm_M1.csv --server-utc-offset 0
```

Le backtest écrit `out/setups.csv` (un setup par ligne, avec ses mesures) et `out/synthese.csv` (taux de TP, intervalle de confiance, espérance en R, trades par jour).

Contrôle anti-triche : sur une marche aléatoire, le taux de TP doit rester vers 28 % (1 / 3,5).

```bash
python scripts/controle_hasard.py --jours 1500
```

Mise du prochain trade (exemple : 10 $, 44 trades restants, 50 % de TP) :

```bash
python scripts/mise.py --solde 10 --trades 44 --p 0.50
```

## Exness, compte Standard

- Lot minimum 0,01. Avec un SL de 3 $ de prix et un spread de 0,30 $, un lot de 0,01 risque 3,30 $.
- Contrat supposé de 100 onces par lot : à vérifier dans la spécification du symbole sous MT5.
- Heure serveur supposée UTC+0 ; si les bougies sont décalées, corriger `--server-utc-offset`.
- Stop out à 0 % et effet de levier élevé : un compte de 10 $ peut ouvrir 0,01 lot.
- Avec p = 50 % sur 44 trades, la chance d'atteindre 1 000 $ est d'environ 50 % en Standard contre 59 % en Standard Cent, car la mise ne peut pas descendre sous 3,30 $.

## Feuille de route

| Phase | Contenu | Critère pour passer à la suite |
|---|---|---|
| 0 | Données tick 2015-2026 | historique complet, trous identifiés |
| 1 | Backtest des règles (ce dépôt) | au moins 1 000 setups, taux de TP hors échantillon ≥ 45 % avec ≥ 1,5 trade par jour |
| 2 | Méta-modèle (étiquettes triple barrière, walk-forward purgé) | Deflated Sharpe ≥ 0,95, probabilité de surapprentissage < 20 % |
| 3 | Expert Advisor MQL5 (modèle ONNX, mise optimale) | MT5 et Python d'accord à ± 3 points de taux de TP |
| 4 | Démo Exness pendant 4 semaines | résultats dans l'intervalle du backtest |
| 5 | Compte réel de 10 $ sur 30 jours | |

Les modèles C (continuation) et D (sweep de nombre rond) et les filtres de biais (calendrier économique, DXY, taux réels) viennent après la première mesure des modèles A et B.
