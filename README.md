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
  events.py      étude d'événements (sweep de l'Asie, confirmation, contrôle neutre)
  traversee.py   traversée des niveaux de stops (multiples de 50 $, veille, semaine)
scripts/         téléchargement, backtest, contrôle sur hasard, calcul de mise, études
recherche/       tests des règles tirées des documents de trading (Sir Jones, BFX, SMV)
tests/           tests sur bougies construites à la main
```

## Lancer

```bash
pip install -r requirements.txt
python -m pytest -q
```

Données HistData (2015 à aujourd'hui en M1, quelques minutes de téléchargement) :

```bash
python scripts/download_histdata.py --start 2015 --end 2026-09
python scripts/run_backtest.py --m1-dir data/m1
```

Les prix HistData sont les prix bid de Dukascopy, identiques à la minute près (vérifié sur 2025-2026). HistData ne donne pas le spread : il est estimé à 1,7 point de base du prix, la valeur mesurée sur Dukascopy (0,27 $ en 2020, 0,55 à 0,8 $ en 2025-2026).

Ou les ticks Dukascopy, avec le spread réel. Le serveur limite fortement le débit (environ 15 s par fichier horaire, erreurs 503 au-delà d'une requête par seconde) : compter plusieurs jours pour 12 ans.

```bash
python scripts/download_dukascopy.py --start 2015-01-01 --end 2026-10-01
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

Entonnoir : à quelle règle chaque niveau candidat est éliminé.

```bash
python scripts/entonnoir.py --m1-dir data/m1 --start 2015-02-01 --end 2024-01-01
```

Étude du sweep de l'Asie : après le sweep du haut ou du bas asiatique et une confirmation, le mouvement qui suit est comparé au même trade pris dans les deux sens au même instant (contrôle neutre). Un écart nul veut dire que la confirmation n'apporte pas d'information sur le sens.

```bash
python scripts/etude_asie.py --m1-dir data/m1 --start 2015-02-01 --end 2024-01-01 --confirmation cisd
```

Traversée des niveaux de stops : au premier contact du jour d'un multiple de 50 $, du plus haut ou du plus bas de la veille ou de la semaine passée, ordre stop dans le sens de la traversée, stop 0,5 ATR H1, sortie à 16h New York. Mesuré à 2,5R : +0,16R par trade sur 2015-2023 (2 118 trades, t 4,5), +0,09R sur 2024-2026 (1 020 trades, t 1,9), +0,02R avec 0,05 ATR H1 de glissement. Les niveaux non ronds (`--decalage 16.67`) perdent le spread.

```bash
python scripts/etude_traversee.py --m1-dir data/m1 --start 2015-02-01 --end 2024-01-01
python scripts/etude_traversee.py --m1-dir data/m1 --start 2024-01-01 --glissement-atr-h1 0.05
```

L'avantage vient surtout des multiples de 100 $. Sur ces niveaux, le prix continue dans le sens de la cassure pendant 2 à 4 heures (`recherche/documents/h_trav_elan.py`), alors qu'au plus haut / bas de la veille l'élan s'arrête dans la minute du contact. Une entrée au marché à la clôture de cette minute suffit, avec un stop de 1 ATR H1, une cible de 3R et une sortie à 16h : +0,28R par trade sur 2015-2023 (376 trades, t 3,1), +0,10R sur 2024-2026 (339 trades, t 1,2). Le réglage a été choisi sur 2015-2023 (`recherche/documents/h_trav_exits.py`), puis testé une fois sur 2024-2026.

```bash
python scripts/etude_traversee.py --m1-dir data/m1 --start 2024-01-01 --familles ronds --pas 100 --stop-atr-h1 1 --entree close
```

Sur d'autres marchés, la même règle ne gagne pas après frais (`recherche/documents/h_trav_marches.py`). Sur USD/JPY, le prix continue bien dans le sens de la cassure des figures et demi-figures, et seulement sur ces niveaux (écart au contrôle +0,11R sur 2015-2023, t 5,6), mais le spread mange l'avantage : +0,04R par trade sur 2015-2023, -0,01R sur 2024-2026. L'argent, EUR/USD, GBP/USD, EUR/JPY, GBP/JPY et AUD/JPY perdent.

```bash
python scripts/download_histdata.py --pair USDJPY --start 2015 --end 2026-09 --spread-bp 1 --data data/usdjpy
python scripts/etude_traversee.py --m1-dir data/usdjpy/m1 --start 2024-01-01 --pas 0.5 --stop-atr-h1 1 --entree close --out out/usdjpy
python recherche/documents/h_trav_marches.py out/marches USDJPY=out/usdjpy/traversee.csv:1
```

Les règles des documents de trading (M/W de Sir Jones, retracement 62 % de BFX, equal highs, cycle de 3 jours...) sont testées dans `recherche/documents/` : aucune ne bat le hasard.

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
