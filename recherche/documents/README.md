# Règles des documents de trading, testées sur l'or

Chaque règle est codée comme le document la décrit, puis comparée au contrôle neutre
(même instant, même stop, achat et vente au hasard). Les pips EURUSD des documents sont
convertis en part de l'ATR journalier : 75 pips ~ 1 ADR. Étude sur 2015-2023.

Les données viennent de `data/m1` (ou du dossier donné par `LAB_M1`) :

```bash
python recherche/documents/h_btmm.py 2015-02-01 2024-01-01 out/btmm.csv
LAB_M1=chemin/vers/m1 python recherche/documents/h_stats.py 2015-02-01 2024-01-01
```

| Script | Règle | Source | Résultat 2015-2023 |
|---|---|---|---|
| `h_btmm.py` | M ou W de la 2e jambe après le stop hunt de Londres (01h-05h NY), stop 0,11 ADR au-delà du HOD/LOD | Sir Jones, BTMM | 861 trades, -0,09R par trade à 2,5R, -0,06R contre le hasard (t -1,5) |
| `h_bfx.py` | Londres haussier, achat sur le 62 % entre 07h et 09h NY, cibles -0,27 et -0,62 | BFX | 98 trades, -0,10R ; sans minimum de retracement 1 187 trades, +0,01R contre le hasard |
| `h_stats.py` | Londres puis New York ; stop hunt = HOD/LOD ; cycle de 3 jours ; jour du plus haut / bas de la semaine | BFX, Sir Jones | corrélation +0,03 ; 25,5 % contre 24,1 % au hasard ; rien ; mardi-mercredi 27,2 % contre 27,9 % |
| `h_levels.py` | equal highs / lows comme aimant ; chiffres ronds comme retournement | Sir Jones, SMV, BFX | 49,6 % contre 50 % ; au 1er contact d'un multiple de 50 $, retournement 42 % seulement |
| `h_filters.py` | biais HTF et premium / discount sur les sweeps de l'Asie (`scripts/etude_asie.py`) | fiches SMV | au mieux +0,08R contre le hasard, espérance nette négative |

Seule la dernière ligne de `h_levels.py` donne un effet net, et il dit l'inverse du document :
le prix traverse les chiffres ronds. La règle de trading qui en découle est dans
`sniper/traversee.py` et `scripts/etude_traversee.py`.
