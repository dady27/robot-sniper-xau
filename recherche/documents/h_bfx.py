"""BFX (power of three et sessions) : Londres haussier -> achat sur retracement de 62 % entre 07h et 09h New York.

Règles du document :
1. confirmer que la session de Londres était haussière (ici : clôture 07h NY > ouverture 03h NY) ;
2. entre 07h et 09h New York, attendre un retracement (idéalement >= 20 pips -> 0,27 ADR) ;
3. pas de retracement à 09h New York : on laisse tomber ;
4. entrée sur le 62 % du Fibonacci tracé de corps à corps (bougies M15) sur la jambe de Londres ;
5. cibles -0,27 (T1) et -0,62 (T2) ; -1 = prix symétrique.
Ajouts : le stop n'est pas donné -> sous la mèche basse de la jambe - 0,05 ADR ; sortie à 16h NY.
Miroir pour un Londres baissier.
usage: h_bfx.py START END OUT [ret_min_adr] [bias: none|prevday|sma20] [fib]
"""
import sys
from datetime import date

import numpy as np
import pandas as pd

from lab import PIP_ADR, Space, _span, business_days, by_year, daily, evaluate, load, ny, show


def bfx_day(m1, D, day, ret_min=20 * PIP_ADR, bias="none", fib=0.618, stop_buf=0.05):
    if day not in D.index or not np.isfinite(D.at[day, "adr"]):
        return []
    adr = D.at[day, "adr"]
    bars = _span(m1, ny(day, 3), ny(day, 16))
    sp = Space(bars)
    if sp.n < 500:
        return []
    i_lon_end, i_entry_end = sp.idx(ny(day, 7)), sp.idx(ny(day, 9))
    if i_lon_end < 200 or i_entry_end >= sp.n:
        return []
    O, H, L, C = sp.raw
    sign = 1 if C[i_lon_end - 1] > O[0] else -1
    if bias != "none":
        b = D.at[day, bias]
        if not np.isfinite(b) or b != sign:
            return []
    o, h, l, c, ec, xc = sp.sp[sign]
    t = sp.t[:i_lon_end]
    bucket = t.floor("15min").asi8
    starts = np.r_[0, np.nonzero(bucket[1:] != bucket[:-1])[0] + 1]
    ends = np.r_[starts[1:], i_lon_end]
    bo, bc = o[starts], c[ends - 1]
    bot, top = np.minimum(bo, bc), np.maximum(bo, bc)
    k_lo = int(np.argmin(bot))
    k_hi = k_lo + int(np.argmax(top[k_lo:]))
    lo_b, hi_b = bot[k_lo], top[k_hi]
    leg = hi_b - lo_b
    if leg <= 0 or k_hi == k_lo:
        return []
    level = hi_b - fib * leg
    if hi_b - level < ret_min * adr:
        return []
    # prix déjà sous le niveau à 07h : le retracement a eu lieu avant la fenêtre
    if c[i_lon_end - 1] + ec[i_lon_end - 1] <= level:
        return []
    wick_lo = l[starts[k_lo]:ends[k_hi]].min()
    stop = wick_lo - stop_buf * adr
    fill = np.nonzero(l[i_lon_end:i_entry_end] + ec[i_lon_end:i_entry_end] <= level)[0]
    if len(fill) == 0:
        return []
    k = i_lon_end + int(fill[0])
    row = evaluate(sp, k, sign, level - stop, sp.n, entry=level,
                   extra={"T1": hi_b + 0.27 * leg, "T2": hi_b + 0.62 * leg, "sym": hi_b + leg})
    if not row:
        return []
    row.update({"day": day, "leg_adr": leg / adr, "risk_adr": (level - stop) / adr,
                "fill_min": (sp.t[k] - sp.t[i_lon_end]).total_seconds() / 60})
    return [row]


def main():
    start, end, out = date.fromisoformat(sys.argv[1]), date.fromisoformat(sys.argv[2]), sys.argv[3]
    ret_min = float(sys.argv[4]) if len(sys.argv) > 4 else 20 * PIP_ADR
    bias = sys.argv[5] if len(sys.argv) > 5 else "none"
    fib = float(sys.argv[6]) if len(sys.argv) > 6 else 0.618
    m1 = load(start, end)
    D = daily(m1)
    D["prevday"] = np.sign(D["close"] - D["open"]).shift(1)
    D["sma20"] = np.sign(D["close"] - D["close"].rolling(20).mean()).shift(1)
    rows = []
    for d in business_days(start, end):
        rows += bfx_day(m1, D, d, ret_min, bias, fib)
    ev = pd.DataFrame(rows)
    ev.to_csv(out, index=False)
    show(ev, f"BFX 62 % NY {start}..{end} ret_min={ret_min:.2f} ADR biais={bias} fib={fib}", extra=("T1", "T2", "sym"))
    if not ev.empty:
        print("écart T1 par année :", by_year(ev, "r_T1", "c_T1"))
        print("x médian T1 / T2 :", round(ev.x_T1.median(), 2), round(ev.x_T2.median(), 2), "R ; jambe médiane",
              round(ev.leg_adr.median(), 2), "ADR ; risque médian", round(ev.risk_adr.median(), 2), "ADR")


main()
