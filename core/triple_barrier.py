import numpy as np
import pandas as pd

def triple_barrier_labeling(
    df: pd.DataFrame,
    tp_pips: float,
    sl_pips: float,
    max_horizon: int,
    price_col: str = "close",
    pip_value: float = 0.0001,
) -> pd.Series:
    prices = df[price_col].values
    n = len(prices)

    tp_dist = tp_pips * pip_value
    sl_dist = sl_pips * pip_value

    labels = np.zeros(n, dtype=np.int8)

    for i in range(n):
        entry = prices[i]
        upper = entry + tp_dist
        lower = entry - sl_dist

        end = min(i + max_horizon, n - 1)
        if end == i:
            continue

        future = prices[i + 1 : end + 1]

        hit_up = np.where(future >= upper)[0]
        hit_dn = np.where(future <= lower)[0]

        if len(hit_up) == 0 and len(hit_dn) == 0:
            labels[i] = 0
            continue

        first_up = hit_up[0] if len(hit_up) else np.inf
        first_dn = hit_dn[0] if len(hit_dn) else np.inf

        if first_up < first_dn:
            labels[i] = 1
        elif first_dn < first_up:
            labels[i] = -1
        else:
            labels[i] = 0

    return pd.Series(labels, index=df.index, name="target")