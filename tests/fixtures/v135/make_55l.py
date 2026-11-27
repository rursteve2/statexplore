"""Generator of StatExplore_Test55L_LegacyUnderflow.csv (v1.3.5, AUD134-01): a single-level Gaussian fixture whose two
strongest focal interactions both have p below the double range (ordinary p == 0) with different log tails."""
import sys
import numpy as np
import pandas as pd
from scipy import stats

def make(b13, b12, n=6000, seed=135):
    rng = np.random.default_rng(seed)
    x1, x2, x3 = rng.normal(size=(3, n))
    y = 1 + 0.5 * x1 + 0.5 * x2 + 0.5 * x3 + b12 * x1 * x2 + b13 * x1 * x3 + rng.normal(size=n)
    return pd.DataFrame({"ID": np.arange(1, n + 1), "x1": x1.round(6), "x3": x3.round(6), "x2": x2.round(6),
                         "y": y.round(6)})

def logp(d, a, b):
    X0 = np.column_stack([np.ones(len(d)), d[a], d[b]])
    X1 = np.column_stack([X0, d[a] * d[b]])
    y = d["y"].to_numpy()
    r0 = y - X0 @ np.linalg.lstsq(X0, y, rcond=None)[0]
    r1 = y - X1 @ np.linalg.lstsq(X1, y, rcond=None)[0]
    df2 = len(d) - 4
    F = ((r0 @ r0) - (r1 @ r1)) / ((r1 @ r1) / df2)
    from statexplore.analysis.lrtest import log_upper_gamma_q
    return round(F, 1), log_upper_gamma_q(0.5, F / 2) / np.log(10)   # chi-square(1) approximation for tuning only

if __name__ == "__main__":
    b13, b12 = float(sys.argv[1]), float(sys.argv[2])
    d = make(b13, b12)
    print("x1:x3", logp(d, "x1", "x3"), "x1:x2", logp(d, "x1", "x2"), "x3:x2", logp(d, "x3", "x2"))
    if len(sys.argv) > 3:
        d.to_csv(sys.argv[3], index=False)
