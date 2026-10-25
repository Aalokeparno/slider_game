"""Bootstrap / Wilcoxon test and summary tables, moved out of the Stage 1 notebooks.

The function bodies are unchanged. Functions that used notebook variables
(`play`, `per_run`, `res`, `topologies`) now take them as arguments.
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# ------------------------------------------------------- Stage 1, HW 3 (H1 test)

def paired_test(play, datasets, top_a, top_b, column):
    d = play[play["dataset"].isin(datasets)]
    means = d.groupby(["file", "topology"])[column].mean().unstack()   # one row per group
    diff = (means[top_a] - means[top_b]).dropna().values               # d_g (a group with no round of one topology is dropped)

    # bootstrap 95% CI of the mean of d_g: resample the groups 10,000 times
    rng = np.random.default_rng(0)
    boot = rng.choice(diff, size=(10000, len(diff))).mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])

    p = wilcoxon(diff).pvalue                                          # signed-rank test, two-sided
    return [len(diff), (diff > 0).sum(), diff.mean(), low, high, p]


# --------------------------------------------------- Stage 1, HW 4 (Ring traps)

# table: rows pooled / CL / EL, all ratios in %
def trap_table(per_run, suffix=""):
    out = {}
    for label, datasets in [("pooled", ["CL", "EL"]), ("CL", ["CL"]), ("EL", ["EL"])]:
        d = per_run[per_run["dataset"].isin(datasets)]
        T52, T43, F52, F43 = [d[k + suffix].sum() for k in ["T52", "T43", "F52", "F43"]]
        out[label] = {"runs": len(d), "T52": T52, "T43": T43, "F52": F52, "F43": F43,
                      "P(4-3|F) %  [50 unif, 75 vol]": 100 * F43 / (F43 + F52),
                      "occupancy 4-3 %  [50]": 100 * T43 / (T43 + T52),
                      "P(F|4-3) %  [8.33]": 100 * F43 / T43,
                      "P(F|5-2) %  [2.78]": 100 * F52 / T52}
    return pd.DataFrame(out).T.astype({"runs": int})


def ratios(F52, F43, T52, T43):
    return [100 * F43 / (F43 + F52), 100 * T43 / (T43 + T52), (F43 / T43) / (F52 / T52)]   # P(4-3|F) %, occupancy %, thick/thin


# --------------------------------------------------------- Stage 1, HW 5 (APEM)

def summary(success, time, fraught):
    return {"runs": len(success),
            "success rate %": 100 * np.mean(success),
            "mean time to consensus (s)": np.mean(time[success]),
            "s.d. time to consensus (s)": np.std(time[success], ddof=1),
            "mean % fraught (hold)": np.mean(fraught),
            "# runs > 10% fraught": np.sum(fraught > 10)}


# 5.3 summary plus the authors' measure (successful runs only)
def summary_b(success, time, hold, osf):
    out = summary(success, time, hold)
    out["mean % fraught, successful runs (warp)"] = np.mean(osf[success])
    out["# successful runs > 10% (warp)"] = np.sum(osf[success] > 10)
    return out


# the five metrics of one cell (used for APEM and for humans)
def metrics(success, t_end, hold, interp):
    success, t_end, hold, interp = [np.array(x) for x in [success, t_end, hold, interp]]
    success = success.astype(bool)
    return {"runs": len(success),
            "success %": 100 * success.mean(),
            "mean time (s)": t_end[success].mean(),
            "% fraught (hold)": hold.mean(),
            "% fraught (authors)": interp[success].mean(),
            "% runs > 10% (hold)": 100 * (hold > 10).mean()}


# direction of a ratio (up = above 1)
def direction(x):
    if np.isnan(x):
        return "n/a"                                     # no successful run in a cell
    return "up" if x > 1 else ("down" if x < 1 else "equal")


# ranking of topologies by % fraught (hold) for one source and group of res
def ranking(source, group, res, topologies):
    values = {top: round(res.loc[(source, group, top), "% fraught (hold)"], 2) for top in topologies}
    order = sorted(topologies, key=lambda top: (-values[top], top))     # highest first, ties alphabetical
    text = order[0]
    for a, b in zip(order, order[1:]):
        text += (" = " if values[a] == values[b] else " > ") + b      # "=" for a tie (e.g. both 0%)
    return text
