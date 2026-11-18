"""Unit 5: which graph-only measures predict trap time under the logit rule (pre-registered).

Locked choices: docs/u5_preregistration.md (commit config.U5_H_LOCK). Trap graphs only; T = 0.2, 0.5, 1.0;
outcomes log Yb (primary), log Ya; 14 predictors (some logged); OLS line per predictor x T x outcome, fitted on
S-train and frozen; R2 on the scoring set (SS_tot around its own mean), Spearman; bootstrap over scoring graphs
(2000, seed 573) for paired CIs; two-term law on log duration.

S-train work here (train_ranking, train_law, the expectation file). S-test is touched only by open_test, run once
by scripts/u5_test.py --open; dry_run is its stand-in on a 70 / 30 split of the S-train trap graphs.
"""
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from slider import config, markov
from slider import landscape as L

T_U5 = [0.2, 0.5, 1.0]
OUTCOMES = ["log_Yb", "log_Ya"]
LOCAL = ["n_traps", "n_plateaus", "B_max", "B_mean", "W_max", "W_mean", "P_max", "n_gates_min",
         "progress_share_mean"]
SPECTRAL = ["lam2", "conductance", "modularity", "lamP"]
PREDICTORS = LOCAL + SPECTRAL + ["n_edges"]
LOGGED = {"n_traps", "n_plateaus", "P_max", "W_max", "W_mean", "n_gates_min", "n_edges"}
N_BOOT = 2000
N_FOLDS = 5
FACTOR_OUTCOMES = ["log_entries", "log_dur"]
FAMILY = {**{p: "global" for p in SPECTRAL}, **{p: "landscape" for p in LOCAL}, "n_edges": "baseline"}
REASON = "Unit 5: the one pre-registered S-test scoring (docs/u5_preregistration.md)"
CALLER = "scripts/u5_test.py --open"


# ---------------------------------------------------------------- data

def predictors():
    """(X, split): the 14 transformed predictors of every trap graph (index atlas_id; graph-only, all splits)
    and the graph's split. W_mean in X is log W_mean."""
    G = pd.read_csv(config.U1_GRAPHS)
    traps = pd.read_csv(config.U1_TRAPS)
    G = G[G["n_traps"] > 0].set_index("atlas_id")
    G["progress_share_mean"] = traps.groupby("atlas_id")["progress_share_P"].mean().reindex(G.index).fillna(0)
    X = pd.DataFrame({p: np.log(G[p]) if p in LOGGED else G[p] for p in PREDICTORS}, index=G.index)
    return X, G["split"]


def outcomes_from_rows(rows):
    """atlas_id, T, log_Yb, log_Ya, log_dur at T in T_U5, from trap_time rows (u2b_graph_train.csv columns)."""
    d = pd.DataFrame(rows)
    d = d[d["T"].isin(T_U5)].copy()
    d["log_Yb"], d["log_Ya"], d["log_dur"] = np.log(d["Yb"]), np.log(d["Ya"]), np.log(d["duration"])
    d["log_entries"] = np.log(d["entries"])
    return d[["atlas_id", "T", "log_Yb", "log_Ya", "log_dur", "log_entries"]].reset_index(drop=True)


def train_outcomes():
    """Outcomes of the S-train trap graphs, from u2b_graph_train.csv (Unit 2b)."""
    X, split = predictors()
    rows = pd.read_csv(config.U2B_GRAPH)
    d = outcomes_from_rows(rows[rows["atlas_id"].isin(X.index[split == "train"])])
    assert d["atlas_id"].nunique() == (split == "train").sum()
    return d


def compute_outcomes(atlas_ids, allow_test=False, reason=None):
    """Outcomes computed with markov.graph_outcomes (raises for an S-test graph unless allow_test with a reason)."""
    graphs, split, u1 = L.atlas_graphs(), L.load_graph_split(), pd.read_csv(config.U1_TRAPS)
    rows = [r for a in atlas_ids for r in markov.graph_outcomes(a, split, graphs, u1, T_U5, allow_test, reason)]
    return outcomes_from_rows(rows)


def frame(X, d, T):
    """Outcomes at T joined with the predictors, one row per graph (sorted by atlas_id)."""
    F = d[d["T"] == T].set_index("atlas_id").sort_index().join(X, how="left")
    assert F[PREDICTORS].notna().all().all()
    return F


# ---------------------------------------------------------------- fits and scores

def line(x, y):
    """OLS y = a + b x; returns (a, b)."""
    b, a = np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)
    return a, b


def r2(y, pred):
    """1 - SS_res / SS_tot, SS_tot around the mean of y; works along the last axis (rows = resamples)."""
    return 1 - np.sum((y - pred) ** 2, axis=-1) / np.sum((y - y.mean(axis=-1, keepdims=True)) ** 2, axis=-1)


def score(y, preds, x, seed=config.SEED, n_boot=N_BOOT):
    """R2 and Spearman (x vs y) of each predictor's predictions; rank by R2 (1 = best); 95% paired-bootstrap CI
    of R2 - R2(rank 1), resampling graphs. Returns (table, boot) with boot = R2 per resample x predictor."""
    y, P = np.asarray(y, float), preds.to_numpy(float)
    cols = list(preds.columns)
    R2 = np.array([r2(y, P[:, j]) for j in range(len(cols))])
    idx = np.random.default_rng(seed).integers(0, len(y), (n_boot, len(y)))
    boot = np.column_stack([r2(y[idx], P[idx, j]) for j in range(len(cols))])
    top = int(np.argmax(R2))
    lo, hi = np.percentile(boot - boot[:, [top]], [2.5, 97.5], axis=0)
    table = pd.DataFrame({"predictor": cols, "R2": R2,
                          "spearman": [spearmanr(x[c], y)[0] for c in cols],
                          "rank": pd.Series(R2).rank(ascending=False, method="min").astype(int).to_numpy(),
                          "vs_top": cols[top], "diff_lo": lo, "diff_hi": hi})
    return table, pd.DataFrame(boot, columns=cols)


def rank_table(X, d_fit, d_score, outcomes=OUTCOMES):
    """Lines fitted on d_fit and frozen, scored on d_score: one row per outcome x T x predictor
    (a, b, R2, spearman, rank, CI vs rank 1). boot[(outcome, T)] = bootstrap R2 per predictor."""
    out, boot = [], {}
    for T in T_U5:
        F, S = frame(X, d_fit, T), frame(X, d_score, T)
        for outcome in outcomes:
            ab = {p: line(F[p], F[outcome]) for p in PREDICTORS}
            preds = pd.DataFrame({p: ab[p][0] + ab[p][1] * S[p] for p in PREDICTORS})
            t, boot[(outcome, T)] = score(S[outcome], preds, S)
            t.insert(0, "outcome", outcome)
            t.insert(1, "T", T)
            t.insert(3, "a", [ab[p][0] for p in t["predictor"]])
            t.insert(4, "b", [ab[p][1] for p in t["predictor"]])
            t["n_fit"], t["n_score"] = len(F), len(S)
            out.append(t)
    return pd.concat(out, ignore_index=True), boot


def law_fit(F, T):
    """Fixed: log dur - B_max / T = a + b log W_mean; free: log dur = a + b log W_mean + c B_max (OLS)."""
    a_f, b_f = line(F["W_mean"], F["log_dur"] - F["B_max"] / T)
    M = np.column_stack([np.ones(len(F)), F["W_mean"], F["B_max"]])
    a, b, c = np.linalg.lstsq(M, F["log_dur"].to_numpy(float), rcond=None)[0]
    return {"a_fixed": a_f, "b_fixed": b_f, "a_free": a, "b_free": b, "c_free": c}


def c_boot(S, seed=config.SEED, n_boot=N_BOOT):
    """c of the free law model refitted on bootstrap resamples of the graphs (NaN if B_max is constant)."""
    idx = np.random.default_rng(seed).integers(0, len(S), (n_boot, len(S)))
    W, B, y = S["W_mean"].to_numpy(float), S["B_max"].to_numpy(float), S["log_dur"].to_numpy(float)
    cs = np.full(n_boot, np.nan)
    for r in range(n_boot):
        i = idx[r]
        if np.ptp(B[i]) > 0:
            M = np.column_stack([np.ones(len(i)), W[i], B[i]])
            cs[r] = np.linalg.lstsq(M, y[i], rcond=None)[0][2]
    return cs


def law_table(X, d_fit, d_score):
    """Per T: law fits on d_fit (frozen) and their R2 on d_score; c by OLS on d_score with bootstrap CI over
    d_score graphs; law_fits = the CI contains 1/T."""
    rows = []
    for T in T_U5:
        F, S = frame(X, d_fit, T), frame(X, d_score, T)
        f = law_fit(F, T)
        fixed = f["a_fixed"] + f["b_fixed"] * S["W_mean"] + S["B_max"] / T
        free = f["a_free"] + f["b_free"] * S["W_mean"] + f["c_free"] * S["B_max"]
        cs = c_boot(S)
        lo, hi = np.nanpercentile(cs, [2.5, 97.5])
        rows.append({"T": T, **f, "R2_fixed": r2(S["log_dur"].to_numpy(), fixed.to_numpy()),
                     "R2_free": r2(S["log_dur"].to_numpy(), free.to_numpy()),
                     "c_score": law_fit(S, T)["c_free"], "c_lo": lo, "c_hi": hi, "inv_T": 1 / T,
                     "law_fits": bool(lo <= 1 / T <= hi), "n_walled_score": int((S["B_max"] > 0).sum()),
                     "n_boot_c_undefined": int(np.isnan(cs).sum()), "n_fit": len(F), "n_score": len(S)})
    return pd.DataFrame(rows)


def judge(rank, boot, law, r2_col="R2"):
    """P1-P4 (docs/u5_preregistration.md) on log Yb: pass / fail with details. A beats B: the 2.5th percentile
    of the bootstrap R2(A) - R2(B) is > 0."""
    def order(T):
        r = rank[(rank["outcome"] == "log_Yb") & (rank["T"] == T)]
        return list(r.sort_values(r2_col, ascending=False)["predictor"])

    def beats(T, A, B):
        d = boot[("log_Yb", T)]
        return bool(np.percentile(d[A] - d[B], 2.5) > 0)

    o2, o10 = order(0.2), order(1.0)
    rows = [{"prediction": "P1", "result": o2[0] == "B_max",
             "detail": f"T = 0.2: #1 = {o2[0]}; B_max is #{o2.index('B_max') + 1}"},
            {"prediction": "P2", "result": o10[0] == "n_traps",
             "detail": f"T = 1.0: #1 = {o10[0]}; n_traps #{o10.index('n_traps') + 1}, P_max #{o10.index('P_max') + 1}"}]
    bad = []
    for T in T_U5:
        best = next(p for p in order(T) if p in LOCAL)
        bad += [f"T={T}: {s} beats {t}" for s in SPECTRAL for t in [best, "n_edges"] if beats(T, s, t)]
    rows.append({"prediction": "P3", "result": not bad,
                 "detail": "; ".join(bad) or "no spectral measure beats the best local measure or n_edges"})
    fits = law.set_index("T")["law_fits"]
    rows.append({"prediction": "P4", "result": bool(fits[0.2] and not fits[1.0]),
                 "detail": f"law fits at T = 0.2: {fits[0.2]}; at T = 1.0: {fits[1.0]}"})
    out = pd.DataFrame(rows)
    out["result"] = out["result"].map({True: "pass", False: "fail"})
    return out


def run_scoring(X, d_fit, d_score, paths):
    """The pre-registered scoring: rank table, law table, P1-P4; written to paths (ranking, law, predictions)."""
    rank, boot = rank_table(X, d_fit, d_score)
    law = law_table(X, d_fit, d_score)
    preds = judge(rank, boot, law)
    for table, path in zip([rank, law, preds], paths):
        table.to_csv(path, index=False)
    return rank, law, preds


# ---------------------------------------------------------------- S-train ranking and expectation

def folds(n, seed=config.SEED, k=N_FOLDS):
    """k folds of range(n) (a random permutation split into k nearly equal parts)."""
    return np.array_split(np.random.default_rng(seed).permutation(n), k)


def train_ranking(outcomes=OUTCOMES):
    """S-train only. Per outcome x T x predictor: a, b, R2_train (in sample), spearman_train, rank_train
    (by R2_train), R2_cv (pooled out-of-fold predictions, 5 folds, seed 573), rank_cv (by R2_cv) and the
    paired-bootstrap CI of R2_cv - R2_cv(rank_cv 1) over S-train graphs. boot as in rank_table (out-of-fold)."""
    X, split = predictors()
    d = train_outcomes()
    out, boot = [], {}
    for T in T_U5:
        F = frame(X, d, T)
        parts = folds(len(F))
        for outcome in outcomes:
            y = F[outcome].to_numpy(float)
            oof = pd.DataFrame(index=F.index, columns=PREDICTORS, dtype=float)
            rows = []
            for p in PREDICTORS:
                x = F[p].to_numpy(float)
                a, b = line(x, y)
                for part in parts:
                    keep = np.setdiff1d(np.arange(len(F)), part)
                    a_k, b_k = line(x[keep], y[keep])
                    oof.iloc[part, oof.columns.get_loc(p)] = a_k + b_k * x[part]
                rows.append({"outcome": outcome, "T": T, "predictor": p, "a": a, "b": b,
                             "R2_train": r2(y, a + b * x), "spearman_train": spearmanr(x, y)[0]})
            t = pd.DataFrame(rows)
            t["rank_train"] = t["R2_train"].rank(ascending=False, method="min").astype(int)
            s, boot[(outcome, T)] = score(y, oof, F)
            t["R2_cv"], t["rank_cv"] = s["R2"].to_numpy(), s["rank"].to_numpy()
            t["cv_vs_top"], t["cv_diff_lo"], t["cv_diff_hi"] = s["vs_top"], s["diff_lo"], s["diff_hi"]
            t["n_train"] = len(F)
            out.append(t)
    return pd.concat(out, ignore_index=True), boot


def train_law():
    """Law on S-train: fits, in-sample R2, c with bootstrap CI over S-train graphs (fit set = scoring set)."""
    X, _ = predictors()
    d = train_outcomes()
    return law_table(X, d, d)


def counts():
    """Trap graphs per split (pre-registration item 1) and walled graphs (B_max > 0) per split."""
    X, split = predictors()
    return pd.DataFrame({"trap_graphs": split.value_counts(), "walled": split[X["B_max"] > 0].value_counts()})


def expectation_text(rank, law, preds, n):
    lines = ["# Unit 5 — S-train expectation (before opening S-test)", "",
             "Written by `u5.save_train()` (notebooks/path2/u5_train_ranking.ipynb) from S-train only "
             f"({n['train']} trap graphs). Pre-registration: docs/u5_preregistration.md (H_lock {config.U5_H_LOCK}).",
             "Ranking by R2_cv: pooled out-of-fold R2 of 5-fold CV within S-train (seed 573). "
             "\"Beats\" for P3: 95% paired-bootstrap CI (2000, seed 573, over S-train graphs) of the out-of-fold "
             "R2 difference is above 0.", ""]
    for outcome in OUTCOMES:
        lines += [f"## Top 5 by R2_cv — {outcome}", "", "| T | #1 | #2 | #3 | #4 | #5 |", "|---|---|---|---|---|---|"]
        for T in T_U5:
            r = rank[(rank["outcome"] == outcome) & (rank["T"] == T)].sort_values("R2_cv", ascending=False).head(5)
            lines.append(f"| {T} | " + " | ".join(f"{p} ({v:.3f})" for p, v in zip(r["predictor"], r["R2_cv"])) + " |")
        lines.append("")
    lines += ["## Two-term law on S-train (log duration)", "",
              "| T | 1/T | c (free, S-train) | 95% CI | law fits | R2 fixed | R2 free | walled |",
              "|---|---|---|---|---|---|---|---|"]
    for _, r in law.iterrows():
        lines.append(f"| {r['T']} | {r['inv_T']:.2f} | {r['c_score']:.3f} | [{r['c_lo']:.3f}, {r['c_hi']:.3f}] | "
                     f"{'yes' if r['law_fits'] else 'no'} | {r['R2_fixed']:.3f} | {r['R2_free']:.3f} | "
                     f"{r['n_walled_score']} |")
    lines += ["", "## Predictions", ""]
    for _, r in preds.iterrows():
        verdict = "train supports" if r["result"] == "pass" else "train contradicts"
        lines.append(f"- {r['prediction']}: {verdict} ({r['detail']})")
    return "\n".join(lines) + "\n"


def save_train():
    """S-train ranking, law and P1-P4 on S-train; writes config.U5_TRAIN_RANKING, U5_TRAIN_LAW, U5_EXPECTATION."""
    rank, boot = train_ranking()
    law = train_law()
    preds = judge(rank, boot, law, r2_col="R2_cv")
    rank.to_csv(config.U5_TRAIN_RANKING, index=False)
    law.to_csv(config.U5_TRAIN_LAW, index=False)
    config.U5_EXPECTATION.write_text(expectation_text(rank, law, preds, counts()["trap_graphs"]))
    return rank, law, preds


# ---------------------------------------------------------------- addendum: entries vs duration (S-train)

def two_var_line(F, xE, xD, outcome="log_Yb"):
    """OLS outcome = a + bE xE + bD xD on F; returns (a, bE, bD)."""
    M = np.column_stack([np.ones(len(F)), F[xE], F[xD]])
    return np.linalg.lstsq(M, F[outcome].to_numpy(float), rcond=None)[0]


def two_var_predict(coef, S, xE, xD):
    return coef[0] + coef[1] * S[xE] + coef[2] * S[xD]


def top_by(rank, outcome, T, col="R2_cv", k=0):
    """The k-th best predictor (0 = #1) for outcome at T, by col (descending)."""
    r = rank[(rank["outcome"] == outcome) & (rank["T"] == T)]
    return r.sort_values(col, ascending=False)["predictor"].iloc[k]


def train_factors():
    """S-train only. Unit 5 single-predictor ranking (train_ranking) for log entries and log duration, with the
    predictor family (model = "single"); per T one row (model = "two_var") for log Yb = a + bE x_E + bD x_D,
    x_E = #1 for log entries, x_D = #1 for log duration (#2 if it equals x_E), by R2_cv: R2_train, R2_cv (pooled
    out-of-fold, the same 5 folds) and the paired-bootstrap CI (out-of-fold, over S-train graphs) of
    R2_cv(two_var) - R2_cv(best single for log Yb)."""
    single, _ = train_ranking(FACTOR_OUTCOMES)
    single["family"] = single["predictor"].map(FAMILY)
    single["model"] = "single"
    rank_y, _ = train_ranking(["log_Yb"])
    X, _ = predictors()
    d = train_outcomes()
    rows = []
    for T in T_U5:
        xE = top_by(single, "log_entries", T)
        xD = top_by(single, "log_dur", T, k=0 if top_by(single, "log_dur", T) != xE else 1)
        best = top_by(rank_y, "log_Yb", T)
        F = frame(X, d, T)
        y = F["log_Yb"].to_numpy(float)
        coef = two_var_line(F, xE, xD)
        oof = pd.DataFrame(index=F.index, columns=["two_var", best], dtype=float)
        for part in folds(len(F)):
            keep = np.setdiff1d(np.arange(len(F)), part)
            Fk, Sk = F.iloc[keep], F.iloc[part]
            oof.iloc[part, 0] = two_var_predict(two_var_line(Fk, xE, xD), Sk, xE, xD).to_numpy()
            a, b = line(Fk[best], Fk["log_Yb"])
            oof.iloc[part, 1] = (a + b * Sk[best]).to_numpy()
        _, boot = score(y, oof, oof)
        lo, hi = np.percentile(boot["two_var"] - boot[best], [2.5, 97.5])
        rows.append({"outcome": "log_Yb", "T": T, "predictor": f"{xE} + {xD}", "model": "two_var",
                     "x_E": xE, "x_D": xD, "a": coef[0], "b_E": coef[1], "b_D": coef[2],
                     "R2_train": r2(y, two_var_predict(coef, F, xE, xD).to_numpy()),
                     "R2_cv": r2(y, oof["two_var"].to_numpy()), "best_single": best,
                     "R2_cv_best_single": r2(y, oof[best].to_numpy()), "cv_diff_lo": lo, "cv_diff_hi": hi,
                     "n_train": len(F)})
    return pd.concat([single, pd.DataFrame(rows)], ignore_index=True)


def story(factors, T):
    """S-train: entries' #1 is a global measure and duration's #1 is a landscape measure."""
    e, dur = top_by(factors, "log_entries", T), top_by(factors, "log_dur", T)
    return FAMILY[e] == "global" and FAMILY[dur] == "landscape"


def addendum_text(factors):
    single, two = factors[factors["model"] == "single"], factors[factors["model"] == "two_var"].set_index("T")
    name = {"log_entries": "log entries", "log_dur": "log duration"}
    lines = ["# Unit 5 — addendum: entries vs duration (S-train only; S-test closed)", "",
             "Written by `u5.save_factors()` (notebooks/path2/u5_factors.ipynb) from S-train only, before S-test is "
             f"opened. Pre-registration: docs/u5_preregistration.md (H_lock {config.U5_H_LOCK}); S-train expectation: "
             f"docs/u5_train_expectation.md (H_exp {config.U5_H_EXP}); neither is changed. Table: "
             "results/tables/u5_train_factors.csv.",
             "Method as Unit 5: trap graphs, T = 0.2, 0.5, 1.0, the 14 predictors and transforms, one-variable OLS, "
             "R2_cv = pooled 5-fold out-of-fold R2 (seed 573). Outcomes: log entries and log duration "
             "(duration = Y_b / entries) from u2b_graph_train.csv. Families: global = {modularity, conductance, "
             "lam2, lamP}; landscape = {n_traps, n_plateaus, B_max, B_mean, W_max, W_mean, P_max, n_gates_min, "
             "progress_share_mean}; baseline = {n_edges}.", ""]
    for outcome in FACTOR_OUTCOMES:
        lines += [f"## (i) Top 5 by R2_cv — {name[outcome]}", "", "| T | #1 | #2 | #3 | #4 | #5 |",
                  "|---|---|---|---|---|---|"]
        for T in T_U5:
            r = single[(single["outcome"] == outcome) & (single["T"] == T)].sort_values("R2_cv", ascending=False)
            lines.append(f"| {T} | " + " | ".join(f"{p} ({v:.3f}, {FAMILY[p]})" for p, v in
                                                  zip(r["predictor"].head(5), r["R2_cv"].head(5))) + " |")
        lines.append("")
    lines += ["Two-variable model for log Y_b (x_E = #1 for log entries, x_D = #1 for log duration, or its #2 if "
              "the same), against the best single predictor for log Y_b (S-train, R2_cv):", "",
              "| T | x_E | x_D | R2_cv two-variable | best single | R2_cv best single | CI of difference |",
              "|---|---|---|---|---|---|---|"]
    for T in T_U5:
        r = two.loc[T]
        lines.append(f"| {T} | {r['x_E']} | {r['x_D']} | {r['R2_cv']:.3f} | {r['best_single']} | "
                     f"{r['R2_cv_best_single']:.3f} | [{r['cv_diff_lo']:.3f}, {r['cv_diff_hi']:.3f}] |")
    lines += ["", "## (ii) S-train verdict on the story (description only)", "",
              "Story: entries' #1 is a global measure, duration's #1 is a landscape measure.", ""]
    for T in T_U5:
        e, dur = top_by(factors, "log_entries", T), top_by(factors, "log_dur", T)
        lines.append(f"- T = {T}: train {'supports' if story(factors, T) else 'contradicts'} "
                     f"(entries #1 = {e}, {FAMILY[e]}; duration #1 = {dur}, {FAMILY[dur]})")
    tops = {(o, T): top_by(factors, o, T) for o in FACTOR_OUTCOMES for T in T_U5}
    lines += ["", "## (iii) Confirmatory hypotheses (S-test, scored once by scripts/u5_test.py --open)", "",
              "Rules as the pre-registration: lines fitted on S-train and frozen; S-test R2 = 1 - SS_res / SS_tot "
              "(SS_tot around the S-test mean); bootstrap over S-test graphs, 2000 resamples, seed 573; "
              "\"A beats B\" = the 95% paired-bootstrap CI of R2(A) - R2(B) excludes 0 with A higher. Each hypothesis "
              "holds only if every case listed under it passes.", "",
              "- A1 (transfer). The S-train #1 predictor is not beaten on S-test by any of the other 13 predictors:"]
    for o in FACTOR_OUTCOMES:
        lines.append(f"  - {name[o]}: " + "; ".join(f"T = {T}: {tops[(o, T)]}" for T in T_U5))
    lines.append("- A2 (families). The S-test #1 predictor is in the same family as the S-train #1:")
    for o in FACTOR_OUTCOMES:
        lines.append(f"  - {name[o]}: " + "; ".join(f"T = {T}: {FAMILY[tops[(o, T)]]} (S-train #1 {tops[(o, T)]})"
                                                   for T in T_U5))
    lines.append("- A3 (two-part model). The two-variable line log Y_b = a + b_E x_E + b_D x_D (fitted on S-train, "
                 "frozen) beats the best single predictor for log Y_b on S-test, i.e. the S-test #1 of the 14 "
                 "pre-registered single-predictor lines (frozen on S-train):")
    lines.append("  - " + "; ".join(f"T = {T}: {two.loc[T, 'x_E']} + {two.loc[T, 'x_D']} (S-train best single "
                                    f"{two.loc[T, 'best_single']})" for T in T_U5))
    return "\n".join(lines) + "\n"


def save_factors():
    """Writes config.U5_TRAIN_FACTORS and config.U5_ADDENDUM (S-train only)."""
    factors = train_factors()
    factors.to_csv(config.U5_TRAIN_FACTORS, index=False)
    config.U5_ADDENDUM.write_text(addendum_text(factors))
    return factors


def addendum_scoring(X, d_fit, d_score, factors, path):
    """A1-A3 on the scoring set with lines frozen on d_fit; names from factors (u5_train_factors.csv).
    One row per hypothesis x outcome x T: result pass / fail, R2s and the CI behind the call."""
    rank_f, boot_f = rank_table(X, d_fit, d_score, FACTOR_OUTCOMES)
    rows = []
    for o in FACTOR_OUTCOMES:
        for T in T_U5:
            top = top_by(factors[factors["model"] == "single"], o, T)
            r = rank_f[(rank_f["outcome"] == o) & (rank_f["T"] == T)].set_index("predictor")
            test_top = r["R2"].idxmax()
            b = boot_f[(o, T)]
            beaten = [q for q in PREDICTORS if q != top and np.percentile(b[q] - b[top], 2.5) > 0]
            lo, hi = np.percentile(b[test_top] - b[top], [2.5, 97.5])
            base = {"outcome": o, "T": T, "train_top": top, "R2_train_top": r.loc[top, "R2"], "test_top": test_top,
                    "R2_test_top": r.loc[test_top, "R2"], "diff_lo": lo, "diff_hi": hi}
            rows.append({"hypothesis": "A1", **base, "result": not beaten,
                         "detail": ("beaten by " + ", ".join(beaten)) if beaten else "not beaten"})
            rows.append({"hypothesis": "A2", **base, "result": FAMILY[test_top] == FAMILY[top],
                         "detail": f"train {FAMILY[top]}, test {FAMILY[test_top]}"})
    two = factors[factors["model"] == "two_var"].set_index("T")
    for T in T_U5:
        xE, xD = two.loc[T, "x_E"], two.loc[T, "x_D"]
        F, S = frame(X, d_fit, T), frame(X, d_score, T)
        preds = {}
        for p in PREDICTORS:
            a, b = line(F[p], F["log_Yb"])
            preds[p] = a + b * S[p]
        preds["two_var"] = two_var_predict(two_var_line(F, xE, xD), S, xE, xD)
        preds = pd.DataFrame(preds)
        t, boot = score(S["log_Yb"], preds, preds)
        t = t.set_index("predictor")
        best = t.loc[PREDICTORS, "R2"].idxmax()
        lo, hi = np.percentile(boot["two_var"] - boot[best], [2.5, 97.5])
        rows.append({"hypothesis": "A3", "outcome": "log_Yb", "T": T, "train_top": f"{xE} + {xD}",
                     "R2_train_top": t.loc["two_var", "R2"], "test_top": best, "R2_test_top": t.loc[best, "R2"],
                     "diff_lo": lo, "diff_hi": hi, "result": bool(lo > 0),
                     "detail": f"two-variable {xE} + {xD} vs best single {best}; CI of R2(two) - R2(best)"})
    out = pd.DataFrame(rows)
    out["result"] = out["result"].map({True: "pass", False: "fail"})
    out.to_csv(path, index=False)
    return out


# ---------------------------------------------------------------- dry run and the one S-test opening

def dry_split(seed=config.SEED):
    """S-train trap graphs split 70 / 30 (fit, score), seed 573."""
    X, split = predictors()
    ids = np.sort(X.index[split == "train"].to_numpy())
    perm = np.random.default_rng(seed).permutation(ids)
    n_fit = int(round(0.7 * len(ids)))
    return np.sort(perm[:n_fit]), np.sort(perm[n_fit:])


def dry_run():
    """Every --open output on the stand-in (fit 70%, score 30% of S-train), prefix u5_dryrun_. Outcomes of the
    score part are recomputed with markov (the --open path, without allow_test) and checked against Unit 2b."""
    X, _ = predictors()
    d = train_outcomes()
    fit, hold = dry_split()
    d_score = compute_outcomes(hold)
    ref = d[d["atlas_id"].isin(hold)].sort_values(["atlas_id", "T"]).reset_index(drop=True)
    got = d_score.sort_values(["atlas_id", "T"]).reset_index(drop=True)
    cols = ["log_Yb", "log_Ya", "log_dur", "log_entries"]
    assert np.allclose(ref[cols], got[cols], rtol=1e-9)
    d_fit = d[d["atlas_id"].isin(fit)]
    rank, law, preds = run_scoring(X, d_fit, d_score,
                                   [config.U5_DRYRUN_RANKING, config.U5_DRYRUN_LAW, config.U5_DRYRUN_PREDICTIONS])
    add = addendum_scoring(X, d_fit, d_score, pd.read_csv(config.U5_TRAIN_FACTORS), config.U5_DRYRUN_ADDENDUM)
    return rank, law, preds, add


def check_unchanged(path, commit, root=config.ROOT):
    """commit is an ancestor of HEAD, path is in commit, and the working-tree file equals it; else SystemExit."""
    git = ["git", "-C", str(root)]
    rel = path.relative_to(root).as_posix()
    if subprocess.run(git + ["merge-base", "--is-ancestor", commit, "HEAD"], capture_output=True).returncode != 0:
        raise SystemExit(f"STOP: {commit} is not an ancestor of HEAD")
    if subprocess.run(git + ["cat-file", "-e", f"{commit}:{rel}"], capture_output=True).returncode != 0:
        raise SystemExit(f"STOP: {rel} is not in {commit}")
    if subprocess.run(git + ["diff", "--quiet", commit, "--", rel], capture_output=True).returncode != 0:
        raise SystemExit(f"STOP: {rel} differs from {commit}")


def check_log(log):
    """SystemExit if the graph-test access log already has a u5 entry."""
    if log.exists() and pd.read_csv(log)["caller"].astype(str).str.contains("u5_test").any():
        raise SystemExit(f"STOP: {log} already has a u5 entry; S-test is opened once")


def open_test(prereg=None, h_lock=None, expectation=None, h_exp=None, addendum=None, factors=None, h_add=None,
              log=None, cache=None, root=None):
    """The one S-test scoring. Checks first (pre-registration, expectation, addendum and its factors table
    unchanged since H_lock / H_exp / H_add; no u5 log entry); then S-test outcomes via markov (allow_test with
    REASON), cached; one log row; then run_scoring (pre-registered) and addendum_scoring (A1-A3).
    Arguments default to the config values (tests pass temporary copies)."""
    root = root or config.ROOT
    log = log or config.GRAPH_TEST_ACCESS_LOG
    cache = cache or config.U5_TEST_CACHE
    factors = factors or config.U5_TRAIN_FACTORS
    check_unchanged(prereg or config.U5_PREREG, h_lock or config.U5_H_LOCK, root)
    check_unchanged(expectation or config.U5_EXPECTATION, h_exp or config.U5_H_EXP, root)
    check_unchanged(addendum or config.U5_ADDENDUM, h_add or config.U5_H_ADD, root)
    check_unchanged(factors, h_add or config.U5_H_ADD, root)
    check_log(log)
    X, split = predictors()
    test_ids = np.sort(X.index[split == "test"].to_numpy())
    d_score = compute_outcomes(test_ids, allow_test=True, reason=REASON)
    cache.mkdir(parents=True, exist_ok=True)
    d_score.to_csv(cache / "outcomes.csv", index=False)
    row = pd.DataFrame([{"timestamp": datetime.now().isoformat(timespec="seconds"), "caller": CALLER,
                         "reason": REASON, "n_graphs": len(test_ids)}])
    row.to_csv(log, mode="a", header=not log.exists(), index=False)
    d_fit = train_outcomes()
    rank, law, preds = run_scoring(X, d_fit, d_score,
                                   [config.U5_TEST_RANKING, config.U5_TEST_LAW, config.U5_TEST_PREDICTIONS])
    add = addendum_scoring(X, d_fit, d_score, pd.read_csv(factors), config.U5_TEST_ADDENDUM)
    return rank, law, preds, add
