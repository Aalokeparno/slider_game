"""Path A P25, diagnosis of the build-check misses (no code changed; run once on Oct 30, 2026).

Part 1 (M2' lab v09 3/6): the same simulation as notebooks/path1/p25_final.ipynb (G6train, final, R = 2, seed 573);
the lab-v09 M2' refit without and with each group copy's drawn u as a known offset; old-CL q_s sampling SEs.
Part 2 (old CL q_s): share of "resume" halt exits whose next own step goes against the step before the halt
(in the simulator a resume never changes direction, so these are reversals counted as resumes).

    python scripts/p25_diagnose.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy.special import expit, logit
from slider import config, design, fit, frailty, halts, hazard, interface, motion, sim
from slider import slider_logs as sl
pd.set_option("display.width", 220)

G6 = design.design("G6train")
rounds = design.simulate_rounds(G6, "final", R=2, seed=573)
u = {(f"sim{r}_{row.group}"): out[r]["u"] for row, (out, _) in zip(G6.itertuples(), rounds) for r in range(2)}
runs = [run for r in range(2) for run in design.copy_runs(G6, rounds, r, 573)]
ht, _ = hazard.build_hazard_table(runs, b=0.25, m=2)
hs = halts.add_ticks(motion.motion_table(runs)[1])
ht["u"] = ht["group_id"].map(u)
assert ht["u"].notna().all()

# ---- M2' lab v09: plain refit (as the notebook), refit with the drawn u as a known offset
TRUE = sim.coefs("M2'"); true09 = {**TRUE, "const": TRUE["const"] + TRUE["lab"]}
v09 = ht[ht["interface"] == "lab_v09"]
X = fit.design(v09, frailty.M2P, False).drop(columns="lab")
yv = v09["y_rev"].astype(float)
rows = []
for name, off in [("plain (notebook)", None), ("offset = drawn u_g", v09["u"].to_numpy())]:
    m = sm.Logit(yv, X, offset=off).fit(disp=0)
    for term, b in m.params.items():
        se = m.bse[term]
        rows.append({"refit": name, "term": term, "true": true09[term], "recovered": b, "SE": se,
                     "z": (b - true09[term]) / se, "pass": abs(b - true09[term]) <= 2 * se})
r = pd.DataFrame(rows)
print(r.round(4).to_string(index=False))
print(r.groupby("refit", sort=False)["pass"].agg(["sum", "size"]))
w = v09.groupby("group_id").agg(bins=("u", "size"), rev=("y_rev", "sum"), u=("u", "first"))
print(f"lab v09 copies: {len(w)}; mean u {w['u'].mean():.4f} (SE {w['u'].std(ddof=1)/np.sqrt(len(w)):.4f}); "
      f"bin-weighted mean u {np.average(w['u'], weights=w['bins']):.4f}; sigma^2/2 = {sim.sigma_g()**2/2:.4f}")
# frailty-induced duration dependence: corr of u with mean time_since_rev across copies
v09g = v09.groupby("group_id").agg(u=("u", "first"), r=("time_since_rev_log1p", "mean"), e=("time_at_end_log1p", "mean"))
print("corr over copies: u vs mean log1p time_since_rev", round(v09g["u"].corr(v09g["r"]), 3),
      "| u vs mean log1p time_at_end", round(v09g["u"].corr(v09g["e"]), 3))

# ---- old CL q_s: sampling SE (observed information) vs the 2-group bootstrap CI
def info_se(d):
    n, e, w = halts.counts(d)
    r = halts.fit(n, e, w)
    x0 = logit([r["p"], r["q_f"], r["q_s"]])
    f = lambda x: -halts.loglik(n, e, w, *expit(x))
    H = np.zeros((3, 3)); eps = 1e-4
    for i in range(3):
        for j in range(3):
            a = np.zeros(3); a[i] = eps; b = np.zeros(3); b[j] = eps
            H[i, j] = (f(x0 + a + b) - f(x0 + a - b) - f(x0 - a + b) + f(x0 - a - b)) / (4 * eps ** 2)
    cov = np.linalg.inv(H)
    pr = expit(x0); jac = pr * (1 - pr)                        # delta method to the probability scale
    return r, np.sqrt(np.diag(cov)) * jac
human = halts.load_halts()
fits = pd.read_csv(config.P25_HALTS)
for iface in motion.INTERFACES:
    rh, seh = info_se(human[human["interface"] == iface])
    rs, ses = info_se(hs[hs["interface"] == iface])
    t = fits[(fits.interface == iface) & (fits.parameter == "q_s")].iloc[0]
    print(f"{iface}: q_s human {rh['q_s']:.4f} (info SE {seh[2]:.4f}; bootstrap CI [{t.ci_low:.4f}, {t.ci_high:.4f}], "
          f"groups {t.groups}) | sim {rs['q_s']:.4f} (info SE {ses[2]:.4f}) | z = {(rs['q_s']-rh['q_s'])/np.hypot(seh[2], ses[2]):.2f}")
oc = human[human["interface"] == "old_cl"]
for f, d in oc.groupby("file"):
    r = halts.fit(*halts.counts(d)); print("  old CL group", f, {k: round(r[k], 4) for k in halts.PARAMS}, "halts", len(d))
for f, d in hs[hs["interface"] == "old_cl"].groupby("file"):
    r = halts.fit(*halts.counts(d)); print("  sim old CL copy", f, {k: round(r[k], 4) for k in halts.PARAMS}, "halts", len(d))

# ------------------------------------------------------------------ part 2
def flips(runs):
    files = interface.interface_table(runs).set_index("file")
    out = []
    for run in runs:
        iface = motion.round_iface(run, files.loc[run["file"]])
        if iface["interface"] != "old_cl":
            continue
        _, h = motion.motion_spells(run, iface)
        t = np.r_[0.0, run["t"]]
        s = np.vstack([np.zeros(7), sl.sign_correct(run)])
        for j, d in h[h["exit"] == "resume"].groupby("player"):
            ds = np.diff(s[:, j]); rows = np.flatnonzero(np.abs(ds) > 1e-9) + 1
            for t0, dur in zip(d["start"], d["duration"]):
                k = np.flatnonzero(np.isclose(t[rows], t0))[-1]
                out.append({"n": max(1, round(dur / iface["dt"])), "flip": np.sign(ds[rows[k] - 1]) != np.sign(ds[rows[k + 1] - 1])})
    return pd.DataFrame(out)

runs, table = sl.load_runs("behavior")          # G-train (P17 set)
hum = flips(runs)
G6 = design.design("G6train"); oc = G6[G6["interface"] == "old_cl"].reset_index(drop=True)
sims, _ = design.simulate_design(oc, "final", R=2, seed=573)
sim_flips = flips(sims)
for name, d in [("human", hum), ("sim final", sim_flips)]:
    long = d[d["n"] >= 10]
    print(f"{name}: resume exits {len(d)}, direction flips {d['flip'].mean():.3%}; among halts >= 10 ticks: "
          f"{len(long)} resumes, flips {long['flip'].mean():.1%}")
