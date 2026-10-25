"""HW 6, H5 (obdurate splits): committed splits, their spells, and who breaks them.

A committed split is a row with mixed signs and every |s_i| >= cutoff (see fraught.committed_mask).
"""
import numpy as np
import pandas as pd

from slider import config
from slider import slider_logs as sl
from slider.fraught import committed_mask, last_to_reach, round_end, sign_labels, spell_breaker, spells
from slider.stats import group_bootstrap_mean, kaplan_meier

AFTER = ["consensus", "relapse", "unresolved", "censored"]
TOPOLOGIES = ["Full", "Hierarchy", "Star"]


def committed_spells(runs, table, cutoff=0.9, tau=1.0):
    """Committed-split spells of every run, and the % of each round spent in committed splits.

    Returns (sp, per_run).
    sp: one row per spell: file, wave, dataset, topology, the spells() columns, and
        k             minority size at the spell's last committed row
        breaker, breaker_status   from spell_breaker (only for spells that are not censored)
        minority      True if the breaker was on the minority side
        last_reacher, reach_status  from last_to_reach
    per_run: file, wave, dataset, topology, % committed (forward hold, out of t_end).
    """
    parts, rows = [], []
    for k, run in enumerate(runs):
        t = run["t"]
        s = sl.sign_correct(run)
        mask = committed_mask(s, cutoff)
        t_end = round_end(t, table["end"][k], table["session"][k])
        info = {"file": table["file"][k], "wave": table["wave"][k],
                "dataset": table["dataset"][k], "topology": table["topology"][k]}
        dt = np.diff(np.append(t, t_end))
        rows.append({**info, "% committed": 100 * np.sum(dt * mask) / t_end})

        sp = spells(t, mask, t_end, censored=not table["success"][k], tau=tau, labels=sign_labels(s))
        for spell in sp.to_dict("records"):
            last = s[spell["last_row"]]
            n_plus, n_minus = np.sum(last > 0), np.sum(last < 0)
            spell["k"] = min(n_plus, n_minus)
            if spell["after"] == "censored":
                breaker, status = None, "censored"
            else:
                breaker, status = spell_breaker(s, spell["last_row"], cutoff)
            spell["breaker"], spell["breaker_status"] = breaker, status
            spell["minority"] = (breaker is not None) and (min(n_plus, n_minus) == (n_plus if last[breaker] > 0 else n_minus))
            spell["last_reacher"], spell["reach_status"] = last_to_reach(s, spell["first_row"], cutoff)
            parts.append({**info, **spell})
    sp = pd.DataFrame(parts, columns=list(info) + list(sp.columns) + ["k", "breaker", "breaker_status", "minority",
                                                                      "last_reacher", "reach_status"])
    return sp, pd.DataFrame(rows)


def spell_summary(sp, by, keys):
    """For each value of column `by` (in the order of keys): spells, Kaplan-Meier median duration
    (escape = the spell ended; censored = still open when a failed round ended), and % of each after type."""
    rows = []
    for key in keys:
        d = sp[sp[by] == key]
        median = kaplan_meier(d["duration"], d["after"] != "censored")[1] if len(d) > 0 else np.nan
        row = {by: key, "spells": len(d), "KM median (s)": median}
        for a in AFTER:
            row[a + " %"] = 100 * np.mean(d["after"] == a) if len(d) > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index(by)


def topology_summary(sp, per_run):
    """Per topology: runs, mean % of round in committed splits, and spell_summary."""
    g = per_run.groupby("topology")["% committed"]
    out = pd.DataFrame({"runs": g.size(), "mean % committed": g.mean()}).reindex(TOPOLOGIES)
    return out.join(spell_summary(sp, "topology", TOPOLOGIES))


def breaker_summary(sp, n_boot=10000, seed=config.SEED):
    """Breakers of the spells that are not censored. Under the null the breaker is a random player,
    so P(minority) = k/7. Returns observed vs expected minority count and (observed - expected) / n
    with a 95% CI that resamples groups (log files)."""
    ended = sp[sp["after"] != "censored"]
    ok = ended[ended["breaker_status"] == "ok"]
    d = ok["minority"].astype(float) - ok["k"] / 7              # observed - expected, per spell
    low, high = group_bootstrap_mean(d, ok["file"], n_boot, seed)
    return {"spells ended": len(ended),
            "ambiguous": int((ended["breaker_status"] == "ambiguous").sum()),
            "no drop": int((ended["breaker_status"] == "no drop").sum()),
            "breaker spells (n)": len(ok),
            "groups": ok["file"].nunique(),
            "observed minority": int(ok["minority"].sum()),
            "expected minority (sum k/7)": (ok["k"] / 7).sum(),
            "observed share": ok["minority"].mean() if len(ok) > 0 else np.nan,
            "expected share": (ok["k"] / 7).mean() if len(ok) > 0 else np.nan,
            "(obs - exp) / n": d.mean() if len(ok) > 0 else np.nan,
            "CI low": low, "CI high": high}


def confound_summary(sp):
    """Share of breakers who were also the last player to reach the cutoff before the spell (null 1/7)."""
    ok = sp[(sp["breaker_status"] == "ok") & (sp["reach_status"] == "ok")]
    same = ok["breaker"] == ok["last_reacher"]
    return {"breaker spells with a clear last reacher (n)": len(ok),
            "breaker = last to reach": int(same.sum()),
            "share": same.mean() if len(ok) > 0 else np.nan,
            "null share": 1 / 7,
            "ambiguous last reacher": int(((sp["breaker_status"] == "ok") & (sp["reach_status"] != "ok")).sum())}


def sensitivity(runs, table, cutoffs=(0.8, 0.9, 1.0), taus=(0.5, 1.0, 2.0), n_boot=10000, seed=config.SEED):
    """Main numbers of parts 1 and 2 for every cutoff x tau: one row per (cutoff, tau, topology);
    the breaker columns are filled on the Full rows only."""
    rows = []
    for cutoff in cutoffs:
        for tau in taus:
            sp, per_run = committed_spells(runs, table, cutoff, tau)
            topo = topology_summary(sp, per_run)
            brk = breaker_summary(sp[sp["topology"] == "Full"], n_boot, seed)
            for topology in TOPOLOGIES:
                row = {"cutoff": cutoff, "tau": tau, "topology": topology, **topo.loc[topology].to_dict()}
                if topology == "Full":
                    row.update(brk)
                rows.append(row)
    return pd.DataFrame(rows)
