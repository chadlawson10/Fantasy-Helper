"""
Train the fitted projection model and write model_coefs.json.

What it learns: expected rest-of-season fantasy points per game for each player, from
  season average (shrunk for small samples), recent form, target share, rush share,
  expected fantasy points (opportunity), snap share and last season's production.

How it works: one regularised linear (ridge) model per position and scoring format,
trained on 2022-2025. Inputs are computed with the SAME functions the live app uses
(start_sit_core.build_pool / model_features), so training and serving can't drift apart.
Everything is validated leave-one-season-out: train on three seasons, test on the fourth.

Run once a year (after the season) or whenever you want to refresh:   python train_model.py
"""

import json
import pickle
import sys
import time
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import start_sit_core as core

warnings.filterwarnings("ignore")

SEASONS = [2022, 2023, 2024, 2025]
WEEKS = list(range(4, 16))                  # projection weeks used for training samples
SCORINGS = ["PPR", "Half PPR", "Standard"]
POS = core.POSITIONS
FEATURES = ["ppg_s", "base", "tgt", "rush", "xfp_s", "snap", "prev"]
LAMBDAS = [5.0, 20.0, 80.0]
STARTERS, FLEX_POS, TEAMS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}, ("RB", "WR", "TE"), 12
CACHE = Path("/tmp/model_samples.pkl")
OUT = Path(__file__).with_name("model_coefs.json")


# --------------------------------- Dataset ----------------------------------
def build_samples():
    """One pool per (season, scoring, week), plus the rows used to fit the model."""
    pools, frames = {}, []
    for s in SEASONS:
        weekly, sched, inj, _ = core.load_data(s)
        opp, snaps = core.load_model_data(s)
        for sc in SCORINGS:
            col = core.SCORING_COLUMNS[sc]
            for w in WEEKS:
                pool, _ = core.build_pool(weekly, sched, inj, s, scoring=sc, week=w, opp=opp, snaps=snaps)
                # Hand-built composite (for an apples-to-apples comparison later)
                lv, rk = core.replacement_levels(pool, TEAMS, STARTERS, 1, FLEX_POS)
                full = core.add_values(pool, lv)
                full, _, _ = core.add_fp_plus(full, w, "Season average", 0.5)
                full, _ = core.add_composite_value(full, rk, core.RECOMMENDED_WEIGHTS, 2.0)

                fut = weekly[(weekly["season"] == s) & (weekly["week"] >= w) & (weekly["week"] <= 18)] \
                    .groupby("player_id")[col].agg(["mean", "size"])
                this = weekly[(weekly["season"] == s) & (weekly["week"] == w)] \
                    .drop_duplicates("player_id").set_index("player_id")[col]
                full["ros_ppg"] = full["player_id"].map(fut["mean"])
                full["ros_n"] = full["player_id"].map(fut["size"])
                full["wk_pts"] = full["player_id"].map(this)

                feats = core.model_features(full, w)       # on the FULL pool, exactly as in the app
                keep = (full["recent"] & full["current"] & (full["season_games"] >= 1)
                        & (full["ros_n"] >= 3)).to_numpy()
                df = feats[keep].copy()
                for c in ("player_id", "pos", "ros_ppg", "wk_pts", "base", "matchup", "inj_mult"):
                    df[c] = full.loc[keep, c].to_numpy()
                df["season"], df["week"], df["scoring"] = s, w, sc
                frames.append(df)
                pools[(s, sc, w)] = (full, rk)
        print(f"  built season {s}", flush=True)
    return pd.concat(frames, ignore_index=True), pools


# ----------------------------------- Model ----------------------------------
def fit_ridge(X, y, lam):
    mu, sd = X.mean(), X.std().replace(0, 1.0)
    Z = ((X - mu) / sd).to_numpy()
    b = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - y.mean()).to_numpy())
    return mu, sd, b, float(y.mean())


def fit_position(df, pos, lam):
    d = df[df["pos"] == pos]
    feats = [f for f in FEATURES if d[f].notna().any()]
    impute = d[feats].mean()
    mu, sd, b, icpt = fit_ridge(d[feats].fillna(impute), d["ros_ppg"], lam)
    return {"features": feats, "impute": impute.to_dict(), "mean": mu.to_dict(), "sd": sd.to_dict(),
            "coef": dict(zip(feats, map(float, b))), "intercept": icpt}


def predict_df(m, d):
    X = d[m["features"]].fillna(pd.Series(m["impute"]))
    z = (X - pd.Series(m["mean"])) / pd.Series(m["sd"])
    return (m["intercept"] + z @ pd.Series(m["coef"])).clip(lower=0)


def rank_corr(a, b):
    return pd.Series(np.asarray(a)).rank().corr(pd.Series(np.asarray(b)).rank())


def r2(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def loso(df, scoring, lam):
    """Leave-one-season-out predictions + per-fold models."""
    d = df[df["scoring"] == scoring]
    preds, base_preds, folds = pd.Series(np.nan, index=d.index), pd.Series(np.nan, index=d.index), {}
    for hold in SEASONS:
        tr, te = d[d["season"] != hold], d[d["season"] == hold]
        folds[hold] = {p: fit_position(tr, p, lam) for p in POS}
        for p in POS:
            idx = te.index[te["pos"] == p]
            preds[idx] = predict_df(folds[hold][p], te.loc[idx])
            # baseline: season average only, same ridge machinery
            b = tr[tr["pos"] == p]
            mu, sd, coef, icpt = fit_ridge(b[["ppg_s"]], b["ros_ppg"], lam)
            base_preds[idx] = ((te.loc[idx, ["ppg_s"]] - mu) / sd).to_numpy() @ coef + icpt
    return d, preds, base_preds, folds


def summarise(d, preds, base_preds):
    out = {"rank_corr": {}, "baseline_rank_corr": {}, "r2": {}, "baseline_r2": {}}
    for p in POS:
        rows = d["pos"] == p
        rc, brc, rr, brr = [], [], [], []
        for s in SEASONS:
            m = rows & (d["season"] == s)
            rc.append(rank_corr(preds[m], d.loc[m, "ros_ppg"]))
            brc.append(rank_corr(base_preds[m], d.loc[m, "ros_ppg"]))
            rr.append(r2(d.loc[m, "ros_ppg"], preds[m]))
            brr.append(r2(d.loc[m, "ros_ppg"], base_preds[m]))
        out["rank_corr"][p], out["baseline_rank_corr"][p] = float(np.mean(rc)), float(np.mean(brc))
        out["r2"][p], out["baseline_r2"][p] = float(np.mean(rr)), float(np.mean(brr))
    for k in list(out):
        out[k]["avg"] = float(np.mean([out[k][p] for p in POS]))
    return out


# ------------------------------------ Main ----------------------------------
def main():
    t0 = time.time()
    if CACHE.exists() and "--rebuild" not in sys.argv:
        df, pools = pickle.load(open(CACHE, "rb")); print("Loaded cached samples")
    else:
        print("Building samples (about a minute or two)...")
        df, pools = build_samples()
        pickle.dump((df, pools), open(CACHE, "wb"))
    print(f"{len(df):,} training rows, {time.time() - t0:.0f}s")

    # Choose the regularisation strength on PPR, out of sample
    best_lam, best = None, -1
    for lam in LAMBDAS:
        d, pr, bp, _ = loso(df, "PPR", lam)
        s = summarise(d, pr, bp)["r2"]["avg"]
        print(f"  lambda {lam:>5}: out-of-sample R2 {s:.4f}")
        if s > best:
            best, best_lam = s, lam
    print(f"Using lambda = {best_lam}")

    result = {"version": 1, "trained": str(date.today()), "seasons": SEASONS, "lambda": best_lam,
              "shrink_k": core.MODEL_K, "metrics": {}, "models": {}}
    for sc in SCORINGS:
        d, pr, bp, folds = loso(df, sc, best_lam)
        result["metrics"][sc] = summarise(d, pr, bp)
        resid = (d["ros_ppg"] - pr)
        final = {p: fit_position(d, p, best_lam) for p in POS}
        for p in POS:
            final[p]["resid_sd"] = float(resid[d["pos"] == p].std())   # honest (out-of-sample) error
        result["models"][sc] = final

    OUT.write_text(json.dumps(result, indent=1))
    print(f"\nWrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")

    # Report
    for sc in SCORINGS:
        m = result["metrics"][sc]
        print(f"\n{sc}: out-of-sample, within-position (avg of {len(SEASONS)} held-out seasons)")
        print(pd.DataFrame({"model rank corr": m["rank_corr"], "season-avg rank corr": m["baseline_rank_corr"],
                            "model R2": m["r2"], "season-avg R2": m["baseline_r2"]}).round(3))
    print("\nStandardised coefficients (PPR):")
    print(pd.DataFrame({p: pd.Series(result["models"]["PPR"][p]["coef"]) for p in POS}).round(2))
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
