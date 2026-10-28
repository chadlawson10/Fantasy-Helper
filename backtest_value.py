"""
Backtest for the composite player value metric.

For past seasons, at several mid-season cutoffs, compute every component using ONLY
data available at that point, then check how well each weighting ranks players by
how they actually scored for the rest of the season.

Evaluation is *within position* (rank correlation of the composite vs. rest-of-season
points per game), because that is where the weighting choices matter; cross-position
comparison is handled separately by the FP+ / value-over-replacement scaling.

Run:  python backtest_value.py
"""

import numpy as np
import pandas as pd
import nflreadpy as nfl

SEASONS = [2022, 2023, 2024, 2025]
CUTOFFS = [4, 6, 8, 10, 12]
POS = ["QB", "RB", "WR", "TE"]
RECENT_W = [0.4, 0.3, 0.2, 0.1]
LAST_WEEK = 18
QUALIFY = 0.5
TEAMS, STARTERS, FLEX = 12, {"QB": 1, "RB": 2, "WR": 2, "TE": 1}, 1
FLEX_SHARE = {"QB": 0.0, "RB": 0.4, "WR": 0.5, "TE": 0.1}
PPR = "fantasy_points_ppr"

COMPONENTS = ["fp", "vorp", "tgt", "rush", "sos"]


def load():
    seasons = [min(SEASONS) - 1] + SEASONS
    w = nfl.load_player_stats(seasons).to_pandas()
    w = w[(w["season_type"] == "REG") & w["position"].isin(POS)].copy()
    team_carries = w.groupby(["season", "week", "team"])["carries"].transform("sum")
    w["rush_share"] = np.where(team_carries > 0, w["carries"] / team_carries, 0.0)
    sched = nfl.load_schedules(seasons).to_pandas()
    sched = sched[sched["game_type"] == "REG"]
    return w, sched


def _weighted(df, col):
    g = df.groupby("player_id")
    return (g.apply(lambda x: (x["wt"] * x[col]).sum() / x["wt"].sum(), include_groups=False))


def features(w, sched, season, cutoff, k):
    """One row per player: components as of `cutoff`, plus the rest-of-season target."""
    past = w[(w["season"] < season) | ((w["season"] == season) & (w["week"] <= cutoff))]
    cur = past[past["season"] == season]
    fut = w[(w["season"] == season) & (w["week"] > cutoff) & (w["week"] <= LAST_WEEK)]

    n = cur.groupby("player_id").size().rename("n")
    ppg = cur.groupby("player_id")[PPR].mean().rename("ppg")

    ps = past.sort_values(["season", "week"], ascending=False).copy()
    ps["k"] = ps.groupby("player_id").cumcount()
    win = ps[ps["k"] < len(RECENT_W)].copy()
    win["wt"] = win["k"].map(dict(enumerate(RECENT_W)))
    base = _weighted(win, PPR).rename("base")
    tgt = _weighted(win, "target_share").rename("tgt")
    rush = _weighted(win, "rush_share").rename("rush")

    latest = ps[ps["k"] == 0].set_index("player_id")[["position", "team", "week", "season"]]
    df = latest.join([n, ppg, base, tgt, rush])
    df["n"] = df["n"].fillna(0)
    df = df[(df["season"] == season) & (df["week"] >= cutoff - 2) & (df["n"] >= 1)].copy()

    target = fut.groupby("player_id")[PPR].agg(["mean", "size"])
    df = df.join(target.rename(columns={"mean": "ros_ppg", "size": "ros_n"}))
    df = df[df["ros_n"] >= 3].copy()
    pos = df["position"]

    # Qualified players define each position's average (as in the app)
    min_games = max(1, int(np.ceil(QUALIFY * cutoff)))
    q = df["n"] >= min_games
    pos_avg = df.loc[q].groupby("position")["ppg"].mean()
    df["pos_avg"] = pos.map(pos_avg)

    # Small-sample shrinkage toward the position average
    wprod = df["n"] / (df["n"] + k) if k > 0 else 1.0
    df["ppg_s"] = df["pos_avg"] + (df["ppg"] - df["pos_avg"]) * wprod
    df["fp"] = 100 * df["ppg_s"] / df["pos_avg"]

    # Replacement level by position, from the stabilised rates of active players
    repl = {}
    for p in POS:
        flex = FLEX_SHARE[p] * FLEX
        rank = max(1, int(round(TEAMS * (STARTERS.get(p, 0) + flex))))
        vals = df.loc[pos == p, "ppg_s"].sort_values(ascending=False)
        repl[p] = vals.iloc[min(rank, len(vals)) - 1] if len(vals) else 0.0
    df["vorp"] = (df["ppg_s"] - pos.map(repl)).clip(lower=0)

    # Usage shares (shrunk with half the prior, since they stabilise faster)
    wuse = df["n"] / (df["n"] + k / 2) if k > 0 else 1.0
    for col in ("tgt", "rush"):
        m = df.loc[q].groupby("position")[col].mean()
        df[col] = pos.map(m) + (df[col] - pos.map(m)) * wuse
    df.loc[pos == "QB", "tgt"] = np.nan
    df.loc[pos != "RB", "rush"] = np.nan

    # Rest-of-season schedule: how much remaining opponents allow to this position
    allowed = past.groupby(["opponent_team", "position", "season", "week"])[PPR].sum().reset_index()
    per_def = allowed.groupby(["opponent_team", "position"])[PPR].mean()
    league = allowed.groupby("position")[PPR].mean()
    dfac = per_def / per_def.index.get_level_values("position").map(league)
    rem = sched[(sched["season"] == season) & (sched["week"] > cutoff)]
    long = pd.concat([rem[["home_team", "away_team"]].set_axis(["team", "opp"], axis=1),
                      rem[["away_team", "home_team"]].set_axis(["team", "opp"], axis=1)])
    sos = {}
    for p in POS:
        f = dfac.reindex(pd.MultiIndex.from_arrays([long["opp"], [p] * len(long)])).to_numpy()
        sos[p] = pd.Series(np.where(np.isnan(f), 1.0, f), index=long["team"].values).groupby(level=0).mean()
    df["sos"] = [sos[p].get(t, 1.0) for p, t in zip(df["position"], df["team"])]

    # Standardise (z-scores among qualified players), exactly as the app does
    qq = df["n"] >= min_games
    for c in ("fp", "vorp"):  # pooled across positions: these are cross-position scales
        mu, sd = df.loc[qq, c].mean(), df.loc[qq, c].std()
        df["z_" + c] = (df[c] - mu) / sd
    for c in ("tgt", "rush", "sos"):  # within position
        g = df.loc[qq].groupby("position")[c]
        mu, sd = df["position"].map(g.mean()), df["position"].map(g.std())
        df["z_" + c] = (df[c] - mu) / sd
    df["season"], df["cutoff"] = season, cutoff
    return df.reset_index()


def composite(df, weights):
    """Weighted z-score; each player's weights are renormalised over the components he has."""
    num = np.zeros(len(df))
    den = np.zeros(len(df))
    for c in COMPONENTS:
        wt = weights.get(c, 0.0)
        z = df["z_" + c].to_numpy(dtype=float)
        ok = ~np.isnan(z) & (wt > 0)
        num += np.where(ok, wt * np.nan_to_num(z), 0.0)
        den += np.where(ok, wt, 0.0)
    return pd.Series(np.where(den > 0, num / np.where(den > 0, den, 1), np.nan), index=df.index)


def within_pos_spearman(df, score):
    """Average (over positions) of the rank correlation between score and ROS points/game."""
    out = {}
    for p in POS:
        m = df["position"] == p
        s, y = score[m], df.loc[m, "ros_ppg"]
        if m.sum() > 10:
            out[p] = s.rank().corr(y.rank())
    return out


SCHEMES = {
    "FP+ only, raw (current app)": ("raw", {"fp": 1}),
    "FP+ only, stabilised": (None, {"fp": 1}),
    "Production only (FP+ 50 / VORP 50)": (None, {"fp": .5, "vorp": .5}),
    "Your proposal (30/30/15/15/10)": (None, {"fp": .30, "vorp": .30, "tgt": .15, "rush": .15, "sos": .10}),
    "Proposal without schedule": (None, {"fp": .35, "vorp": .35, "tgt": .15, "rush": .15}),
    "Usage-heavier (25/25/20/20/10)": (None, {"fp": .25, "vorp": .25, "tgt": .20, "rush": .20, "sos": .10}),
    "Usage-heavy (20/20/25/25/10)": (None, {"fp": .20, "vorp": .20, "tgt": .25, "rush": .25, "sos": .10}),
}


def run(k=4, verbose=True):
    w, sched = load()
    frames, raw_frames = [], []
    for s in SEASONS:
        for c in CUTOFFS:
            frames.append(features(w, sched, s, c, k))
            raw_frames.append(features(w, sched, s, c, 0))
    data, raw = pd.concat(frames, ignore_index=True), pd.concat(raw_frames, ignore_index=True)

    rows = []
    for name, (mode, weights) in SCHEMES.items():
        d = raw if mode == "raw" else data
        res = {"scheme": name}
        scores = composite(d, weights)
        per_pos = within_pos_spearman(d, scores)
        res.update(per_pos)
        res["avg"] = np.mean(list(per_pos.values()))
        for label, cs in (("early", [4, 6]), ("late", [10, 12])):
            sub = d["cutoff"].isin(cs)
            e = within_pos_spearman(d[sub], scores[sub])
            res[label] = np.mean(list(e.values()))
        rows.append(res)
    table = pd.DataFrame(rows).set_index("scheme").round(3)
    return table, data, raw


def shrinkage_sweep(ks=(0, 2, 4, 6, 8, 12)):
    w, sched = load()
    out = []
    for k in ks:
        d = pd.concat([features(w, sched, s, c, k) for s in SEASONS for c in CUTOFFS], ignore_index=True)
        sp = within_pos_spearman(d, composite(d, {"fp": 1}))
        early = d["cutoff"].isin([4, 6])
        spe = within_pos_spearman(d[early], composite(d, {"fp": 1})[early])
        out.append({"k": k, "avg": np.mean(list(sp.values())), "early_cutoffs": np.mean(list(spe.values()))})
    return pd.DataFrame(out).round(3).set_index("k")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    table, data, raw = run(k=4)
    print("\nWithin-position rank correlation with rest-of-season points/game")
    print("(higher is better; 'early' = cutoffs after weeks 4 and 6, 'late' = weeks 10 and 12)\n")
    print(table)
    print("\nShrinkage sweep (FP+ only): games of prior vs accuracy")
    print(shrinkage_sweep())
    zc = ["z_fp", "z_vorp", "z_tgt", "z_rush", "z_sos"]
    print("\nCorrelation between components:")
    print(data[zc].corr().round(2))
