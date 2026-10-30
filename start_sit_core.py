"""
Core fantasy logic: projections, injuries, player value, lineups, team needs,
and trade evaluation. Pure functions that return data; no printing or UI.
"""

import nflreadpy as nfl
import numpy as np
import pandas as pd

POSITIONS = ["QB", "RB", "WR", "TE"]
DEFAULT_WEIGHTS = [0.4, 0.3, 0.2, 0.1]  # most recent game first

SCORING_COLUMNS = {
    "PPR": "fantasy_points_ppr",
    "Half PPR": "fantasy_points_half",
    "Standard": "fantasy_points",
}

# Expected fraction of normal output by official game status. "Questionable" is
# user-adjustable because it varies a lot (many questionable players suit up).
INJURY_FACTORS = {"Out": 0.0, "Doubtful": 0.1}

# Rough share of flex slots each position tends to fill (used only to estimate
# replacement level). Normalised over whichever positions are flex-eligible.
FLEX_WEIGHT = {"QB": 1.5, "RB": 0.4, "WR": 0.5, "TE": 0.1}

BAD_STATUSES = ("Out", "Doubtful")


# ------------------------------- Data loading -------------------------------
def load_data(season: int):
    """Load player stats (this + last season), schedule, and injury reports."""
    weekly = nfl.load_player_stats([season - 1, season]).to_pandas()
    weekly = weekly[
        (weekly["season_type"] == "REG") & (weekly["position"].isin(POSITIONS))
    ].copy()
    weekly["fantasy_points_half"] = (
        weekly["fantasy_points"] + weekly["fantasy_points_ppr"]
    ) / 2
    schedule = nfl.load_schedules([season]).to_pandas()
    try:
        injuries = nfl.load_injuries([season]).to_pandas()
    except Exception:  # e.g. no report published yet this season
        injuries = pd.DataFrame()
    try:  # current team + roster status, which picks up trades, signings and cuts
        rosters = nfl.load_rosters([season]).to_pandas()
        rosters = (rosters.sort_values("week").drop_duplicates("gsis_id", keep="last")
                   [["gsis_id", "team", "status"]])
    except Exception:
        rosters = pd.DataFrame(columns=["gsis_id", "team", "status"])
    return weekly, schedule, injuries, rosters


def next_week(weekly: pd.DataFrame, season: int) -> int:
    played = weekly[weekly["season"] == season]
    return int(played["week"].max()) + 1 if not played.empty else 1


def _reg(schedule):
    return schedule[schedule["game_type"] == "REG"] if "game_type" in schedule.columns else schedule


def current_week(schedule) -> int:
    """The first week that still has an unplayed game (so a week stays 'current' until Monday night ends)."""
    reg = _reg(schedule)
    pending = reg[reg["home_score"].isna()]
    return int(pending["week"].min()) if len(pending) else int(reg["week"].max())


def week_statuses(schedule) -> dict:
    """{week: 'Complete' | 'In progress' | 'Upcoming'} for every regular-season week."""
    out = {}
    for wk, g in _reg(schedule).groupby("week"):
        done = int(g["home_score"].notna().sum())
        out[int(wk)] = "Complete" if done == len(g) else "In progress" if done else "Upcoming"
    return out


# --------------------------------- Injuries ---------------------------------
def injury_report(injuries: pd.DataFrame, week: int):
    """
    Return (report keyed by gsis_id, report_week, is_stale).
    The report is 'stale' when the newest published report is for an earlier
    week than the one we're projecting (e.g. it's Tuesday and the new report
    isn't out yet).
    """
    empty = pd.DataFrame(columns=["injury_status", "injury_text"])
    if injuries.empty:
        return empty, None, False

    inj = injuries
    if "game_type" in inj.columns:
        inj = inj[inj["game_type"] == "REG"]
    if inj.empty:
        return empty, None, False

    if (inj["week"] == week).any():
        rep_week, stale = week, False
    else:
        rep_week, stale = int(inj["week"].max()), True
    inj = inj[inj["week"] == rep_week].drop_duplicates("gsis_id", keep="last").copy()

    def describe(r):
        status, injury = r["report_status"], r["report_primary_injury"]
        if pd.notna(status):
            text = status + (f" ({injury})" if pd.notna(injury) else "")
        elif str(r["practice_status"]).startswith("Did Not"):
            text = "Practice: DNP" + (f" ({injury})" if pd.notna(injury) else "")
        elif str(r["practice_status"]).startswith("Limited"):
            text = "Practice: Limited" + (f" ({injury})" if pd.notna(injury) else "")
        else:
            return pd.Series({"injury_status": None, "injury_text": ""})
        if stale:
            text = f"Wk{rep_week}: {text}"
        return pd.Series({"injury_status": status if pd.notna(status) else None,
                          "injury_text": text})

    out = inj.apply(describe, axis=1)
    out.index = inj["gsis_id"].values
    return out, rep_week, stale


# ----------------------------- Player pool / values -------------------------
def _defense_factors(weekly: pd.DataFrame, col: str) -> pd.Series:
    """Points allowed per position by each defense relative to league average."""
    allowed = (
        weekly.groupby(["opponent_team", "position", "season", "week"])[col]
        .sum().reset_index()
    )
    per_def = allowed.groupby(["opponent_team", "position"])[col].mean()
    league = allowed.groupby("position")[col].mean()
    return per_def / per_def.index.get_level_values("position").map(league)


def build_pool(
    weekly, schedule, injuries, season,
    scoring="PPR", matchup_strength=0.5, questionable_factor=0.85,
    schedule_strength=0.5, weights=None, min_games=2, week=None, rosters=None,
):
    """
    One row per player: form, the matchup for `week`, injury, games left.

    `week` is the week being projected (default: the current week). Form and defensive
    ratings use only games played BEFORE that week, so choosing an earlier week shows
    what the projections looked like then, and choosing a later week projects ahead.
    """
    weights = weights or DEFAULT_WEIGHTS
    col = SCORING_COLUMNS[scoring]
    cur_week = current_week(schedule)
    week = int(week) if week else cur_week

    actual_rows = weekly[(weekly["season"] == season) & (weekly["week"] == week)]
    weekly = weekly[(weekly["season"] < season) | (weekly["week"] < week)]  # as-of history
    cur_season = season if (weekly["season"] == season).any() else season - 1

    w = weekly.sort_values(["season", "week"], ascending=False).copy()
    w["n"] = w.groupby("player_id").cumcount()
    # Usage: share of the team's carries (targets already come with a target_share column)
    team_carries = w.groupby(["season", "week", "team"])["carries"].transform("sum")
    w["rush_share"] = np.where(team_carries > 0, w["carries"] / team_carries, 0.0)
    w["target_share"] = w["target_share"].fillna(0.0)

    # Recency-weighted average of last few games (vectorised)
    rec = w[w["n"] < len(weights)].copy()
    rec["wt"] = rec["n"].map(dict(enumerate(weights)))
    rec["wp"] = rec["wt"] * rec[col]
    rec["w_ts"] = rec["wt"] * rec["target_share"]
    rec["w_rs"] = rec["wt"] * rec["rush_share"]
    g = rec.groupby("player_id")
    base = (g["wp"].sum() / g["wt"].sum()).rename("base")
    tgt_share = (g["w_ts"].sum() / g["wt"].sum()).rename("target_share")
    rush_share = (g["w_rs"].sum() / g["wt"].sum()).rename("rush_share")

    last8 = w[w["n"] < 8].groupby("player_id")[col]
    floor = last8.quantile(0.2).rename("floor")
    ceiling = last8.quantile(0.8).rename("ceiling")
    games = w.groupby("player_id").size().rename("games")

    latest = (
        w[w["n"] == 0].set_index("player_id")[
            ["player_display_name", "position", "team", "season", "week"]
        ].rename(columns={"player_display_name": "player", "position": "pos",
                          "season": "last_season", "week": "last_week"})
    )
    cur = w[w["season"] == cur_season].groupby("player_id")[col]
    season_ppg = cur.mean().rename("season_ppg")
    season_games = cur.size().rename("season_games")

    pool = latest.join([base, floor, ceiling, games, season_ppg, season_games,
                        tgt_share, rush_share])
    pool["season_games"] = pool["season_games"].fillna(0).astype(int)
    pool["season_ppg"] = pool["season_ppg"].round(1)
    # Targets matter for QB-less positions only; carries matter for RBs only
    pool.loc[pool["pos"] == "QB", "target_share"] = np.nan
    pool.loc[pool["pos"] != "RB", "rush_share"] = np.nan
    pool = pool[pool["games"] >= min_games].copy()
    pool["base"] = pool["base"].fillna(0.0)

    # Upcoming opponent + matchup factor
    opp_map = {}
    for _, gm in schedule[schedule["week"] == week].iterrows():
        opp_map[gm["home_team"]] = gm["away_team"]
        opp_map[gm["away_team"]] = gm["home_team"]
    # Current team and roster status (trades, signings, cuts) matter for this week and later;
    # earlier weeks keep the team the player was actually on then.
    pool["roster_status"] = ""
    if rosters is not None and len(rosters) and week >= cur_week:
        ros = rosters.drop_duplicates("gsis_id").set_index("gsis_id")
        status = pool.index.map(ros["status"]).to_series(index=pool.index).fillna("")
        new_team = pool.index.map(ros["team"]).to_series(index=pool.index)
        move = status.isin(["ACT", "RES", "INA", "DEV"]) & new_team.notna()
        pool.loc[move, "team"] = new_team[move]
        pool["roster_status"] = status

    pool["opp"] = pool["team"].map(opp_map)
    on_bye = pool["opp"].isna()

    dfac = _defense_factors(weekly, col)
    idx = pd.MultiIndex.from_arrays([pool["opp"], pool["pos"]])
    raw = dfac.reindex(idx).to_numpy()
    raw = np.where(np.isnan(raw), 1.0, raw)
    pool["matchup"] = (1 + (raw - 1) * matchup_strength).round(2)

    # Injuries (only applied to projections when the report is current)
    inj, inj_week, stale = injury_report(injuries, week)
    pool = pool.join(inj)
    pool["injury_text"] = pool["injury_text"].fillna("")
    mult = pool["injury_status"].map(
        {**INJURY_FACTORS, "Questionable": questionable_factor}
    ).fillna(1.0)
    if stale:
        mult[:] = 1.0

    # Players who are on injured reserve, released or retired can't play, whatever the weekly report says
    gone = pool["roster_status"].isin(["RES", "CUT", "RET"])
    label = pool["roster_status"].map({"RES": "Reserve/IR", "CUT": "Released", "RET": "Retired"})
    no_text = gone & (pool["injury_text"] == "")
    pool.loc[no_text, "injury_text"] = label[no_text]
    pool.loc[gone & pool["injury_status"].isna(), "injury_status"] = "Out"
    mult[gone] = 0.0

    pool["proj"] = (pool["base"] * pool["matchup"] * mult).round(1)
    pool.loc[on_bye, "proj"] = 0.0
    pool["opp"] = pool["opp"].fillna("BYE")

    # Game status for this week, and actual points once the game is final
    final_teams = set()
    for _, gm in schedule[schedule["week"] == week].iterrows():
        if pd.notna(gm.get("home_score")):
            final_teams.update([gm["home_team"], gm["away_team"]])
    pool["game_status"] = np.where(pool["opp"] == "BYE", "BYE",
                                   np.where(pool["team"].isin(final_teams), "Final", "Upcoming"))
    pts = actual_rows.drop_duplicates("player_id").set_index("player_id")[col]
    pool["actual"] = pool.index.map(pts).to_series(index=pool.index)
    pool.loc[pool["game_status"] != "Final", "actual"] = np.nan
    pool.loc[(pool["game_status"] == "Final") & pool["actual"].isna(), "actual"] = 0.0
    pool["base"] = pool["base"].round(1)
    pool["floor"] = pool["floor"].round(1)
    pool["ceiling"] = pool["ceiling"].round(1)

    # Activity flags + positional rank among recently active players
    pool["current"] = pool["last_season"] == cur_season
    pool["recent"] = pool["current"] & (pool["last_week"] >= week - 4)
    pool["pos_rank"] = np.nan
    r = pool[pool["recent"]].groupby("pos")["base"].rank(ascending=False, method="first")
    pool.loc[r.index, "pos_rank"] = r

    # Regular-season games remaining for each player's team
    sched = schedule
    if "game_type" in sched.columns:
        sched = sched[sched["game_type"] == "REG"]
    sched = sched[sched["week"] >= week]
    counts = pd.concat([sched["home_team"], sched["away_team"]]).value_counts()
    pool["games_left"] = pool["team"].map(counts).fillna(0).astype(int)

    # Rest-of-season strength of schedule: average opponent-defense factor over
    # every remaining game, by position, dampened like the weekly matchup.
    long = pd.concat([
        sched[["home_team", "away_team"]].set_axis(["team", "opp"], axis=1),
        sched[["away_team", "home_team"]].set_axis(["team", "opp"], axis=1),
    ])
    sos_by_pos = {}
    for pos in POSITIONS:
        f = dfac.reindex(pd.MultiIndex.from_arrays([long["opp"], [pos] * len(long)])).to_numpy()
        f = np.where(np.isnan(f), 1.0, f)
        sos_by_pos[pos] = pd.Series(f, index=long["team"].values).groupby(level=0).mean()
    sos_raw = np.array([sos_by_pos[ps].get(t, 1.0) for ps, t in zip(pool["pos"], pool["team"])])
    pool["sos_raw"] = sos_raw  # undampened; used by the composite value
    pool["sos"] = (1 + (sos_raw - 1) * schedule_strength).round(2)
    pool["ros_pts"] = (pool["base"] * pool["sos"]).round(1)

    pool["label"] = pool["player"] + " (" + pool["pos"] + ", " + pool["team"] + ")"
    pool = pool.rename_axis("player_id").reset_index()
    meta = {"week": week, "injury_week": inj_week, "injury_stale": stale,
            "current_week": cur_week}
    return pool, meta


def _flex_shares(flex_positions):
    raw = {p: FLEX_WEIGHT[p] for p in flex_positions}
    tot = sum(raw.values()) or 1
    return {p: v / tot for p, v in raw.items()}


def replacement_levels(pool, teams, starters, flex_slots, flex_positions):
    """
    Replacement level = base output of the last 'startable' player at each
    position, i.e. who a manager could realistically pick up off waivers.
    Returns ({pos: level}, {pos: rank cutoff}).
    """
    shares = _flex_shares(flex_positions) if flex_slots else {}
    levels, ranks = {}, {}
    for pos in POSITIONS:
        n = teams * (starters.get(pos, 0) + flex_slots * shares.get(pos, 0))
        n = max(1, int(round(n)))
        cand = pool[(pool["pos"] == pos) & pool["recent"]].sort_values(
            "base", ascending=False)
        levels[pos] = float(cand["base"].iloc[min(n, len(cand)) - 1]) if len(cand) else 0.0
        ranks[pos] = n
    return levels, ranks


def add_values(pool, levels):
    """Value over replacement per week and over the rest of the season (schedule-adjusted)."""
    p = pool.copy()
    p["repl"] = p["pos"].map(levels)
    p["vorp_week"] = (p["ros_pts"] - p["repl"]).clip(lower=0).round(1)
    p["ros_value"] = (p["vorp_week"] * p["games_left"]).round(0)
    return p


# --------------------------------- Lineups ----------------------------------
def add_fillers(df, levels, starters, flex_slots, flex_positions):
    """
    Add replacement-level waiver players so every slot can be filled. This makes
    lineups comparable before/after a trade (e.g. a 2-for-1 leaves a hole).
    """
    rows = []
    for pos in POSITIONS:
        n = starters.get(pos, 0) + (flex_slots if pos in flex_positions else 0)
        for i in range(int(n)):
            rows.append(dict(
                player_id=f"FA-{pos}-{i}", player=f"Replacement {pos} (waivers)",
                pos=pos, team="FA", opp="—", base=levels[pos], proj=levels[pos],
                ros_pts=levels[pos], sos=1.0, trade_value=0.0,
                matchup=1.0, injury_text="", injury_status=None, vorp_week=0.0,
                ros_value=0.0, games_left=0, is_filler=True))
    out = pd.concat([df.assign(is_filler=False), pd.DataFrame(rows)], ignore_index=True)
    return out


def build_lineup(df, starters, flex_slots=0, flex_positions=("RB", "WR", "TE"),
                 value_col="proj", skip_bye=True):
    """Assign each player a slot: position, 'FLX', or 'BENCH'."""
    df = df.sort_values(value_col, ascending=False).copy()
    df["slot"] = "BENCH"
    ok = (df["opp"] != "BYE") if skip_bye else pd.Series(True, index=df.index)

    for pos, n in starters.items():
        elig = df[(df["pos"] == pos) & ok]
        df.loc[elig.head(n).index, "slot"] = pos

    pool = df[(df["slot"] == "BENCH") & (df["pos"].isin(flex_positions)) & ok]
    df.loc[pool.head(flex_slots).index, "slot"] = "FLX"
    return df


def lineup_total(lineup, value_col):
    return float(lineup.loc[lineup["slot"] != "BENCH", value_col].sum())


# ------------------------------- Team needs ---------------------------------
def _grade(row, levels, ranks):
    if row.get("is_filler"):
        return "Empty"
    if row["base"] <= levels[row["pos"]]:
        return "Below replacement"
    r, depth = row["pos_rank"], ranks[row["pos"]]
    if pd.isna(r):
        return "Not playing?"
    if r <= depth / 3:
        return "Strong"
    if r <= 2 * depth / 3:
        return "Solid"
    return "Weak"


def team_needs(pool, my_ids, levels, ranks, starters, flex_slots, flex_positions):
    """Grade each starting slot, find weak spots and tradeable depth."""
    roster = pool[pool["player_id"].isin(my_ids)]
    filled = add_fillers(roster, levels, starters, flex_slots, flex_positions)
    lu = build_lineup(filled, starters, flex_slots, flex_positions,
                      value_col="ros_pts", skip_bye=False)
    lu["grade"] = lu.apply(lambda r: _grade(r, levels, ranks), axis=1)
    lu["alert"] = ""
    lu.loc[lu["injury_status"].isin(BAD_STATUSES), "alert"] = "Injured"
    lu.loc[lu["opp"] == "BYE", "alert"] = "BYE this week"

    starting = lu[lu["slot"] != "BENCH"]
    bench = lu[(lu["slot"] == "BENCH") & (~lu["is_filler"])]

    order = {"Empty": 0, "Below replacement": 1, "Not playing?": 1, "Weak": 2}
    need_rows = starting[starting["grade"].isin(order) | (starting["alert"] == "Injured")]
    need_rows = need_rows.assign(_o=need_rows["grade"].map(order).fillna(3)).sort_values("_o")

    needs = []
    for _, r in need_rows.iterrows():
        rk = f" ({r['pos']}{int(r['pos_rank'])})" if pd.notna(r["pos_rank"]) else ""
        note = r["grade"] if r["grade"] in order else f"listed {r['injury_text']}"
        name = r["player"] if not r["is_filler"] else "No player rostered"
        needs.append(f"{r['slot']}: {name}{rk}, {note.lower()}")

    surplus_rows = bench[bench["vorp_week"] > 0].sort_values("ros_value", ascending=False)
    surplus = [f"{r['player']} ({r['pos']}), FP+ {int(r['fp_plus']) if pd.notna(r['fp_plus']) else '—'}, "
               f"about {r['vorp_week']:.1f} pts/wk above replacement"
               for _, r in surplus_rows.iterrows()]

    need_pos = sorted({r["pos"] for _, r in need_rows.iterrows()})
    surplus_pos = sorted(set(surplus_rows["pos"]))
    return {"starting": starting, "bench": bench, "needs": needs, "surplus": surplus,
            "need_pos": need_pos, "surplus_pos": surplus_pos}


# ---------------------------------- Trades ----------------------------------
def _roster_view(pool, ids, levels, starters, flex_slots, flex_positions):
    roster = pool[pool["player_id"].isin(ids)]
    filled = add_fillers(roster, levels, starters, flex_slots, flex_positions)
    wk = build_lineup(filled, starters, flex_slots, flex_positions, "proj", True)
    ros = build_lineup(filled, starters, flex_slots, flex_positions, "ros_pts", False)
    return {"week": lineup_total(wk, "proj"), "ros": lineup_total(ros, "ros_pts"),
            "ros_lineup": ros}


def _value_label(give_val, get_val):
    big = max(give_val, get_val)
    if big < 1e-9 or abs(get_val - give_val) <= 0.10 * big:
        return "even"
    return "you_win" if get_val > give_val else "they_win"


_LABEL_TEXT = {"you_win": "favors you", "even": "is even", "they_win": "favors the other side"}

FP_BASES = ("Season average", "Recent form")
TRADE_METHODS = ("Composite value", "FP+", "Value over replacement")


# ----------------------------------- FP+ index ------------------------------
def add_fp_plus(pool, week, basis="Season average", qualify_share=0.5):
    """
    FP+ works like OPS+ in baseball: 100 is exactly the average fantasy points
    per game for a player's position, 120 is 20% better than average, 80 is 20%
    worse.

    The position average is taken over *qualified* players only (those who have
    played at least `qualify_share` of the season's games so far), so backups
    who barely played don't drag the average down.

    basis = "Season average": each player's average over this season's games.
    basis = "Recent form":    the recency-weighted average of his last 4 games.

    Returns (pool with fp_plus, {pos: average}, minimum games to qualify).
    """
    p = pool.copy()
    value = p["season_ppg"] if basis == "Season average" else p["base"]
    value = value.fillna(p["base"])  # players with no games yet this season

    min_games = max(1, int(np.ceil(qualify_share * max(week - 1, 0))))
    p["qualified"] = p["current"] & (p["season_games"] >= min_games)

    q_avg = value[p["qualified"]].groupby(p.loc[p["qualified"], "pos"]).mean()
    all_avg = value.groupby(p["pos"]).mean()  # fallback if a position has no qualifiers
    p["pos_avg"] = p["pos"].map(q_avg).fillna(p["pos"].map(all_avg)).round(1)
    p["fp_value"] = value.round(1)
    p["fp_plus"] = (100 * value / p["pos_avg"]).round(0)

    averages = {pos: float(q_avg.get(pos, all_avg.get(pos, np.nan))) for pos in POSITIONS}
    return p, averages, min_games


def set_trade_value(pool, method="Composite value"):
    """Choose which number is used as a player's trade value."""
    p = pool.copy()
    if method == "Composite value" and "value_index" in p:
        p["trade_value"] = p["value_index"].fillna(0.0)
    elif method == "FP+":
        p["trade_value"] = p["fp_plus"].fillna(0.0)
    else:
        p["trade_value"] = p["ros_value"]
    return p


# ------------------------------- Composite value -----------------------------
COMPONENT_KEYS = ("fp", "vorp", "tgt", "rush", "sos")
COMPONENT_LABELS = {"fp": "FP+ (stabilised)", "vorp": "Value over replacement",
                    "tgt": "Target share", "rush": "Rush share", "sos": "Rest-of-season schedule"}

# Defaults come from a backtest over 2022-2025 (see backtest_value.py)
RECOMMENDED_WEIGHTS = {"fp": 0.31, "vorp": 0.26, "tgt": 0.20, "rush": 0.20, "sos": 0.03}
ORIGINAL_WEIGHTS = {"fp": 0.30, "vorp": 0.30, "tgt": 0.15, "rush": 0.15, "sos": 0.10}
DEFAULT_SHRINK_K = 2.0  # games of "prior"; backtest minimised error at about 2


def add_composite_value(pool, ranks, weights, shrink_k=DEFAULT_SHRINK_K):
    """
    One value number per player, built from five components:

      fp    FP+ (100 = position average), stabilised for small samples
      vorp  points per game above a waiver-wire replacement (no schedule adjustment)
      tgt   target share   (RB/WR/TE)
      rush  share of team carries (RB)
      sos   strength of the remaining schedule at his position

    Small samples are pulled toward the position average by n / (n + k), where n is
    games played this season (usage shares use k/2 since they stabilise faster).
    Each component is converted to a z-score among qualified players (FP+ and value
    over replacement pooled across positions, since they're cross-position scales;
    usage and schedule within position). The composite is a weighted average of the
    z-scores the player actually has (QBs have no usage shares, only RBs have rush
    share), so weights are renormalised per player. Finally it is mapped back onto
    the FP+ scale (100 = average qualified player) so it reads like FP+.

    Needs add_fp_plus() to have run first. Returns (pool, info).
    """
    p = pool.copy()
    k = float(shrink_k)
    n = p["season_games"].astype(float)
    w_prod = n / (n + k) if k > 0 else 1.0
    w_use = n / (n + k / 2) if k > 0 else 1.0
    q = p["qualified"]

    p["ppg_stab"] = p["pos_avg"] + (p["fp_value"] - p["pos_avg"]) * w_prod
    p["fp_stab"] = 100 * p["ppg_stab"] / p["pos_avg"]

    # Replacement level from the stabilised rates of recently active players
    repl = {}
    for pos in POSITIONS:
        vals = p.loc[(p["pos"] == pos) & p["recent"], "ppg_stab"].sort_values(ascending=False)
        r = ranks[pos]
        repl[pos] = float(vals.iloc[min(r, len(vals)) - 1]) if len(vals) else 0.0
    p["vorp_stab"] = (p["ppg_stab"] - p["pos"].map(repl)).clip(lower=0)

    for src, dst in (("target_share", "tgt_stab"), ("rush_share", "rush_stab")):
        mean = p["pos"].map(p.loc[q].groupby("pos")[src].mean())
        p[dst] = mean + (p[src] - mean) * w_use

    def z_pooled(col):
        v = p.loc[q, col]
        return (p[col] - v.mean()) / v.std()

    def z_within(col):
        g = p.loc[q].groupby("pos")[col]
        return (p[col] - p["pos"].map(g.mean())) / p["pos"].map(g.std())

    p["z_fp"], p["z_vorp"] = z_pooled("fp_stab"), z_pooled("vorp_stab")
    p["z_tgt"], p["z_rush"], p["z_sos"] = z_within("tgt_stab"), z_within("rush_stab"), z_within("sos_raw")

    num = np.zeros(len(p))
    den = np.zeros(len(p))
    for c in COMPONENT_KEYS:
        wt = float(weights.get(c, 0.0))
        z = p["z_" + c].to_numpy(dtype=float)
        ok = ~np.isnan(z) & (wt > 0)
        num += np.where(ok, wt * np.nan_to_num(z), 0.0)
        den += np.where(ok, wt, 0.0)
    p["composite_z"] = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)

    scale = float(p.loc[q, "fp_stab"].std())
    p["value_index"] = (100 + p["composite_z"] * scale).clip(lower=0).round(0).fillna(0.0)

    # Effective weights per position (after renormalising for unavailable components)
    avail = {"QB": ("fp", "vorp", "sos"), "RB": COMPONENT_KEYS,
             "WR": ("fp", "vorp", "tgt", "sos"), "TE": ("fp", "vorp", "tgt", "sos")}
    effective = {}
    for pos, comps in avail.items():
        tot = sum(weights.get(c, 0.0) for c in comps) or 1.0
        effective[pos] = {c: weights.get(c, 0.0) / tot for c in comps}
    info = {"scale": scale, "replacement": repl, "effective": effective, "shrink_k": k}
    return p, info


# ---------------------------------- Trades ----------------------------------
def evaluate_trade(pool, my_ids, give_ids, get_ids, levels,
                   starters, flex_slots, flex_positions):
    """Compare value and lineup strength before and after a proposed trade."""
    my_ids = set(my_ids)
    after_ids = (my_ids - set(give_ids)) | set(get_ids)
    args = (levels, starters, flex_slots, flex_positions)
    before = _roster_view(pool, my_ids, *args)
    after = _roster_view(pool, after_ids, *args)

    give = pool[pool["player_id"].isin(give_ids)]
    get = pool[pool["player_id"].isin(get_ids)]
    gv, tv = float(give["trade_value"].sum()), float(get["trade_value"].sum())
    big = max(gv, tv)
    value_label = _value_label(gv, tv)

    # Show both methods side by side so disagreements are visible
    fp = (float(give["fp_plus"].fillna(0).sum()), float(get["fp_plus"].fillna(0).sum()))
    vo = (float(give["ros_value"].sum()), float(get["ros_value"].sum()))
    fl, vl = _value_label(*fp), _value_label(*vo)
    notes = []
    if fl != vl:
        notes.append(f"FP+ {_LABEL_TEXT[fl]}, but value over replacement {_LABEL_TEXT[vl]}. "
                     "FP+ compares each player to his own position's average, so the gap "
                     "usually comes from positional scarcity.")

    d_ros = after["ros"] - before["ros"]
    lineup_label = "better" if d_ros > 0.25 else "worse" if d_ros < -0.25 else "same"

    headline = {
        ("you_win", "better"): ("success", "Strong trade for you"),
        ("you_win", "same"): ("success", "Favors you on value"),
        ("you_win", "worse"): ("warning", "You win on value, but your starting lineup gets weaker (a depth play)"),
        ("even", "better"): ("success", "Fair trade that improves your lineup"),
        ("even", "same"): ("info", "Fair trade with little lineup change"),
        ("even", "worse"): ("warning", "Even on value, but it weakens your lineup"),
        ("they_win", "better"): ("warning", "You give up more value, but it improves your lineup"),
        ("they_win", "same"): ("error", "Favors the other side"),
        ("they_win", "worse"): ("error", "Unfavorable: you lose value and lineup strength"),
    }[(value_label, lineup_label)]

    if value_label == "you_win" and big > 0 and (tv - gv) > 0.25 * big:
        notes.append("This leans heavily your way, so the other manager may be unlikely to accept.")
    for _, r in pd.concat([give, get]).iterrows():
        if not r["qualified"] and r["season_games"] > 0:
            notes.append(f"{r['player']} has played only {int(r['season_games'])} game(s) this season, "
                         "so his FP+ rests on a small sample.")
    for _, r in get.iterrows():
        if r["injury_status"] in BAD_STATUSES or (
                r["injury_text"].startswith("Wk") and any(x in r["injury_text"] for x in BAD_STATUSES)):
            notes.append(f"{r['player']}: {r['injury_text']}")
        elif not r["recent"]:
            notes.append(f"{r['player']} hasn't played in 4+ weeks; check whether he's injured or inactive.")
        if r["opp"] == "BYE":
            notes.append(f"{r['player']} is on a bye this week.")

    return {"before": before, "after": after, "give": give, "get": get,
            "give_value": gv, "get_value": tv, "fp_values": fp, "vorp_values": vo,
            "headline": headline, "value_label": value_label,
            "lineup_label": lineup_label, "notes": notes}
