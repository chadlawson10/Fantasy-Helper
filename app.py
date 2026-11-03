from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

try:
    from espn_import import fetch_league, friendly_error, load_crosswalk, map_league
    from trade_finder import describe_suggestion, suggest_trades
    from start_sit_core import (
        BAD_STATUSES,
        COMPONENT_KEYS,
        COMPONENT_LABELS,
        DEFAULT_SHRINK_K,
        ROLE_MIN_BOOST,
        ROLE_SHRINK,
        MODEL_WEIGHTS,
        FP_BASES,
        ORIGINAL_WEIGHTS,
        RECOMMENDED_WEIGHTS,
        SCORING_COLUMNS,
        TRADE_METHODS,
        active_free_agents,
        add_composite_value,
        add_fp_plus,
        apply_model,
        add_role_changes,
        add_values,
        apply_role_boost,
        build_lineup,
        build_pool,
        current_week,
        evaluate_trade,
        grade_projections,
        league_replacement_levels,
        load_data,
        load_model,
        load_model_data,
        model_weekly_projection,
        rank_free_agents,
        replacement_levels,
        set_trade_value,
        summarize_grades,
        team_needs,
        week_statuses,
    )
except ImportError as _exc:  # almost always one file in the repo is an older version than the others
    st.error(
        "**An app file is out of date.** `app.py` expects newer code than the other files in your "
        "repository have. Replace **start_sit_core.py**, **espn_import.py** and **trade_finder.py** with "
        "the latest versions (all in the same folder as app.py, with `model_coefs.json`), then reboot "
        f"the app.\n\nTechnical detail: `{_exc}`"
    )
    st.stop()

st.set_page_config(page_title="Fantasy Football Helper", page_icon="🏈", layout="wide")

SEASON = 2026  # update each year


# ------------------------------- Cached loaders -----------------------------
# Stats, schedule, injury reports and rosters are re-downloaded every 3 hours (or on demand with
# the Refresh button), so new injury reports, trades, signings and game results show up daily.
@st.cache_data(ttl=3 * 60 * 60, show_spinner="Loading NFL data...")
def get_data(season):
    weekly, schedule, injuries, rosters = load_data(season)
    opp, snaps = load_model_data(season)
    return weekly, schedule, injuries, rosters, opp, snaps, datetime.now(timezone.utc)


@st.cache_resource
def get_model():
    """Fitted projection model (None if model_coefs.json isn't in the repo)."""
    return load_model()


@st.cache_data(ttl=3 * 60 * 60, show_spinner="Crunching projections...")
def get_pool(season, week, scoring, matchup_strength, questionable_factor, schedule_strength):
    weekly, schedule, injuries, rosters, opp, snaps, _ = get_data(season)
    return build_pool(weekly, schedule, injuries, season, scoring=scoring,
                      matchup_strength=matchup_strength,
                      questionable_factor=questionable_factor,
                      schedule_strength=schedule_strength, week=week, rosters=rosters,
                      opp=opp, snaps=snaps)


@st.cache_data(ttl=24 * 60 * 60)
def get_crosswalk():
    return load_crosswalk()


@st.cache_data(ttl=3 * 60 * 60, show_spinner="Grading past projections...")
def get_grades(season, scoring, matchup_strength, questionable_factor, schedule_strength,
               use_model, use_role):
    """Re-create every finished week's projections and grade them against actual points."""
    weekly, schedule, injuries, _rosters, opp, snaps, _ = get_data(season)
    g = grade_projections(weekly, schedule, injuries, season, scoring,
                          get_model() if use_model else None, opp=opp, snaps=snaps,
                          matchup_strength=matchup_strength, questionable_factor=questionable_factor,
                          schedule_strength=schedule_strength, use_role=use_role)
    return g, (summarize_grades(g) if len(g) else None)


# League/settings widgets are keyed so an ESPN import can fill them in.
DEFAULTS = {"teams_n": 12, "n_qb": 1, "n_rb": 2, "n_wr": 2, "n_te": 1, "n_flex": 1,
            "superflex": False, "scoring": "PPR"}
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


def do_import():
    """Button callback: fetch the ESPN league and apply its settings."""
    st.session_state["import_error"] = None
    try:
        league = fetch_league(st.session_state.get("espn_league_id", "").strip(), SEASON,
                              st.session_state.get("espn_s2", "").strip(),
                              st.session_state.get("espn_swid", "").strip())
    except Exception as exc:
        st.session_state["import_error"] = friendly_error(exc)
        return
    cfg = league["settings"]
    st.session_state["league"] = league
    st.session_state.update({
        "teams_n": min(max(cfg["teams"], 6), 20), "n_qb": min(cfg["qb"], 3),
        "n_rb": min(cfg["rb"], 5), "n_wr": min(cfg["wr"], 5), "n_te": min(cfg["te"], 3),
        "n_flex": min(cfg["flex"], 4), "superflex": cfg["superflex"],
    })
    if cfg["scoring"] in SCORING_COLUMNS:
        st.session_state["scoring"] = cfg["scoring"]
    st.session_state["_loaded_team"] = None
    # Don't keep the cookies around once they've been used (unless asked, so refreshing is one click)
    if not st.session_state.get("keep_cookies"):
        st.session_state["espn_s2"] = ""
        st.session_state["espn_swid"] = ""


def clear_league():
    st.session_state["league"] = None
    st.session_state["_loaded_team"] = None


def matchup_icon(m, opp):
    if opp in ("BYE", "—"):
        return opp
    return "🟢 Good" if m >= 1.05 else "🔴 Tough" if m <= 0.95 else "⚪ Neutral"


def sched_icon(s):
    if s != s:  # NaN
        return "—"
    return "🟢 Easy" if s >= 1.03 else "🔴 Hard" if s <= 0.97 else "⚪ Avg"


# ------------------------------- Sidebar ------------------------------------
_weekly, _schedule, _inj, _ros, _opp, _snaps, fetched_at = get_data(SEASON)
model = get_model()
cur_week = current_week(_schedule)
statuses = week_statuses(_schedule)
all_weeks = sorted(statuses)
if st.session_state.get("week_sel") not in all_weeks:
    st.session_state["week_sel"] = cur_week
selected_week = st.sidebar.selectbox(
    "Week", all_weeks, key="week_sel",
    format_func=lambda w: f"Week {w} · {statuses[w]}" + (" (current)" if w == cur_week else ""),
    help="Past weeks show what the projections looked like then, next to the actual results. "
         "Future weeks project ahead using the latest data.")
st.sidebar.caption(f"Data loaded {fetched_at:%b %d, %I:%M %p} UTC. Refreshes every 3 hours.")
st.sidebar.button("Refresh data now", on_click=st.cache_data.clear, key="refresh_btn",
                  help="Re-download stats, injury reports and rosters right now.")

with st.sidebar.expander("Import ESPN league", expanded=not st.session_state.get("league")):
    st.text_input("League ID", key="espn_league_id",
                  help="The number after leagueId= in your ESPN league's URL.")
    st.text_input("espn_s2 cookie (private leagues)", key="espn_s2", type="password")
    st.text_input("SWID cookie (private leagues)", key="espn_swid", type="password")
    st.checkbox("Keep cookies for this session", key="keep_cookies",
                help="Lets you re-import after trades or waiver moves without pasting the cookies "
                     "again. They're held in memory for this browser session only.")
    st.button("Import league", on_click=do_import, key="import_btn")
    if st.session_state.get("import_error"):
        st.error(st.session_state["import_error"])
    _lg = st.session_state.get("league")
    if _lg:
        _c = _lg["settings"]
        st.success(f"Imported {_lg['league_name']}")
        st.caption(f"{_c['teams']} teams · QB {_c['qb']}, RB {_c['rb']}, WR {_c['wr']}, "
                   f"TE {_c['te']}, FLX {_c['flex']} · {_c['scoring']}. Check these against "
                   "your league and adjust below if anything is off.")
        for _n in _c["notes"]:
            st.caption("• " + _n)
        st.button("Clear imported league", on_click=clear_league, key="clear_btn")
    st.caption("Private league cookies are used once to fetch your data and then cleared. "
               "Only enter them on an app you run or trust. Public leagues need just the ID.")

st.sidebar.header("League settings")
scoring = st.sidebar.selectbox("Scoring", list(SCORING_COLUMNS), key="scoring")
teams = st.sidebar.number_input("Teams in league", 6, 20, key="teams_n",
                                help="Used to estimate replacement level (who's on waivers).")

st.sidebar.subheader("Lineup slots")
c1, c2 = st.sidebar.columns(2)
n_qb = c1.number_input("QB", 0, 3, key="n_qb")
n_rb = c2.number_input("RB", 0, 5, key="n_rb")
n_wr = c1.number_input("WR", 0, 5, key="n_wr")
n_te = c2.number_input("TE", 0, 3, key="n_te")
n_flex = c1.number_input("FLX", 0, 4, key="n_flex")
superflex = c2.checkbox("Superflex", key="superflex", help="Allow QBs in FLX")

st.sidebar.subheader("Model")
matchup_strength = st.sidebar.slider(
    "This week's matchup weight", 0.0, 1.0, 0.5, 0.1,
    help="0 ignores opponent strength; 1 applies it fully.")
schedule_strength = st.sidebar.slider(
    "Rest-of-season schedule weight", 0.0, 1.0, 0.5, 0.1,
    help="How much the strength of a player's remaining schedule adjusts his lineup value.")
questionable_factor = st.sidebar.slider(
    "Questionable = % of normal output", 0, 100, 85, 5,
    help="Applied only when the injury report is current. Out = 0%, Doubtful = 10%.") / 100

st.sidebar.subheader("Trade value")
trade_method = st.sidebar.selectbox(
    "Trade value method", list(TRADE_METHODS), key="method",
    help="FP+: 100 = average for the position (like OPS+). "
         "Value over replacement: points above a waiver-wire player over the games left.")
fp_basis = st.sidebar.selectbox(
    "FP+ is based on", list(FP_BASES), key="basis",
    help="Season average = fantasy points per game this season. "
         "Recent form = weighted average of the last 4 games.")
qualify_share = st.sidebar.slider(
    "Qualify for the average (% of games played)", 0, 100, 50, 10,
    help="Only players who have played at least this share of the season so far count "
         "toward each position's average, so backups don't drag it down.") / 100

use_real_repl = False
if st.session_state.get("league"):
    use_real_repl = st.sidebar.checkbox(
        "Use my league's real replacement level", value=True, key="use_real_repl",
        help="Replacement level = the best free agent actually available at each position in your "
             "imported league, instead of an estimate from league size.")

st.sidebar.subheader("Projections")
if model:
    use_model = st.sidebar.checkbox(
        "Use fitted projection model", value=True, key="use_model",
        help="A model trained on 2022-2025 that combines season average, recent form, usage, "
             "expected points, snap share and last season's production. Tested on held-out "
             "seasons it explains noticeably more of each week's scoring than recent form alone.")
else:
    use_model = False
    st.sidebar.caption("Fitted model file (model_coefs.json) not found, so hand-weighted "
                       "projections are being used.")

use_role = st.sidebar.checkbox(
    "Adjust for injured teammates (role changes)", value=True, key="use_role",
    help="When a teammate is out, his targets and carries go to the players left. Adds the expected "
         "extra points to this week's projection when the effect is at least half a point.")

weights, shrink_k, source = dict(RECOMMENDED_WEIGHTS), DEFAULT_SHRINK_K, "legacy"
if trade_method == "Composite value":
    presets = (["Fitted model (recommended)"] if use_model else []) + [
        "Hand-weighted (backtested)", "Your original proposal", "Custom"]
    preset = st.sidebar.selectbox(
        "Composite weights", presets, key="preset",
        help="'Fitted model' builds the composite from the model's projections. The hand-weighted "
             "options blend FP+, value over replacement, usage and schedule directly.")
    if preset == "Fitted model (recommended)":
        weights, source = dict(MODEL_WEIGHTS), "model"
    elif preset == "Your original proposal":
        weights = dict(ORIGINAL_WEIGHTS)
    elif preset == "Custom":
        defaults = MODEL_WEIGHTS if use_model else RECOMMENDED_WEIGHTS
        raw = {c: st.sidebar.slider(COMPONENT_LABELS[c], 0, 100, int(round(defaults[c] * 100)),
                                    key=f"w_{c}") for c in COMPONENT_KEYS}
        total = sum(raw.values())
        weights = {c: v / total for c, v in raw.items()} if total else dict(defaults)
        if use_model:
            source = "model" if st.sidebar.checkbox(
                "Use fitted projections for FP+ and value over replacement", value=True,
                key="custom_model") else "legacy"
        st.sidebar.caption("Weights are rescaled to add up to 100%.")
    if source == "legacy":
        shrink_k = st.sidebar.slider(
            "Small-sample shrinkage (games of prior)", 0.0, 6.0, DEFAULT_SHRINK_K, 0.5, key="shrink_k",
            help="Pulls players with few games toward their position's average. 0 = off. "
                 "Backtests found about 2 gives the most accurate values.")

starters = {"QB": n_qb, "RB": n_rb, "WR": n_wr, "TE": n_te}
flex_pos = ("QB", "RB", "WR", "TE") if superflex else ("RB", "WR", "TE")

# -------------------------------- Data --------------------------------------
pool, meta = get_pool(SEASON, selected_week, scoring, matchup_strength, questionable_factor,
                      schedule_strength)
week = meta["week"]
pool = apply_model(pool, week, model, scoring) if use_model else pool.assign(proj_ppg=np.nan, proj_sd=np.nan)
if use_model:
    pool = model_weekly_projection(pool)                                   # this week's projection
    pool["ros_pts"] = (pool["proj_ppg"].fillna(pool["base"]) * pool["sos"]).round(1)  # rest of season
pool = add_role_changes(pool, scoring)
if use_role:
    pool = apply_role_boost(pool)

# ESPN league rosters (needed early: they define who is a free agent and the real replacement level)
league = st.session_state.get("league")
team_map = map_league(league, get_crosswalk(), pool[pool["current"]]) if league else {}
rostered_ids = {i for v in team_map.values() for i in v["ids"]}

rank_basis = pool.assign(base=pool["proj_ppg"].fillna(pool["base"])) if use_model else pool
levels, ranks = replacement_levels(rank_basis, teams, starters, n_flex, flex_pos)
est_levels = dict(levels)
real_repl_ok = bool(use_real_repl and len(rostered_ids) >= teams * 6)
if real_repl_ok:
    levels = league_replacement_levels(pool, rostered_ids, est_levels)
elif use_real_repl:
    st.sidebar.warning("The imported league has too few matched players to find real free agents, "
                       "so estimated replacement levels are being used.")
pool = add_values(pool, levels)
pool, pos_avgs, min_games = add_fp_plus(pool, week, fp_basis, qualify_share)
pool, comp_info = add_composite_value(pool, ranks, weights, shrink_k, source=source)
pool = set_trade_value(pool, trade_method)

st.title("🏈 Fantasy Football Helper")

_last_stat_week = int(_weekly.loc[_weekly["season"] == SEASON, "week"].max())
st.caption(f"Week {week} is **{statuses[week].lower()}**. Player stats include games through "
           f"Week {_last_stat_week}; rosters, trades, signings and injury reports reflect the latest "
           "published data.")

if meta["injury_week"] is None:
    st.caption("No injury report available yet.")
elif meta["injury_stale"]:
    st.caption(f"Injury report shown is from Week {meta['injury_week']}; the Week {week} "
               "report isn't out yet. Statuses are shown for reference but not applied to projections.")
else:
    st.caption(f"Injury report: Week {meta['injury_week']} (applied to projections).")

searchable = pool[pool["current"]].sort_values("label")
label_to_id = dict(zip(searchable["label"], searchable["player_id"]))
id_to_label = {v: k for k, v in label_to_id.items()}

# ESPN league: pick your team and the roster fills in automatically
my_team = None
if league:
    tc1, tc2 = st.columns([4, 1], vertical_alignment="bottom")
    my_team = tc1.selectbox("Your ESPN team", list(team_map),
                            key=f"my_team_{league['league_name']}")
    reload_clicked = tc2.button("Reload roster")
    sig = (league["league_name"], my_team)
    if reload_clicked or st.session_state.get("_loaded_team") != sig:
        st.session_state["roster"] = [id_to_label[i] for i in team_map[my_team]["ids"]]
        st.session_state["_loaded_team"] = sig
    miss = team_map[my_team]["unmatched"]
    if miss:
        st.caption("Not found in this season's stats (so left out): " + ", ".join(miss) + ". "
                   "You can add them manually above if they belong in your lineup.")

picked = st.multiselect(
    "Your roster (type to search)", searchable["label"].tolist(), key="roster",
    placeholder="Start typing a player name...")

tab_ss, tab_needs, tab_trade, tab_ideas, tab_waiver, tab_track = st.tabs(
    ["Start/Sit", "Team needs", "Trade analyzer", "Trade ideas", "Waiver wire", "Accuracy tracker"])

# ------------------------------ Accuracy tracker tab -------------------------
with tab_track:
    st.subheader("Accuracy tracker")
    _grades, _summ = get_grades(SEASON, scoring, matchup_strength, questionable_factor,
                                schedule_strength, use_model, use_role)
    if _summ is None:
        st.info("No games have finished this season yet, so there's nothing to grade.")
    else:
        _wks = sorted(int(x) for x in _grades["week"].unique())
        st.caption(
            f"Re-creates this app's projection for every finished game in Weeks {_wks[0]}-{_wks[-1]} of "
            f"{SEASON}, using only information available before each week, and compares it with what "
            f"players actually scored ({scoring}). Counted: players who played, weren't projected out, "
            f"and were projected at 4+ points ({len(_grades):,} player-weeks)."
            + (" These weeks were never part of the model's training data, so this is a true out-of-sample test."
               if use_model else ""))
        _ov = _summ["overall"]
        _app = _ov.loc["This app"]
        _old = _ov.loc["Recent form × matchup (old method)"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Average miss (pts per player)", f"{_app['Avg miss (pts)']:.2f}",
                  f"{_app['Avg miss (pts)'] - _old['Avg miss (pts)']:+.2f} vs old method", delta_color="inverse")
        k2.metric("Variance explained (R²)", f"{_app['R²']:.2f}", f"{_app['R²'] - _old['R²']:+.2f} vs old method")
        k3.metric("Rank correlation", f"{_app['Rank corr (within pos & week)']:.2f}",
                  f"{_app['Rank corr (within pos & week)'] - _old['Rank corr (within pos & week)']:+.2f} vs old method")
        k4.metric("Bias (projected - actual)", f"{_app['Bias (proj - actual)']:+.2f} pts")
        _b = float(_app["Bias (proj - actual)"])
        if abs(_b) >= 0.4:
            st.warning(
                f"Projections have run about {abs(_b):.1f} points {'low' if _b < 0 else 'high'} per player so far "
                "this season. Check the calibration table below: the miss is concentrated where it says. "
                "A short sample, a higher- or lower-scoring season than the training years, and grading only "
                "players who actually played all push the same way.")
        else:
            st.success("Projections are well centred this season: no meaningful average bias.")

        st.markdown("#### Average miss by week")
        st.line_chart(_summ["by_week"][["App avg miss", "Old method avg miss"]])
        c_a, c_b = st.columns(2)
        with c_a:
            st.markdown("#### By position")
            st.dataframe(_summ["by_pos"].round(2), width="stretch")
        with c_b:
            st.markdown("#### Start/sit calls")
            _pairs = _summ["pairs"].copy()
            _pairs["Higher projection scored more"] = (_pairs["Higher projection scored more"] * 100).round(0).astype(int).astype(str) + "%"
            st.dataframe(_pairs, hide_index=True, width="stretch")
            st.caption("For two players at the same position and week, how often the one projected higher "
                       "scored more. 50% would be a coin flip.")
        st.markdown("#### Calibration: projected vs actual")
        _cal = _summ["calibration"].copy()
        _cal.index = _cal.index.astype(str)
        _cal.index.name = "Projection band"
        d1, d2 = st.columns(2)
        d1.dataframe(_cal, width="stretch")
        d2.bar_chart(_cal[["Projected", "Actual"]])
        m1c, m2c = st.columns(2)
        _mc = {"week": "Week", "player": "Player", "pos": "Pos", "team": "Team", "opp": "Opp",
               "proj_final": "Projected", "actual": "Actual", "error": "Error"}
        with m1c:
            st.markdown("#### Biggest surprises (scored more)")
            st.dataframe(_summ["over"].rename(columns=_mc).round(1), hide_index=True, width="stretch")
        with m2c:
            st.markdown("#### Biggest disappointments (scored less)")
            st.dataframe(_summ["under"].rename(columns=_mc).round(1), hide_index=True, width="stretch")
        st.caption("A single week of fantasy scoring is mostly luck, injuries and game script, so even a good "
                   "model misses by about five points per player. Judge accuracy by the averages, not one result.")

if not picked:
    for _t in (tab_ss, tab_needs, tab_trade, tab_ideas, tab_waiver):
        with _t:
            st.info("Add the players on your roster (above) to get started.")
    st.stop()

my_ids = [label_to_id[p] for p in picked]
roster = pool[pool["player_id"].isin(my_ids)]

# ------------------------------ Start/Sit tab --------------------------------
with tab_ss:
    lineup = build_lineup(roster, starters, n_flex, flex_pos)
    lineup["Matchup"] = [matchup_icon(m, o) for m, o in zip(lineup["matchup"], lineup["opp"])]
    slot_order = {"QB": 0, "RB": 1, "WR": 2, "TE": 3, "FLX": 4, "BENCH": 5}
    lineup = lineup.sort_values(
        ["slot", "proj"], key=lambda s: s.map(slot_order) if s.name == "slot" else -s)

    st.subheader(f"Week {week} recommendations ({scoring})")

    starting = lineup[lineup["slot"] != "BENCH"]
    bench = lineup[lineup["slot"] == "BENCH"]

    for _, r in starting.iterrows():
        if r["injury_status"] in BAD_STATUSES:
            st.warning(f"**{r['player']}** ({r['slot']}) is listed {r['injury_text']}. "
                       "Consider a replacement.")
        elif r["injury_text"].startswith("Wk") and any(s in r["injury_text"] for s in BAD_STATUSES):
            st.info(f"{r['player']} was listed {r['injury_text']}. Check his current status.")

    played = starting["game_status"] == "Final"
    to_play = starting["game_status"] == "Upcoming"
    scored = float(starting.loc[played, "actual"].sum())
    remaining = float(starting.loc[to_play, "proj"].sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Expected lineup total", f"{scored + remaining:.1f}")
    m2.metric("Scored so far", f"{scored:.1f}")
    m3.metric("Projected still to play", f"{remaining:.1f}")
    m4.metric("Players on roster", len(lineup))
    if played.any():
        st.caption(f"Projection for this lineup before the games: {starting['proj'].sum():.1f}. "
                   "Players whose game is final show their actual points, so changes for them "
                   "aren't possible any more.")

    lineup["Status"] = lineup["game_status"]
    starting = lineup[lineup["slot"] != "BENCH"]
    bench = lineup[lineup["slot"] == "BENCH"]
    cols = {"slot": "Slot", "player": "Player", "team": "Team", "opp": "Opp",
            "Status": "Game", "Matchup": "Matchup", "injury_text": "Injury", "role_note": "Role change",
            "base": "Recent avg",
            "proj": "Projected", "actual": "Actual", "floor": "Floor", "ceiling": "Ceiling"}
    st.markdown("#### Start")
    st.dataframe(starting[list(cols)].rename(columns=cols), hide_index=True, width="stretch")
    if not bench.empty:
        st.markdown("#### Bench")
        st.dataframe(bench[list(cols)].rename(columns=cols), hide_index=True, width="stretch")

    if use_role:
        _n_role = int((starting["role_boost"] > 0).sum())
        if _n_role:
            st.caption(f"{_n_role} of your starters get a role-change boost from injured teammates; it is "
                       "included in Projected (see the Role change column).")
    if use_model:
        _m = model["metrics"].get(scoring, {})
        st.caption("Projection = fitted model's expected points per game × opponent matchup × injury "
                   "adjustment (when the report is current). On held-out seasons the model's baseline "
                   "explains about a third of week-to-week scoring, up from about a fifth for recent "
                   "form alone. Individual weeks stay very hard to predict.")
    else:
        st.caption("Projection = recency-weighted average of recent games × opponent matchup "
                   "× injury adjustment (when the report is current).")

# ------------------------------ Team needs tab -------------------------------
with tab_needs:
    st.subheader("Team needs")
    t = team_needs(pool, my_ids, levels, ranks, starters, n_flex, flex_pos)

    show = t["starting"].copy()
    show["Rank"] = [f"{r['pos']}{int(r['pos_rank'])}" if r["pos_rank"] == r["pos_rank"] else "—"
                    for _, r in show.iterrows()]
    show["Schedule"] = [sched_icon(s) if not f else "—"
                        for s, f in zip(show["sos"], show["is_filler"])]
    cols = {"slot": "Slot", "player": "Player", "Rank": "Pos rank", "base": "Recent avg",
            "fp_plus": "FP+", "Schedule": "ROS schedule",
            "vorp_week": "Above replacement (pts/wk)", "grade": "Grade", "alert": "Alert"}
    slot_order = {"QB": 0, "RB": 1, "WR": 2, "TE": 3, "FLX": 4}
    show = show.sort_values("slot", key=lambda s: s.map(slot_order))
    st.dataframe(show[list(cols)].rename(columns=cols), hide_index=True, width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Where you're thin")
        if t["needs"]:
            for n in t["needs"]:
                st.markdown(f"- {n}")
        else:
            st.success("No clear weak spots in your starting lineup.")
    with right:
        st.markdown("#### Tradeable depth")
        if t["surplus"]:
            for s in t["surplus"]:
                st.markdown(f"- {s}")
        else:
            st.markdown("No bench players with meaningful value over replacement.")

    if t["need_pos"] or t["surplus_pos"]:
        parts = []
        if t["need_pos"]:
            parts.append("look to trade **for** " + "/".join(t["need_pos"]))
        if t["surplus_pos"]:
            parts.append("trade **from** " + "/".join(t["surplus_pos"]) + " depth")
        st.info("Suggested focus: " + ", and ".join(parts) + ".")

    st.caption(f"Grades compare each starter to the rest of the league's active players at his "
               f"position. 'Replacement level' is estimated for a {teams}-team league. "
               "ROS schedule reflects how many points the remaining opponents allow to his position.")

# ------------------------------ Trade analyzer tab ---------------------------
with tab_trade:
    st.subheader("Trade analyzer")

    with st.expander("How FP+ works"):
        st.markdown(
            "**FP+ = 100 × (player's fantasy points per game ÷ his position's average).** "
            "100 is exactly average for the position, 150 is 50% better, 80 is 20% worse."
        )
        avg_rows = [{"Position": pos, "Average pts/game (= FP+ 100)": round(v, 1)}
                    for pos, v in pos_avgs.items()]
        st.dataframe(avg_rows, hide_index=True)
        st.caption(f"Based on {fp_basis.lower()} in {scoring}. Averages include only players who "
                   f"have played at least {min_games} game(s) this season.")

    with st.expander("How the composite value works"):
        st.markdown(
            "The composite blends five ingredients into one number on the FP+ scale "
            "(100 = an average qualified player):\n\n"
            "- **FP+**, stabilised: pulled toward the position average for players with few games.\n"
            "- **Value over replacement**: points per game above a waiver-wire player (no schedule "
            "adjustment, so schedule isn't counted twice).\n"
            "- **Target share** (RB/WR/TE) and **rush share** (RB): recency-weighted over the last "
            "four games, also stabilised.\n"
            "- **Rest-of-season schedule**: how much the remaining opponents allow at his position.\n\n"
            "Each ingredient is converted to a z-score among qualified players, then averaged with the "
            "weights below. A player's weights are rescaled over the ingredients he has (QBs have no "
            "usage shares, only RBs have rush share).")
        eff = [{"Position": pos, **{COMPONENT_LABELS[c]: (f"{comp_info['effective'][pos][c] * 100:.0f}%"
                                                         if c in comp_info["effective"][pos] else "n/a")
                                    for c in COMPONENT_KEYS}}
               for pos in comp_info["effective"]]
        st.markdown("**Effective weights by position**")
        st.dataframe(eff, hide_index=True)
        st.markdown(
            "**What the backtest found** (2022-2025, five points per season, ranking players within "
            "position by their actual rest-of-season points per game):\n\n"
            "- Usage adds real information for RB/WR/TE, most of all for TEs. It doesn't for QBs.\n"
            "- Schedule strength has a very small effect, so it gets a small weight. At 10% it hurt accuracy.\n"
            "- Shrinking small samples with about 2 games of prior gave the most accurate *values*. "
            "It doesn't improve the *ordering* of players, so the composite ranks players about as well "
            "as raw FP+ does; its advantages are better-calibrated values and cross-position scaling.\n"
            "- Value over replacement and FP+ move together within a position, so their split barely "
            "changes rankings; value over replacement mainly sets the scale *between* positions.\n\n"
            "Differences between weightings were modest (about 0.01-0.02 in rank correlation), so treat "
            "the weights as a sensible default rather than a precise optimum. Run `backtest_value.py` "
            "to reproduce this.")

    if model:
        with st.expander("How the projection model works"):
            st.markdown(
                "A regularised linear model, one per position and scoring format, trained on 2022-2025. "
                "It predicts a player's expected points per game from:\n\n"
                "- season average (pulled toward the position average when he has few games)\n"
                "- recent form (last four games)\n"
                "- target share and rush share\n"
                "- **expected fantasy points**: what his carries and targets should have produced\n"
                "- offensive snap share\n"
                "- last season's points per game\n\n"
                "This week's projection multiplies that by the opponent matchup and the injury adjustment.")
            _met = model["metrics"].get(scoring)
            if _met:
                st.markdown("**Accuracy on seasons the model never saw** (train on three seasons, test on "
                            "the fourth, rest-of-season points per game, within position):")
                st.dataframe(pd.DataFrame({
                    "Rank correlation, model": _met["rank_corr"], "Rank correlation, season avg only": _met["baseline_rank_corr"],
                    "R² (variance explained), model": _met["r2"], "R², season avg only": _met["baseline_r2"],
                }).round(3))
            _coef = pd.DataFrame({pos: m["coef"] for pos, m in model["models"][scoring].items()}).round(2)
            st.markdown("**What the model leans on** (standardised weights; bigger = more influence; blank = not used for that position)")
            st.dataframe(_coef)
            st.caption(f"Trained {model['trained']}. Typical error in a player's rest-of-season average: "
                       + ", ".join(f"{pos} ±{m['resid_sd']:.1f}" for pos, m in model["models"][scoring].items())
                       + " points per game. Retrain with train_model.py after each season.")

    partner, partner_ids = None, []
    if league and len(team_map) > 1:
        partner = st.selectbox("Trade partner", [n for n in team_map if n != my_team],
                               key=f"partner_{league['league_name']}_{my_team}")
        partner_ids = team_map[partner]["ids"]
        others = searchable[searchable["player_id"].isin(partner_ids)
                            & ~searchable["player_id"].isin(my_ids)]
        if team_map[partner]["unmatched"]:
            st.caption("Not found in this season's stats: "
                       + ", ".join(team_map[partner]["unmatched"]))
    else:
        others = searchable[~searchable["player_id"].isin(my_ids)]
    other_map = dict(zip(others["label"], others["player_id"]))

    g1, g2 = st.columns(2)
    give_labels = g1.multiselect("You give", picked, key="give")
    get_labels = g2.multiselect("You get", others["label"].tolist(), key=f"get_{partner}",
                                placeholder="Search players on other teams...")

    if not give_labels and not get_labels:
        st.info("Pick the players you'd give and get to see how the trade changes your team.")
    else:
        give_ids = [label_to_id[x] for x in give_labels]
        get_ids = [other_map[x] for x in get_labels]
        res = evaluate_trade(pool, my_ids, give_ids, get_ids, levels,
                             starters, n_flex, flex_pos)

        level, text = res["headline"]
        getattr(st, level)(f"**{text}**")
        for n in res["notes"]:
            st.warning(n)

        a, b, c = st.columns(3)
        d_ros = res["after"]["ros"] - res["before"]["ros"]
        d_wk = res["after"]["week"] - res["before"]["week"]
        a.metric("Lineup strength (pts/wk, schedule-adjusted)", f"{res['after']['ros']:.1f}", f"{d_ros:+.1f}")
        b.metric(f"Week {week} projection", f"{res['after']['week']:.1f}", f"{d_wk:+.1f}")
        c.metric(f"Trade value ({trade_method}), you get vs give",
                 f"{res['get_value']:.0f} vs {res['give_value']:.0f}",
                 f"{res['get_value'] - res['give_value']:+.0f}")

        fg, ft = res["fp_values"]
        vg, vt = res["vorp_values"]
        st.caption(f"FP+ totals: get {ft:.0f} vs give {fg:.0f}  |  "
                   f"Value over replacement: get {vt:.0f} vs give {vg:.0f}")

        if partner and partner_ids and give_ids and get_ids:
            their = evaluate_trade(pool, partner_ids, get_ids, give_ids, levels,
                                   starters, n_flex, flex_pos)
            mine_d = res["after"]["ros"] - res["before"]["ros"]
            their_d = their["after"]["ros"] - their["before"]["ros"]
            st.markdown(f"#### {partner}'s side")
            if mine_d > 0.25 and their_d > 0.25:
                st.success("Both lineups improve, which is the kind of trade managers tend to accept.")
            elif mine_d > 0.25 and their_d < -0.25:
                st.warning("This helps you but weakens their starting lineup, so they may decline "
                           "unless they value the depth or future games.")
            elif mine_d < -0.25 and their_d > 0.25:
                st.info("This helps them more than you.")
            else:
                st.info("Little lineup change for either side, so it comes down to value.")
            tc_a, tc_b, tc_c = st.columns(3)
            tc_a.metric("Their lineup strength (pts/wk)", f"{their['after']['ros']:.1f}",
                        f"{their_d:+.1f}")
            tc_b.metric(f"Their Week {week} projection", f"{their['after']['week']:.1f}",
                        f"{their['after']['week'] - their['before']['week']:+.1f}")
            tc_c.metric(f"Their trade value ({trade_method}), get vs give",
                        f"{their['get_value']:.0f} vs {their['give_value']:.0f}",
                        f"{their['get_value'] - their['give_value']:+.0f}")

        def fmt(df):
            d = df.copy()
            d["Schedule"] = [sched_icon(s) for s in d["sos"]]
            d["tgt"] = (d["target_share"] * 100).round(0)
            d["rush"] = (d["rush_share"] * 100).round(0)
            cols = {"player": "Player", "pos": "Pos", "team": "Team", "injury_text": "Injury",
                    "season_games": "Games", "proj_ppg": "Proj pts/gm", "fp_plus": "FP+", "tgt": "Target %", "rush": "Rush %",
                    "Schedule": "ROS schedule", "trade_value": "Trade value"}
            return d[list(cols)].rename(columns=cols)

        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**You give**")
            if len(res["give"]):
                st.dataframe(fmt(res["give"]), hide_index=True, width="stretch")
        with t2:
            st.markdown("**You get**")
            if len(res["get"]):
                st.dataframe(fmt(res["get"]), hide_index=True, width="stretch")

        with st.expander("Component breakdown"):
            both = pd.concat([res["give"].assign(Side="Give"), res["get"].assign(Side="Get")])
            if len(both):
                brk = {"Side": "Side", "player": "Player", "fp_plus": "FP+ (raw)",
                       "fp_stab": "FP+ (stabilised)", "vorp_stab": "Above repl. (pts/gm)",
                       "target_share": "Target share", "rush_share": "Rush share",
                       "sos_raw": "Schedule factor", "composite_z": "Composite z",
                       "value_index": "Composite value"}
                st.dataframe(both[list(brk)].rename(columns=brk).round(2),
                             hide_index=True, width="stretch")
                st.caption("Composite z is the weighted average of the standardised ingredients. "
                           "Composite value = 100 + z × the spread of FP+.")

        with st.expander("Your starting lineup after the trade"):
            lu = res["after"]["ros_lineup"]
            lu = lu[lu["slot"] != "BENCH"]
            slot_order = {"QB": 0, "RB": 1, "WR": 2, "TE": 3, "FLX": 4}
            lu = lu.sort_values("slot", key=lambda s: s.map(slot_order))
            st.dataframe(lu[["slot", "player", "team", "ros_pts"]].rename(
                columns={"slot": "Slot", "player": "Player", "team": "Team",
                         "ros_pts": "Schedule-adj. avg"}),
                hide_index=True, width="stretch")
            st.caption("Empty slots are filled with replacement-level waiver players, "
                       "so a 2-for-1 trade is judged fairly.")

        st.caption("Value comparisons don't include the other manager's roster needs or injury "
                   "recovery timelines, so treat verdicts as a starting point.")

# ------------------------------ Trade ideas tab ------------------------------
with tab_ideas:
    st.subheader("Trade ideas")
    if not league or len(team_map) < 2:
        st.info("Import your ESPN league (sidebar) to get trade suggestions. They need every "
                "team's roster to find trades that work for both sides.")
    else:
        st.caption("Searches every other team for 1-for-1, 2-for-1 and 1-for-2 trades that are "
                   "close in value and improve your starting lineup. Ranked by your gain, with a "
                   "bonus when the other team improves too, since those are the trades that get accepted.")
        f1, f2, f3, f4 = st.columns(4)
        min_gain = f1.slider("Min. lineup gain (pts/wk)", 0.0, 5.0, 0.5, 0.25, key="idea_gain")
        max_gap = f2.slider("Max value gap (%)", 5, 40, 15, 5, key="idea_gap") / 100
        n_ideas = f3.slider("Ideas to show", 3, 15, 8, key="idea_n")
        both = f4.checkbox("Must help them too", value=True, key="idea_both",
                           help="Only show trades that also improve the other team's lineup.")

        other_teams = {n: v["ids"] for n, v in team_map.items() if n != my_team}
        ideas = suggest_trades(pool, my_ids, other_teams, levels, starters, n_flex, flex_pos,
                               min_gain=min_gain, max_value_gap=max_gap, require_both=both,
                               top_n=n_ideas)

        if not ideas:
            st.warning("No trades met those filters. Try lowering the minimum gain, widening the "
                       "value gap, or unchecking 'Must help them too'.")
        for i, sug in enumerate(ideas, 1):
            det = describe_suggestion(pool, my_ids, other_teams, sug, levels, starters, n_flex, flex_pos)
            give_txt = " + ".join(id_to_label.get(x, x) for x in sug["give_ids"])
            get_txt = " + ".join(id_to_label.get(x, x) for x in sug["get_ids"])
            with st.container(border=True):
                st.markdown(f"**{i}. {sug['partner']}**: you give {give_txt}, you get {get_txt}")
                m1, m2, m3 = st.columns(3)
                m1.metric("Your lineup (pts/wk)", f"{det['mine']['after']['ros']:.1f}",
                          f"{sug['my_delta']:+.1f}")
                m2.metric("Their lineup (pts/wk)", f"{det['theirs']['after']['ros']:.1f}",
                          f"{sug['their_delta']:+.1f}")
                m3.metric(f"Value ({trade_method}), you get vs give",
                          f"{sug['get_value']:.0f} vs {sug['give_value']:.0f}",
                          f"{sug['get_value'] - sug['give_value']:+.0f}")
                st.caption(f"**You:** {det['my_text']}")
                st.caption(f"**They:** {det['their_text']}")
                for note in det["mine"]["notes"][:3]:
                    st.caption("⚠️ " + note)

        if ideas:
            st.caption("Ideas use the same value and lineup math as the Trade analyzer, so you can "
                       "plug any of them in there to see the full breakdown. They're starting "
                       "points for a conversation, not guarantees. Real managers also weigh "
                       "reputation, bye weeks, and roster needs this model can't see.")


# ------------------------------ Waiver wire tab ------------------------------
def _wv_table(df, extra=None):
    d = df.copy()
    d["Matchup"] = [matchup_icon(m, o) for m, o in zip(d["matchup"], d["opp"])]
    cols = {"player": "Player", "pos": "Pos", "team": "Team", "opp": "Opp", "Matchup": "Matchup",
            "proj": "Proj (week)", "ros_pts": "Rest-of-season avg", "role_note": "Role change",
            "injury_text": "Injury"}
    cols.update(extra or {})
    return d[list(cols)].rename(columns=cols)


with tab_waiver:
    st.subheader("Waiver wire")
    if week != cur_week:
        st.caption(f"Showing Week {week} data. Switch the Week selector to the current week for live pickups.")

    if not league or len(team_map) < 2:
        st.info("Import your ESPN league (sidebar) so the app knows who is really a free agent and who "
                "you would drop. Until then, this is who gains from injured teammates across the whole "
                "NFL. Check availability on ESPN.")
        _rc = pool[(pool["role_boost"] > 0) & pool["recent"]].sort_values("role_boost", ascending=False).head(15)
        if len(_rc):
            st.dataframe(_wv_table(_rc, {"role_boost": "Boost (pts)"}), hide_index=True, width="stretch")
        else:
            st.caption("No meaningful role changes this week.")
    else:
        _fa_all = active_free_agents(pool, rostered_ids)
        st.caption(f"{len(_fa_all)} free agents: players with this season's stats who aren't on any team in "
                   "your imported league and aren't Out or on injured reserve. Availability is as of your last "
                   "import, so re-import after waiver runs.")
        _ranked = rank_free_agents(pool, my_ids, rostered_ids, levels, starters, n_flex, flex_pos)

        st.markdown("#### Best adds for your lineup")
        _helpful = _ranked[(_ranked["gain_ros"] > 0.05) | (_ranked["gain_week"] > 0.05)].head(10)
        if _helpful.empty:
            st.info("No free agent would meaningfully improve your starting lineup right now. "
                    "Your roster is stronger than what's on the wire.")
        else:
            st.dataframe(_wv_table(_helpful, {
                "gain_week": f"Lineup gain, Week {week} (pts)", "gain_ros": "Lineup gain, rest of season (pts/wk)",
                "drop": "Drop"}).round(2), hide_index=True, width="stretch")
            st.caption("Each row assumes you drop the player who costs you least. Rest-of-season gain is "
                       "schedule-adjusted weekly lineup strength; week gain is this week's projected lineup.")

        st.markdown("#### Role-change pickups")
        _rc = _fa_all[_fa_all["role_boost"] > 0].sort_values("role_boost", ascending=False).head(10)
        if len(_rc):
            st.dataframe(_wv_table(_rc, {"role_boost": "Boost (pts)"}), hide_index=True, width="stretch")
            st.caption("Free agents who inherit targets or carries from injured teammates this week. "
                       "These are mostly short-term plays.")
        else:
            st.caption("No free agent has a meaningful role change this week.")

        st.markdown("#### Best available by position")
        _pp = st.radio("Position", ["QB", "RB", "WR", "TE"], horizontal=True, key="wv_pos")
        _best = _fa_all[_fa_all["pos"] == _pp].sort_values("ros_pts", ascending=False).head(10)
        st.dataframe(_wv_table(_best), hide_index=True, width="stretch")

        if real_repl_ok:
            st.markdown("#### Your league's replacement level")
            st.dataframe(pd.DataFrame({"Estimated from league size": est_levels, "Best free agent (used)": levels}
                                      ).round(1).T, width="stretch")
            st.caption("Replacement level = rest-of-season average of the best free agent at each position. "
                       "It sets how much value a rostered player has over what you could pick up.")

    with st.expander("How the waiver rankings and role changes work"):
        st.markdown(
            "**Waiver ranking.** For each top free agent, the app adds him to your roster, removes whichever "
            "player costs your lineup least, and measures the change in your best starting lineup, both for "
            "this week and for the rest of the season (schedule-adjusted). Empty slots are filled with "
            "replacement-level players, so thin positions show real gains.\n\n"
            "**Role changes.** When a teammate is Out, Doubtful or on injured reserve, his share of targets "
            "and carries is redistributed to the remaining players in proportion to their own shares. The "
            "extra points use values fitted on 2022-2025 (about 45 points per unit of vacated team target "
            "share and 16 per unit of rush share in PPR, scaled down for half and standard scoring). Absences "
            "that are new (the player played his team's last game) count in full; longer absences count less, "
            "because recent usage already reflects them.\n\n"
            f"**What the backtest found.** The effect is real and consistent across WR, TE and RB, and boosts of "
            f"{ROLE_MIN_BOOST:.1f} points or more matched actual extra points closely (for example 2.0 predicted "
            "vs 1.9 actual). Adding it improved variance explained only slightly (about +0.006 out of sample, "
            "in every held-out season) and did not reduce the typical miss, so it mainly corrects a bias for "
            f"players absorbing a lot of usage. Only boosts of at least {ROLE_MIN_BOOST:.1f} points are applied, "
            f"at {int(ROLE_SHRINK * 100)}% strength. It can't see who will actually get the work (a team may "
            "sign a veteran or favor one back), and a backup quarterback effect was too small to adjust for.")
