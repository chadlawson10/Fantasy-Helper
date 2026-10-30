from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from espn_import import fetch_league, friendly_error, load_crosswalk, map_league
from trade_finder import describe_suggestion, suggest_trades
from start_sit_core import (
    BAD_STATUSES,
    COMPONENT_KEYS,
    COMPONENT_LABELS,
    DEFAULT_SHRINK_K,
    FP_BASES,
    ORIGINAL_WEIGHTS,
    RECOMMENDED_WEIGHTS,
    SCORING_COLUMNS,
    TRADE_METHODS,
    add_composite_value,
    add_fp_plus,
    add_values,
    build_lineup,
    build_pool,
    current_week,
    evaluate_trade,
    load_data,
    replacement_levels,
    set_trade_value,
    team_needs,
    week_statuses,
)

st.set_page_config(page_title="Fantasy Football Helper", page_icon="🏈", layout="wide")

SEASON = 2026  # update each year


# ------------------------------- Cached loaders -----------------------------
# Stats, schedule, injury reports and rosters are re-downloaded every 3 hours (or on demand with
# the Refresh button), so new injury reports, trades, signings and game results show up daily.
@st.cache_data(ttl=3 * 60 * 60, show_spinner="Loading NFL data...")
def get_data(season):
    weekly, schedule, injuries, rosters = load_data(season)
    return weekly, schedule, injuries, rosters, datetime.now(timezone.utc)


@st.cache_data(ttl=3 * 60 * 60, show_spinner="Crunching projections...")
def get_pool(season, week, scoring, matchup_strength, questionable_factor, schedule_strength):
    weekly, schedule, injuries, rosters, _ = get_data(season)
    return build_pool(weekly, schedule, injuries, season, scoring=scoring,
                      matchup_strength=matchup_strength,
                      questionable_factor=questionable_factor,
                      schedule_strength=schedule_strength, week=week, rosters=rosters)


@st.cache_data(ttl=24 * 60 * 60)
def get_crosswalk():
    return load_crosswalk()


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
_weekly, _schedule, _inj, _ros, fetched_at = get_data(SEASON)
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

weights, shrink_k = dict(RECOMMENDED_WEIGHTS), DEFAULT_SHRINK_K
if trade_method == "Composite value":
    preset = st.sidebar.selectbox(
        "Composite weights", ["Recommended (backtested)", "Your original proposal", "Custom"],
        key="preset", help="Recommended weights come from a 2022-2025 backtest (see the "
                           "'How the composite value works' box in the Trade analyzer tab).")
    if preset == "Your original proposal":
        weights = dict(ORIGINAL_WEIGHTS)
    elif preset == "Custom":
        raw = {c: st.sidebar.slider(COMPONENT_LABELS[c], 0, 100,
                                    int(round(RECOMMENDED_WEIGHTS[c] * 100)), key=f"w_{c}")
               for c in COMPONENT_KEYS}
        total = sum(raw.values())
        weights = {c: v / total for c, v in raw.items()} if total else dict(RECOMMENDED_WEIGHTS)
        st.sidebar.caption("Weights are rescaled to add up to 100%.")
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
levels, ranks = replacement_levels(pool, teams, starters, n_flex, flex_pos)
pool = add_values(pool, levels)
pool, pos_avgs, min_games = add_fp_plus(pool, week, fp_basis, qualify_share)
pool, comp_info = add_composite_value(pool, ranks, weights, shrink_k)
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
league = st.session_state.get("league")
team_map, my_team = {}, None
if league:
    team_map = map_league(league, get_crosswalk(), searchable)
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

if not picked:
    st.info("Add the players on your roster to get started.")
    st.stop()

my_ids = [label_to_id[p] for p in picked]
roster = pool[pool["player_id"].isin(my_ids)]

tab_ss, tab_needs, tab_trade, tab_ideas = st.tabs(
    ["Start/Sit", "Team needs", "Trade analyzer", "Trade ideas"])

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
            "Status": "Game", "Matchup": "Matchup", "injury_text": "Injury", "base": "Recent avg",
            "proj": "Projected", "actual": "Actual", "floor": "Floor", "ceiling": "Ceiling"}
    st.markdown("#### Start")
    st.dataframe(starting[list(cols)].rename(columns=cols), hide_index=True, width="stretch")
    if not bench.empty:
        st.markdown("#### Bench")
        st.dataframe(bench[list(cols)].rename(columns=cols), hide_index=True, width="stretch")

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
                    "season_games": "Games", "fp_plus": "FP+", "tgt": "Target %", "rush": "Rush %",
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
