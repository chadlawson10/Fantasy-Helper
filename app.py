import streamlit as st

from start_sit_core import (
    BAD_STATUSES,
    FP_BASES,
    SCORING_COLUMNS,
    TRADE_METHODS,
    add_fp_plus,
    add_values,
    build_lineup,
    build_pool,
    evaluate_trade,
    load_data,
    replacement_levels,
    set_trade_value,
    team_needs,
)

st.set_page_config(page_title="Fantasy Football Helper", page_icon="🏈", layout="wide")

SEASON = 2026  # update each year


# ------------------------------- Cached loaders -----------------------------
@st.cache_data(ttl=6 * 60 * 60, show_spinner="Loading NFL data...")
def get_data(season):
    return load_data(season)


@st.cache_data(ttl=6 * 60 * 60, show_spinner="Crunching projections...")
def get_pool(season, scoring, matchup_strength, questionable_factor, schedule_strength):
    weekly, schedule, injuries = get_data(season)
    return build_pool(weekly, schedule, injuries, season, scoring=scoring,
                      matchup_strength=matchup_strength,
                      questionable_factor=questionable_factor,
                      schedule_strength=schedule_strength)


def matchup_icon(m, opp):
    if opp in ("BYE", "—"):
        return opp
    return "🟢 Good" if m >= 1.05 else "🔴 Tough" if m <= 0.95 else "⚪ Neutral"


def sched_icon(s):
    if s != s:  # NaN
        return "—"
    return "🟢 Easy" if s >= 1.03 else "🔴 Hard" if s <= 0.97 else "⚪ Avg"


# ------------------------------- Sidebar ------------------------------------
st.sidebar.header("League settings")
scoring = st.sidebar.selectbox("Scoring", list(SCORING_COLUMNS), index=0, key="scoring")
teams = st.sidebar.number_input("Teams in league", 6, 20, 12,
                                help="Used to estimate replacement level (who's on waivers).")

st.sidebar.subheader("Lineup slots")
c1, c2 = st.sidebar.columns(2)
n_qb = c1.number_input("QB", 0, 3, 1)
n_rb = c2.number_input("RB", 0, 5, 2)
n_wr = c1.number_input("WR", 0, 5, 2)
n_te = c2.number_input("TE", 0, 3, 1)
n_flex = c1.number_input("FLX", 0, 4, 1)
superflex = c2.checkbox("Superflex", value=False, help="Allow QBs in FLX")

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

starters = {"QB": n_qb, "RB": n_rb, "WR": n_wr, "TE": n_te}
flex_pos = ("QB", "RB", "WR", "TE") if superflex else ("RB", "WR", "TE")

# -------------------------------- Data --------------------------------------
pool, meta = get_pool(SEASON, scoring, matchup_strength, questionable_factor, schedule_strength)
week = meta["week"]
levels, ranks = replacement_levels(pool, teams, starters, n_flex, flex_pos)
pool = add_values(pool, levels)
pool, pos_avgs, min_games = add_fp_plus(pool, week, fp_basis, qualify_share)
pool = set_trade_value(pool, trade_method)

st.title("🏈 Fantasy Football Helper")

if meta["injury_week"] is None:
    st.caption("No injury report available yet.")
elif meta["injury_stale"]:
    st.caption(f"Injury report shown is from Week {meta['injury_week']}; the Week {week} "
               "report isn't out yet. Statuses are shown for reference but not applied to projections.")
else:
    st.caption(f"Injury report: Week {meta['injury_week']} (applied to projections).")

searchable = pool[pool["current"]].sort_values("label")
label_to_id = dict(zip(searchable["label"], searchable["player_id"]))

picked = st.multiselect(
    "Your roster (type to search)", searchable["label"].tolist(), key="roster",
    placeholder="Start typing a player name...")

if not picked:
    st.info("Add the players on your roster to get started.")
    st.stop()

my_ids = [label_to_id[p] for p in picked]
roster = pool[pool["player_id"].isin(my_ids)]

tab_ss, tab_needs, tab_trade = st.tabs(["Start/Sit", "Team needs", "Trade analyzer"])

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

    m1, m2 = st.columns(2)
    m1.metric("Projected lineup total", f"{starting['proj'].sum():.1f}")
    m2.metric("Players on roster", len(lineup))

    cols = {"slot": "Slot", "player": "Player", "team": "Team", "opp": "Opp",
            "Matchup": "Matchup", "injury_text": "Injury", "base": "Recent avg",
            "proj": "Projected", "floor": "Floor", "ceiling": "Ceiling"}
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

    others = searchable[~searchable["player_id"].isin(my_ids)]
    other_map = dict(zip(others["label"], others["player_id"]))

    g1, g2 = st.columns(2)
    give_labels = g1.multiselect("You give", picked, key="give")
    get_labels = g2.multiselect("You get", others["label"].tolist(), key="get",
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

        def fmt(df):
            d = df.copy()
            d["Schedule"] = [sched_icon(s) for s in d["sos"]]
            cols = {"player": "Player", "pos": "Pos", "team": "Team", "injury_text": "Injury",
                    "season_games": "Games", "fp_value": "Avg pts/game", "pos_avg": "Pos avg",
                    "fp_plus": "FP+", "Schedule": "ROS schedule", "games_left": "Games left",
                    "ros_value": "Value over repl.", "trade_value": "Trade value"}
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
