"""
Mock trade suggestions.

Searches every other team in your league for 1-for-1, 2-for-1 and 1-for-2 trades that:
  * are roughly even on trade value (within a tolerance you choose),
  * improve your starting lineup by at least a minimum amount, and
  * (optionally) also improve the other team's starting lineup, because those are
    the trades managers actually accept.

For speed, the search uses a light-weight lineup calculator that mirrors the one in
start_sit_core. The best few results are then re-checked with the full evaluate_trade.
"""

from itertools import combinations

from start_sit_core import BAD_STATUSES, POSITIONS, evaluate_trade


# ------------------------------ Fast lineup math -----------------------------
def _strength(members, info, levels, starters, flex_slots, flex_positions):
    """
    Starting-lineup strength (schedule-adjusted points/week) for a set of players.
    Mirrors build_lineup + add_fillers: replacement-level waiver players fill any
    slot a rostered player can't beat, and flex takes the best player left over.
    """
    by_pos = {p: [] for p in POSITIONS}
    for pid in members:
        pos, ros_pts = info[pid][0], info[pid][1]
        by_pos[pos].append(ros_pts)

    total, extra = 0.0, []
    for pos in POSITIONS:
        n = starters.get(pos, 0)
        in_flex = pos in flex_positions
        fillers = n + (flex_slots if in_flex else 0)
        vals = sorted(by_pos[pos] + [levels[pos]] * fillers, reverse=True)
        total += sum(vals[:n])
        if in_flex:
            extra.extend(vals[n:])
    if flex_slots:
        extra.sort(reverse=True)
        total += sum(extra[:flex_slots])
    return total


def _is_unavailable(info_row):
    """Out/Doubtful on a *current* injury report (stale 'Wk' labels don't count)."""
    status, text = info_row[4], info_row[5]
    return status in BAD_STATUSES and not str(text).startswith("Wk")


# ---------------------------------- Search ----------------------------------
def suggest_trades(
    pool, my_ids, other_teams, levels, starters, flex_slots, flex_positions,
    min_gain=0.5, max_value_gap=0.15, require_both=True,
    top_n=10, per_team=2, candidates_per_side=14,
):
    """
    other_teams: {team name: [player_ids]}.
    Returns a list of dicts sorted best-first. Each has the partner, the player ids on
    each side, and the lineup/value numbers that justify it.
    """
    cols = ["pos", "ros_pts", "trade_value", "recent", "injury_status", "injury_text"]
    info = {pid: tuple(row) for pid, row in zip(pool["player_id"], pool[cols].itertuples(index=False))}
    args = (info, levels, starters, flex_slots, flex_positions)

    my_ids = set(my_ids)
    my_before = _strength(my_ids, *args)
    mine = sorted(my_ids, key=lambda i: info[i][2], reverse=True)[:candidates_per_side]

    found = []
    for team, ids in other_teams.items():
        t_ids = set(ids)
        their_before = _strength(t_ids, *args)
        theirs = sorted((i for i in t_ids if info[i][3] and not _is_unavailable(info[i])),
                        key=lambda i: info[i][2], reverse=True)[:candidates_per_side]
        mine_ok = [i for i in mine if not _is_unavailable(info[i])]  # don't hand them a player who's Out

        team_results, seen_get = [], set()
        for n_give, n_get in ((1, 1), (2, 1), (1, 2)):
            for give in combinations(mine_ok, n_give):
                gv = sum(info[i][2] for i in give)
                for get in combinations(theirs, n_get):
                    sv = sum(info[i][2] for i in get)
                    big = max(gv, sv)
                    if big <= 0 or abs(sv - gv) > max_value_gap * big:
                        continue
                    my_delta = _strength((my_ids - set(give)) | set(get), *args) - my_before
                    if my_delta < min_gain:
                        continue
                    their_delta = _strength((t_ids - set(get)) | set(give), *args) - their_before
                    if require_both and their_delta <= 0.0:
                        continue
                    # Rank by your gain, with a bonus (capped at your gain) when they gain too
                    score = my_delta + min(max(their_delta, 0.0), my_delta)
                    team_results.append({
                        "partner": team, "give_ids": list(give), "get_ids": list(get),
                        "my_delta": my_delta, "their_delta": their_delta,
                        "give_value": gv, "get_value": sv, "score": score})

        team_results.sort(key=lambda r: r["score"], reverse=True)
        kept = 0
        for r in team_results:
            key = frozenset(r["get_ids"])
            if key in seen_get:  # avoid several variants of the same target
                continue
            seen_get.add(key)
            found.append(r)
            kept += 1
            if kept >= per_team:
                break

    found.sort(key=lambda r: r["score"], reverse=True)
    return found[:top_n]


# --------------------------------- Details ----------------------------------
def _starters(side):
    lu = side["ros_lineup"]
    lu = lu[lu["slot"] != "BENCH"]
    return {r.player_id: (r.player, r.pos, bool(r.is_filler)) for r in lu.itertuples()}


def changes_text(res, traded_away, who="your"):
    """Describe how the starting lineup changes, e.g. '+ Gibbs (RB) / - Williams (RB, traded)'."""
    before, after = _starters(res["before"]), _starters(res["after"])
    added = [f"{n} ({p})" for k, (n, p, f) in after.items() if k not in before and not f]
    removed = []
    for k, (n, p, f) in before.items():
        if k not in after:
            if f:
                continue
            removed.append(f"{n} ({p}, {'traded' if k in traded_away else 'to bench'})")
    filler_gone = sum(1 for k, (_, _, f) in before.items() if f and k not in after)
    parts = []
    if added:
        parts.append("joins " + who + " lineup: " + ", ".join(added))
    if removed:
        parts.append("leaves it: " + ", ".join(removed))
    if filler_gone and not added:
        parts.append("replaces a waiver-level starter")
    return "; ".join(parts) if parts else "no change to the starting lineup"


def describe_suggestion(pool, my_ids, other_teams, s, levels, starters, flex_slots, flex_positions):
    """Run the full trade evaluation (both sides) for one suggestion."""
    args = (levels, starters, flex_slots, flex_positions)
    mine = evaluate_trade(pool, my_ids, s["give_ids"], s["get_ids"], *args)
    theirs = evaluate_trade(pool, other_teams[s["partner"]], s["get_ids"], s["give_ids"], *args)
    return {
        "mine": mine, "theirs": theirs,
        "my_text": changes_text(mine, set(s["give_ids"]), "your"),
        "their_text": changes_text(theirs, set(s["get_ids"]), "their"),
    }
