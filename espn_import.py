"""
ESPN league import.

Uses the community `espn-api` package (an unofficial wrapper around ESPN's
private API, so it can break if ESPN changes things). Everything is converted to
plain dicts right away so the rest of the app never touches ESPN objects.

Private leagues need two cookies from your browser (espn_s2 and SWID).
Public leagues only need the league ID.
"""

import nflreadpy as nfl
import numpy as np
import pandas as pd

OFFENSE = ("QB", "RB", "WR", "TE")
_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
RECEPTION_STAT_ID = 53  # ESPN's id for "each reception"


# --------------------------------- Helpers ----------------------------------
def _idstr(x):
    """Normalise an ID (str / int / float / NaN / NA) to a clean string or None."""
    if x is None or (not isinstance(x, str) and pd.isna(x)):
        return None
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return None if s.lower() in ("", "nan", "none", "<na>") else s


def _norm_name(name):
    import re
    if not isinstance(name, str):
        return ""
    toks = [t for t in re.sub(r"[^a-z ]", "", name.lower()).split() if t not in _NAME_SUFFIXES]
    return "".join(toks)


def load_crosswalk() -> pd.DataFrame:
    """nflverse ID crosswalk: maps ESPN player IDs to the gsis IDs used in our stats."""
    x = nfl.load_ff_playerids().to_pandas()
    return x[x["gsis_id"].notna()][["gsis_id", "espn_id", "name", "position"]]


# ------------------------------ League -> dict ------------------------------
def parse_settings(settings) -> dict:
    """Pull the settings this app uses out of an espn-api Settings object."""
    counts = dict(getattr(settings, "position_slot_counts", {}) or {})

    # Reception points decide PPR / half PPR / standard
    rec_points = 0.0
    for item in getattr(settings, "scoring_format", []) or []:
        if item.get("id") == RECEPTION_STAT_ID:
            rec_points = float(item.get("points", 0) or 0)
    scoring = min({"PPR": 1.0, "Half PPR": 0.5, "Standard": 0.0}.items(),
                  key=lambda kv: abs(kv[1] - rec_points))[0]

    full_flex = counts.get("RB/WR/TE", 0)
    partial_flex = counts.get("RB/WR", 0) + counts.get("WR/TE", 0)
    superflex = counts.get("OP", 0)

    notes = []
    if partial_flex:
        notes.append("Your league has a restricted flex (RB/WR or WR/TE); it was treated as a "
                     "full RB/WR/TE flex.")
    if superflex:
        notes.append("Your league has a superflex (OP) slot; all flex slots were treated as "
                     "QB-eligible.")
    if abs(rec_points - round(rec_points * 2) / 2) > 1e-9:
        notes.append(f"Your league scores {rec_points} per reception; rounded to the nearest "
                     "standard format.")

    return {
        "name": getattr(settings, "name", "ESPN league"),
        "teams": int(getattr(settings, "team_count", 12)),
        "qb": int(counts.get("QB", 0)), "rb": int(counts.get("RB", 0)),
        "wr": int(counts.get("WR", 0)), "te": int(counts.get("TE", 0)),
        "flex": int(full_flex + partial_flex + superflex),
        "superflex": bool(superflex),
        "scoring": scoring,
        "slot_counts": {k: v for k, v in counts.items() if v},
        "notes": notes,
    }


def _owner_text(owners) -> str:
    names = []
    for o in owners or []:
        if isinstance(o, dict):
            n = o.get("displayName") or " ".join(
                x for x in (o.get("firstName"), o.get("lastName")) if x)
            if n:
                names.append(n)
        elif isinstance(o, str):
            names.append(o)
    return ", ".join(names)


def league_to_dict(league) -> dict:
    """Convert an espn-api League (or any object shaped like one) to plain dicts."""
    teams = []
    for t in league.teams:
        players = [{
            "espn_id": _idstr(p.playerId), "name": p.name, "pos": p.position,
            "team": getattr(p, "proTeam", ""), "slot": getattr(p, "lineupSlot", ""),
            "injury": getattr(p, "injuryStatus", ""),
        } for p in t.roster]
        teams.append({"team_id": t.team_id, "name": t.team_name,
                      "owner": _owner_text(getattr(t, "owners", [])), "players": players})
    settings = parse_settings(league.settings)
    return {"league_name": settings["name"], "settings": settings, "teams": teams}


def fetch_league(league_id, year, espn_s2=None, swid=None) -> dict:
    """Download a league from ESPN. Raises ImportError / ESPN errors on failure."""
    from espn_api.football import League  # imported lazily so the app still loads without it

    league = League(league_id=int(league_id), year=int(year),
                    espn_s2=(espn_s2 or None), swid=(swid or None))
    return league_to_dict(league)


def friendly_error(exc: Exception) -> str:
    """Turn ESPN / network errors into something a person can act on."""
    name = type(exc).__name__
    if name == "ESPNAccessDenied":
        return ("ESPN denied access. If your league is private, enter both the espn_s2 and "
                "SWID cookies from your browser (they expire, so grab fresh ones).")
    if name == "ESPNInvalidLeague":
        return "ESPN says that league doesn't exist for this season. Check the league ID."
    if isinstance(exc, ValueError):
        return "The league ID should be a number (it's in your league's URL after leagueId=)."
    if isinstance(exc, ImportError):
        return "The espn-api package isn't installed. Add `espn-api` to requirements.txt."
    return f"Couldn't import the league ({name}). ESPN may have changed something, or the cookies may be wrong."


# ------------------------------ Match to our data ----------------------------
def map_team_players(players, xwalk: pd.DataFrame, valid: pd.DataFrame) -> dict:
    """
    Match one ESPN roster to our player IDs.

    `valid` is the pool of players we have current-season stats for. Anything not in
    it (kickers, defenses, or players who haven't played this season) is reported
    rather than silently dropped.
    """
    e2g = {}
    for e, g in zip(xwalk["espn_id"].map(_idstr), xwalk["gsis_id"]):
        if e:
            e2g[e] = g
    name_key = {(_norm_name(n), ps): pid
                for n, ps, pid in zip(valid["player"], valid["pos"], valid["player_id"])}
    valid_ids = set(valid["player_id"])

    ids, unmatched, skipped = [], [], 0
    for p in players:
        if p["pos"] not in OFFENSE:
            skipped += 1  # D/ST, K, etc. aren't part of this app
            continue
        gsis = e2g.get(p["espn_id"]) if p["espn_id"] else None
        if gsis not in valid_ids:
            gsis = name_key.get((_norm_name(p["name"]), p["pos"]))
        if gsis in valid_ids and gsis not in ids:
            ids.append(gsis)
        else:
            unmatched.append(p["name"])
    return {"ids": ids, "unmatched": unmatched, "skipped": skipped}


def map_league(league: dict, xwalk: pd.DataFrame, valid: pd.DataFrame) -> dict:
    """Return {team name: {ids, unmatched, skipped}} for every team in the league."""
    return {t["name"]: map_team_players(t["players"], xwalk, valid) for t in league["teams"]}
