"""
LoL Champion Picker - Recommends the best pick for any role
based on team composition. Predicts enemy roles from lane distribution stats.
"""

import json
import os
import shutil
import sys
import requests as http_requests
import urllib3
from concurrent.futures import ThreadPoolExecutor
from itertools import permutations
from flask import Flask, render_template, jsonify, request

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# When bundled with PyInstaller, put writable data/ next to the exe
# (sys._MEIPASS is the ephemeral temp extraction dir, wiped between runs)
if getattr(sys, "frozen", False):
    BUNDLE_DIR = sys._MEIPASS
    BASE_DIR = os.path.dirname(sys.executable)
    app = Flask(__name__,
                template_folder=os.path.join(BUNDLE_DIR, "templates"),
                static_folder=os.path.join(BUNDLE_DIR, "static"))
else:
    BUNDLE_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = BUNDLE_DIR
    app = Flask(__name__)

DATA_ROOT = os.path.join(BASE_DIR, "data")

ROLES = ["top", "jungle", "middle", "bottom", "support"]

# Ally slot keys (your role is excluded dynamically)
ALLY_SLOT = {"top": "ally_top", "jungle": "ally_jungle", "middle": "ally_mid",
             "bottom": "ally_adc", "support": "ally_support"}

ENEMY_WEIGHT_KEY = {"top": "enemy_top", "jungle": "enemy_jungle", "middle": "enemy_mid",
                    "bottom": "enemy_adc", "support": "enemy_support"}

# Data-driven weights: how much each interaction matters when picking for a role
DEFAULT_WEIGHTS = {
    "top":     {"enemy_top": 0.1655, "enemy_jungle": 0.1025, "enemy_mid": 0.1163, "enemy_adc": 0.0985, "enemy_support": 0.1041,
                "ally_jungle": 0.1070, "ally_mid": 0.1133, "ally_adc": 0.0894, "ally_support": 0.1034},
    "jungle":  {"enemy_top": 0.1149, "enemy_jungle": 0.1138, "enemy_mid": 0.1221, "enemy_adc": 0.1074, "enemy_support": 0.1132,
                "ally_top": 0.1101, "ally_mid": 0.1187, "ally_adc": 0.0969, "ally_support": 0.1029},
    "middle":  {"enemy_top": 0.1160, "enemy_jungle": 0.1148, "enemy_mid": 0.1575, "enemy_adc": 0.1008, "enemy_support": 0.1129,
                "ally_top": 0.0995, "ally_jungle": 0.1039, "ally_adc": 0.0900, "ally_support": 0.1046},
    "bottom":  {"enemy_top": 0.1086, "enemy_jungle": 0.0945, "enemy_mid": 0.1122, "enemy_adc": 0.1214, "enemy_support": 0.1366,
                "ally_top": 0.1042, "ally_jungle": 0.0901, "ally_mid": 0.1090, "ally_support": 0.1234},
    "support": {"enemy_top": 0.1116, "enemy_jungle": 0.1002, "enemy_mid": 0.1182, "enemy_adc": 0.1146, "enemy_support": 0.1239,
                "ally_top": 0.1085, "ally_jungle": 0.0980, "ally_mid": 0.1094, "ally_adc": 0.1156},
}

MIN_GAMES = 50
N_RELIABLE = 5000  # games needed for full confidence in a matchup


def reliability_factor(n):
    """Scale factor 0-1 based on sample size. Linearly ramps up to N_RELIABLE."""
    return min(1.0, n / N_RELIABLE)

_data_cache = {}


def load_data(tier):
    if tier in _data_cache:
        return _data_cache[tier]
    tier_dir = os.path.join(DATA_ROOT, tier)
    if not os.path.isdir(tier_dir):
        return None
    data = {}
    for key in ["champions", "counters", "synergy", "overall", "lane_dist", "meta"]:
        path = os.path.join(tier_dir, f"{key}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data[key] = json.load(f)
    _data_cache[tier] = data
    return data


def available_tiers():
    tiers = []
    if os.path.isdir(DATA_ROOT):
        for name in sorted(os.listdir(DATA_ROOT)):
            if os.path.isdir(os.path.join(DATA_ROOT, name)) and \
               os.path.exists(os.path.join(DATA_ROOT, name, "meta.json")):
                tiers.append(name)
    return tiers


def predict_enemy_roles(enemy_champs, lane_dist):
    """
    Given a list of enemy champion names, predict the most likely role assignment.
    Uses lane distribution stats to find the assignment with highest total probability.
    Returns {champ_name: {role: probability}} for each champion.
    """
    if not enemy_champs:
        return {}

    # Get lane distribution for each enemy
    champ_lanes = {}
    for champ in enemy_champs:
        dist = lane_dist.get(champ, {})
        champ_lanes[champ] = {r: dist.get(r, 0) / 100.0 for r in ROLES}

    if len(enemy_champs) == 1:
        # Single champion: just return their lane distribution
        return {enemy_champs[0]: champ_lanes[enemy_champs[0]]}

    # For 2-5 champions: per-champion role marginals via brute force over
    # all role permutations (at most 5! = 120, very fast), weighted by
    # joint probability under an independence assumption.
    champs = list(enemy_champs)
    n = len(champs)
    role_probs = {champ: {r: 0.0 for r in ROLES} for champ in champs}
    total_weight = 0.0

    for perm in permutations(ROLES, n):
        score = 1.0
        for champ, role in zip(champs, perm):
            score *= max(champ_lanes[champ].get(role, 0), 0.001)
        total_weight += score
        for champ, role in zip(champs, perm):
            role_probs[champ][role] += score

    if total_weight > 0:
        for champ in champs:
            for role in ROLES:
                role_probs[champ][role] /= total_weight

    return role_probs


def ban_adjusted_wr(champ_name, picking_role, bans, data):
    """
    Recalculate a champion's effective WR after removing banned champions' games.
    Per-role adjustments are summed (each enemy role is an independent encounter,
    so lifts add rather than average).

    Per role: adjusted_wr_role = (wr - norm_pr * wr_vs) / (1 - norm_pr)
    where norm_pr = champion's PR / sum(all PRs in that role) = true encounter rate.
    Bans without matchup data are skipped so they don't bias the denominator.
    """
    champ_overall = data.get("overall", {}).get(picking_role, {}).get(champ_name, {})
    base_wr = champ_overall.get("wr", 50)
    if not bans:
        return base_wr

    champ_counters = data["counters"].get(picking_role, {}).get(champ_name, {})
    all_overall = data.get("overall", {})
    ban_set = set(bans)

    total_adjustment = 0.0

    for role in ROLES:
        role_overall = all_overall.get(role, {})
        # Total PR in this role (sums to ~200%, both teams)
        total_role_pr = sum(s.get("pr", 0) for s in role_overall.values())
        if total_role_pr <= 0:
            continue

        removed_pr = 0.0
        removed_wr_contribution = 0.0

        for banned in ban_set:
            banned_stats = role_overall.get(banned, {})
            raw_pr = banned_stats.get("pr", 0)
            if raw_pr <= 0:
                continue

            matchup = champ_counters.get(role, {}).get(banned, {})
            if matchup.get("n", 0) < MIN_GAMES:
                continue  # skip: no reliable matchup data

            norm_pr = raw_pr / total_role_pr  # true encounter rate
            removed_pr += norm_pr
            removed_wr_contribution += norm_pr * matchup["vsWr"]

        if removed_pr > 0 and removed_pr < 1.0:
            role_adjusted = (base_wr - removed_wr_contribution) / (1.0 - removed_pr)
            total_adjustment += role_adjusted - base_wr

    return base_wr + total_adjustment


def score_champion(champ_name, picking_role, ally_inputs, enemy_champs, enemy_role_probs,
                   weights, data, bidirectional):
    """
    Score a candidate champion. Enemy scores are weighted by role probability.
    ally_inputs: {slot_key: champ_name} e.g. {"ally_top": "Aatrox"}
    enemy_champs: list of enemy champ names
    enemy_role_probs: {champ: {role: prob}} from predict_enemy_roles
    """
    counters = data["counters"]
    synergy_data = data["synergy"]

    components = {}
    total_score = 0.0       # sum of (weight * reliability * d2) - effective contribution
    total_max_weight = 0.0  # sum of weight for all filled inputs (no reliability scaling)

    # --- Ally synergy (fixed roles) ---
    ally_role_map = {"ally_top": "top", "ally_jungle": "jungle", "ally_mid": "middle",
                     "ally_adc": "bottom", "ally_support": "support"}

    for slot_key, ally_name in ally_inputs.items():
        if not ally_name:
            continue
        ally_role = ally_role_map.get(slot_key)
        if not ally_role:
            continue
        weight = weights.get(slot_key, 0)
        if weight == 0:
            continue

        fwd = synergy_data.get(picking_role, {}).get(champ_name, {}).get(ally_role, {}).get(ally_name)
        fwd_d2 = fwd["d2"] if fwd and fwd["n"] >= MIN_GAMES else None
        fwd_wr = fwd["wr"] if fwd else None
        fwd_n = fwd["n"] if fwd else 0

        if bidirectional:
            rev = synergy_data.get(ally_role, {}).get(ally_name, {}).get(picking_role, {}).get(champ_name)
            rev_d2 = rev["d2"] if rev and rev["n"] >= MIN_GAMES else None
            rev_n = rev["n"] if rev else 0
            if fwd_d2 is not None and rev_d2 is not None:
                d2 = (fwd_d2 + rev_d2) / 2
                wr = (fwd_wr + rev["wr"]) / 2
                # fwd and rev sample the same games from mirrored POVs,
                # so take the larger n instead of summing
                n = max(fwd_n, rev_n)
            elif fwd_d2 is not None:
                d2, wr, n = fwd_d2, fwd_wr, fwd_n
            elif rev_d2 is not None:
                d2, wr, n = rev_d2, rev.get("wr", 0), rev_n
            else:
                d2 = None
        else:
            d2, wr, n = fwd_d2, fwd_wr, fwd_n

        if d2 is not None:
            rel = reliability_factor(n)
            components[slot_key] = {"d2": round(d2, 2), "winrate": round(wr, 2) if wr else 0, "games": n, "weight": weight, "reliability": round(rel, 2)}
            total_score += weight * rel * d2  # reliability scales the contribution
            total_max_weight += weight         # but denominator uses full weight
        elif fwd and fwd["n"] > 0:
            components[slot_key] = {"d2": fwd["d2"], "winrate": fwd["wr"], "games": fwd["n"], "weight": 0, "note": "low sample"}

    # --- Enemy counter (probabilistic roles) ---
    for enemy_name in enemy_champs:
        probs = enemy_role_probs.get(enemy_name, {})

        # Weighted average across possible roles for this enemy
        enemy_d2 = 0.0
        enemy_wr = 0.0
        enemy_n = 0
        enemy_total_w = 0.0
        has_data = False

        for enemy_role, prob in probs.items():
            if prob < 0.01:
                continue
            weight_key = ENEMY_WEIGHT_KEY[enemy_role]
            base_weight = weights.get(weight_key, 0)
            effective_weight = base_weight * prob

            fwd = counters.get(picking_role, {}).get(champ_name, {}).get(enemy_role, {}).get(enemy_name)
            fwd_d2 = fwd["d2"] if fwd and fwd["n"] >= MIN_GAMES else None
            fwd_wr = fwd["vsWr"] if fwd else None
            fwd_n = fwd["n"] if fwd else 0

            if bidirectional:
                rev = counters.get(enemy_role, {}).get(enemy_name, {}).get(picking_role, {}).get(champ_name)
                rev_d2 = -rev["d2"] if rev and rev["n"] >= MIN_GAMES else None
                rev_n = rev["n"] if rev else 0
                if fwd_d2 is not None and rev_d2 is not None:
                    d2 = (fwd_d2 + rev_d2) / 2
                    wr = (fwd_wr + (100 - rev["vsWr"])) / 2
                    # fwd and rev sample the same games from mirrored POVs,
                    # so take the larger n instead of summing
                    n = max(fwd_n, rev_n)
                elif fwd_d2 is not None:
                    d2, wr, n = fwd_d2, fwd_wr, fwd_n
                elif rev_d2 is not None:
                    d2, wr, n = rev_d2, 100 - rev["vsWr"], rev_n
                else:
                    continue
            else:
                if fwd_d2 is None:
                    continue
                d2, wr, n = fwd_d2, fwd_wr, fwd_n

            rel = reliability_factor(n)
            enemy_d2 += effective_weight * rel * d2  # reliability scales contribution
            enemy_wr += effective_weight * (wr or 0)  # no reliability - display value
            enemy_n += n
            enemy_total_w += effective_weight
            has_data = True

        if has_data and enemy_total_w > 0:
            avg_d2 = enemy_d2 / enemy_total_w
            avg_wr = enemy_wr / enemy_total_w  # raw weighted winrate for display
            best_role = max(probs, key=probs.get) if probs else "?"
            components[f"enemy_{enemy_name}"] = {
                "d2": round(avg_d2, 2),
                "winrate": round(avg_wr, 2),
                "games": enemy_n,
                "weight": round(enemy_total_w, 4),
                "predicted_role": best_role,
                "role_probs": {r: round(p, 3) for r, p in probs.items() if p >= 0.01},
            }
            total_score += enemy_d2  # already weighted with reliability
            total_max_weight += enemy_total_w  # full weight for denominator

    if total_max_weight == 0:
        return None

    coverage = round(total_max_weight / max(sum(weights.values()), 0.01) * 100)

    return {
        "champion": champ_name,
        "score": round(total_score / total_max_weight, 2),
        "components": components,
        "data_coverage": min(coverage, 100),
    }


@app.route("/")
def index():
    tiers = available_tiers()
    default_tier = tiers[0] if tiers else "emerald_plus"
    data = load_data(default_tier)
    if not data:
        return "No data available. Run: python scrape_data.py", 500

    meta = data["meta"]
    pools = data["champions"]["pools"]
    id_to_image = data["champions"]["id_to_image"]
    name_to_id = data["champions"]["name_to_id"]
    name_to_image = {n: id_to_image.get(str(c)) for n, c in name_to_id.items() if id_to_image.get(str(c))}

    # All champion names for enemy dropdown (any champ can be enemy)
    all_champs = sorted(name_to_id.keys())

    return render_template("index.html",
        pools=pools, meta=meta, tiers=tiers, default_tier=default_tier,
        default_weights=DEFAULT_WEIGHTS, ddragon_version=meta["ddragon_version"],
        name_to_image=name_to_image, roles=ROLES, all_champs=all_champs)


@app.route("/recommend")
def recommend():
    tier = request.args.get("tier", "emerald_plus")
    data = load_data(tier)
    if not data:
        return jsonify({"error": f"No data for tier '{tier}'"}), 404

    picking_role = request.args.get("picking_role", "support")
    bidirectional = request.args.get("bidirectional", "true") == "true"

    # Ally inputs (fixed roles)
    ally_role_map = {"ally_top": "top", "ally_jungle": "jungle", "ally_mid": "middle",
                     "ally_adc": "bottom", "ally_support": "support"}
    ally_inputs = {}
    for slot_key in ally_role_map:
        if slot_key == ALLY_SLOT.get(picking_role):
            continue  # skip our own role
        val = request.args.get(slot_key, "").strip()
        if val:
            ally_inputs[slot_key] = val

    # Enemy inputs (no roles, just champion names)
    enemy_champs = []
    for i in range(1, 6):
        val = request.args.get(f"enemy_{i}", "").strip()
        if val:
            enemy_champs.append(val)

    # Bans
    bans = []
    for i in range(1, 11):
        val = request.args.get(f"ban_{i}", "").strip()
        if val:
            bans.append(val)

    # Weights
    role_defaults = DEFAULT_WEIGHTS.get(picking_role, {})
    weights = {}
    for key, default in role_defaults.items():
        try:
            weights[key] = float(request.args.get(f"w_{key}", default))
        except (ValueError, TypeError):
            weights[key] = default

    # Predict enemy roles
    lane_dist = data.get("lane_dist", {})
    enemy_role_probs = predict_enemy_roles(enemy_champs, lane_dist)

    pool = data["champions"]["pools"].get(picking_role, {})
    overall = data.get("overall", {}).get(picking_role, {})
    # PR-weighted mean WR in this role — the expected WR of a typical game
    total_pr = sum(s.get("pr", 0) for s in overall.values())
    if overall and total_pr > 0:
        avg_wr = sum(s.get("wr", 50) * s.get("pr", 0) for s in overall.values()) / total_pr
    else:
        avg_wr = 50

    # Filter to champions with statistically reliable data (>= 0.04% PR ≈ 10K+ games)
    viable_champs = set()
    for champ_name in pool.values():
        stats = overall.get(champ_name, {})
        if stats and stats.get("pr", 0) >= 0.04:
            viable_champs.add(champ_name)

    # Exclude already-picked and banned champs
    excluded = set(ally_inputs.values()) | set(enemy_champs) | set(bans)

    # Blind pick / bans only - baseline WR ranking (adjusted for bans)
    if not ally_inputs and not enemy_champs:
        results = []
        for champ_name in viable_champs:
            if champ_name in excluded:
                continue
            stats = overall.get(champ_name, {})
            if stats:
                adj_wr = ban_adjusted_wr(champ_name, picking_role, bans, data) if bans else stats["wr"]
                results.append({
                    "champion": champ_name,
                    "score": round(adj_wr - avg_wr, 2),
                    "components": {},
                    "data_coverage": 100,
                    "overall_wr": stats["wr"],
                    "pick_rate": stats["pr"],
                    "ban_rate": stats.get("br", 0),
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        return jsonify({"recommendations": results, "inputs": {}, "weights": weights,
                        "mode": "blind", "picking_role": picking_role, "enemy_roles": {},
                        "bans": bans})

    # Matchup mode - blend baseline WR with matchup score based on draft coverage
    n_inputs = len(ally_inputs) + len(enemy_champs)
    max_inputs = 9  # 4 allies + 5 enemies
    # Principled ranking: rank = baseline + matchup_score * coverage
    # - baseline = overall_wr - avg_wr (champion strength, always present)
    # - matchup_score = weighted avg d2 across known interactions
    # - coverage = fraction of total interaction weight covered by known inputs
    # No arbitrary blend percentages. Coverage is fully determined by
    # the data-driven weights and which inputs are filled.

    results = []
    for champ_name in viable_champs:
        if champ_name in excluded:
            continue
        result = score_champion(champ_name, picking_role, ally_inputs, enemy_champs,
                                enemy_role_probs, weights, data, bidirectional)
        if result:
            stats = overall.get(champ_name, {})
            result["overall_wr"] = stats.get("wr", 0)
            result["pick_rate"] = stats.get("pr", 0)
            result["ban_rate"] = stats.get("br", 0)
            adj_wr = ban_adjusted_wr(champ_name, picking_role, bans, data) if bans else result["overall_wr"]
            baseline = adj_wr - avg_wr
            matchup = result["score"]
            coverage = result["data_coverage"] / 100.0
            result["rank_value"] = round(baseline + matchup * coverage, 2)
            results.append(result)

    results.sort(key=lambda x: x["rank_value"], reverse=True)

    # Format enemy role predictions for response
    enemy_roles_display = {}
    for champ, probs in enemy_role_probs.items():
        best = max(probs, key=probs.get)
        enemy_roles_display[champ] = {
            "predicted": best,
            "probs": {r: round(p, 3) for r, p in probs.items() if p >= 0.01},
        }

    return jsonify({"recommendations": results, "inputs": {**ally_inputs, "enemies": enemy_champs},
                    "weights": weights, "mode": "matchup", "picking_role": picking_role,
                    "bidirectional": bidirectional, "enemy_roles": enemy_roles_display})


LOCKFILE_PATHS = [
    "C:/Riot Games/League of Legends/lockfile",
    "D:/Riot Games/League of Legends/lockfile",
]


def get_lcu_connection():
    """Read the League client lockfile and return (port, auth_token) or None."""
    for path in LOCKFILE_PATHS:
        if os.path.exists(path):
            with open(path, "r") as f:
                parts = f.read().strip().split(":")
            if len(parts) >= 4:
                return parts[2], parts[3]
    return None, None


def get_champ_name_by_id(champ_id, data):
    """Convert a champion ID (int) to display name."""
    if not champ_id:
        return None
    return data.get("champions", {}).get("id_to_name", {}).get(str(champ_id))


@app.route("/lcu/session")
def lcu_session():
    """Read current champ select state from the League client."""
    port, token = get_lcu_connection()
    if not port:
        return jsonify({"status": "no_client", "message": "League client not found"}), 404

    try:
        auth = ("riot", token)
        base = f"https://127.0.0.1:{port}"
        r = http_requests.get(f"{base}/lol-champ-select/v1/session", auth=auth, verify=False, timeout=3)
    except Exception:
        return jsonify({"status": "no_client", "message": "Cannot connect to League client"}), 404

    if r.status_code == 404:
        return jsonify({"status": "no_session", "message": "Not in champion select"})

    if r.status_code != 200:
        return jsonify({"status": "error", "message": f"LCU returned {r.status_code}"}), 500

    session = r.json()

    tiers = available_tiers()
    data = load_data(tiers[0]) if tiers else None
    if not data:
        return jsonify({"status": "error", "message": "No data loaded"}), 500

    local_player_cell = session.get("localPlayerCellId", -1)
    my_team = []
    their_team = []

    for member in session.get("myTeam", []):
        picked_id = member.get("championId", 0)
        hover_id = member.get("championPickIntent", 0)
        is_me = member.get("cellId") == local_player_cell
        assigned_pos = member.get("assignedPosition", "")  # e.g. "top", "jungle", "middle", "bottom", "utility"

        # Map Riot's "utility" to our "support"
        if assigned_pos == "utility":
            assigned_pos = "support"

        picked_name = get_champ_name_by_id(picked_id, data) if picked_id else None
        hover_name = get_champ_name_by_id(hover_id, data) if hover_id and not picked_id else None

        my_team.append({
            "champion": picked_name or hover_name,
            "picked": picked_name,
            "hovered": hover_name,
            "is_me": is_me,
            "assigned_position": assigned_pos,
            "spell1": member.get("spell1Id"),
            "spell2": member.get("spell2Id"),
        })

    for member in session.get("theirTeam", []):
        picked_id = member.get("championId", 0)
        picked_name = get_champ_name_by_id(picked_id, data) if picked_id else None
        their_team.append({
            "champion": picked_name,
            "picked": bool(picked_name),
        })

    # Parse bans
    bans = []
    for action_group in session.get("actions", []):
        for action in action_group:
            if action.get("type") == "ban" and action.get("completed"):
                champ_id = action.get("championId")
                name = get_champ_name_by_id(champ_id, data)
                if name and name not in bans:
                    bans.append(name)

    # Count confirmed enemy picks
    enemy_picks_confirmed = sum(1 for m in their_team if m["picked"])
    # My champion (picked, not just hovered)
    me = next((m for m in my_team if m["is_me"]), None)
    my_pick_confirmed = bool(me and me.get("picked"))

    return jsonify({
        "status": "in_select",
        "my_team": my_team,
        "their_team": their_team,
        "bans": bans,
        "all_enemies_picked": enemy_picks_confirmed == 5,
        "my_pick_confirmed": my_pick_confirmed,
    })


A3_API = "https://a3.lolalytics.com/mega/"
A3_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://lolalytics.com/",
    "Origin": "https://lolalytics.com",
}

BUILD_CACHE_DIR = os.path.join(BASE_DIR, "data", "build_cache")
BUILD_CACHE_TTL = 86400  # 24 hours in seconds


def _build_cache_path(my_slug, my_lane, enemy_slug, enemy_lane, tier, patch):
    """Return the cache file path for a matchup build."""
    key = f"{my_slug}_{my_lane}_vs_{enemy_slug}_{enemy_lane}_{tier}_{patch}"
    return os.path.join(BUILD_CACHE_DIR, f"{key}.json")


def _load_from_cache(cache_path):
    """Load cached build if it exists and is fresh (< 24h old)."""
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        ts = cached.get("_cached_at", 0)
        import time
        if time.time() - ts < BUILD_CACHE_TTL:
            return cached.get("data")
    except Exception:
        pass
    return None


def _save_to_cache(cache_path, data):
    """Save build data to cache with timestamp."""
    os.makedirs(BUILD_CACHE_DIR, exist_ok=True)
    import time
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"_cached_at": time.time(), "data": data}, f)
    except Exception:
        pass


def fetch_vs_build(my_champ, my_lane, enemy_name, enemy_lane, tier, patch):
    """Fetch matchup-specific build, using 24h cache when available."""
    my_slug = my_champ.lower().replace("'", "").replace(" ", "").replace(".", "")
    enemy_slug = enemy_name.lower().replace("'", "").replace(" ", "").replace(".", "")

    cache_path = _build_cache_path(my_slug, my_lane, enemy_slug, enemy_lane, tier, patch)
    cached = _load_from_cache(cache_path)
    if cached:
        return {"enemy": enemy_name, "enemy_lane": enemy_lane, "data": cached, "cached": True}

    params = {
        "ep": "build-full", "v": 1,
        "c": my_slug, "lane": my_lane,
        "vs": enemy_slug, "vslane": enemy_lane,
        "tier": tier, "patch": patch,
    }
    try:
        r = http_requests.get(A3_API, params=params, headers=A3_HEADERS, timeout=10)
        data = r.json()
        if data.get("summary"):
            _save_to_cache(cache_path, data)
            return {"enemy": enemy_name, "enemy_lane": enemy_lane, "data": data, "cached": False}
    except Exception:
        pass
    return {"enemy": enemy_name, "enemy_lane": enemy_lane, "data": None, "cached": False}


def fetch_unconditioned_build(my_champ, my_lane, tier, patch):
    """Fetch the champion-lane build with no enemy filter (for intrinsic stats)."""
    my_slug = my_champ.lower().replace("'", "").replace(" ", "").replace(".", "")
    cache_path = os.path.join(BUILD_CACHE_DIR,
                              f"{my_slug}_{my_lane}_uncond_{tier}_{patch}.json")
    cached = _load_from_cache(cache_path)
    if cached:
        return {"data": cached, "cached": True}

    params = {
        "ep": "build-full", "v": 1,
        "c": my_slug, "lane": my_lane,
        "tier": tier, "patch": patch,
    }
    try:
        r = http_requests.get(A3_API, params=params, headers=A3_HEADERS, timeout=10)
        data = r.json()
        if data.get("summary"):
            _save_to_cache(cache_path, data)
            return {"data": data, "cached": False}
    except Exception:
        pass
    return {"data": None, "cached": False}


# Bayesian shrinkage: per-enemy rune/spell WRs are pulled toward the champion's
# baseline WR (or rune's intrinsic WR) with prior strength K. n=K → half-trusted;
# n>>K → fully trusted.
# K=1000 derived empirically (calibrate_shrink_k.py): the observed variance of
# matchup-specific shifts (after subtracting binomial sample noise) is ~2.4 pp²
# across 195 (champ, rune) pairs. From the Beta-Binomial relation
# prior_var = baseline*(100-baseline)/(K+1), this gives K ≈ 1040.
SHRINK_K = 1000
# Minimum aggregate pick rate (%) for a summoner pair to be eligible
MIN_PAIR_PR = 3.0
# Per-matchup minimum pick rate / sample size for a rune entry to enter aggregation.
# Keystones below ~15% PR are dominated by skilled-player self-selection (players
# who deliberately deviate from the meta tend to be above-average), inflating
# the rune's apparent WR. Minors/shards have weaker selection bias since they're
# more interchangeable, so a softer 5% threshold is OK.
MIN_RUNE_PR = 5.0
MIN_KEYSTONE_PR = 15.0
MIN_RUNE_N = 30
_RIOT_RECOMMENDED_CACHE = None


def _load_riot_recommended():
    """Load the latest patch's Riot-recommended pages (or empty if missing)."""
    global _RIOT_RECOMMENDED_CACHE
    if _RIOT_RECOMMENDED_CACHE is not None:
        return _RIOT_RECOMMENDED_CACHE
    base = os.path.join(BASE_DIR, "data", "riot_recommended")
    if not os.path.isdir(base):
        _RIOT_RECOMMENDED_CACHE = {}
        return _RIOT_RECOMMENDED_CACHE
    patches = sorted(os.listdir(base))
    for patch in reversed(patches):
        p = os.path.join(base, patch, "runes.json")
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    _RIOT_RECOMMENDED_CACHE = json.load(f)
                    return _RIOT_RECOMMENDED_CACHE
            except Exception:
                continue
    _RIOT_RECOMMENDED_CACHE = {}
    return _RIOT_RECOMMENDED_CACHE


def get_recommended_perk_slots(champion_name, lane):
    """Return {rune_stats_key (str): set of slots ('pri'|'sec')} for runes that
    Riot recommends for this champ+lane, unioned across 1-3 recommended pages.

    Each page's 9-element perks array maps to slots:
      [0] keystone           → 'pri'
      [1..3] primary minors  → 'pri'
      [4..5] secondary minors→ 'sec'
      [6] shard row 1        → 'pri'
      [7] shard row 2 (flex) → 'pri', stored as '{id}f'
      [8] shard row 3        → 'pri'

    Used only for informational "differs from Riot's recommendation" flags
    in hidden_gems analysis, not for scoring.
    """
    cache = _load_riot_recommended()
    pages = cache.get("pages", {}).get(champion_name, {}).get(lane, [])
    out = {}
    for page in pages:
        perks = page.get("perk_ids") or []
        for i, pid in enumerate(perks):
            if pid is None:
                continue
            if i in (4, 5):
                slot = "sec"
                key = str(pid)
            elif i == 7:
                slot = "pri"
                key = f"{pid}f"
            else:
                slot = "pri"
                key = str(pid)
            out.setdefault(key, set()).add(slot)
    return out


def build_enemy_weights(my_lane, enemies):
    """
    Given picking lane and [(enemy_name, enemy_lane), ...], return a parallel list
    of renormalized weights from DEFAULT_WEIGHTS[my_lane]. Sums to 1.
    """
    role_weights = DEFAULT_WEIGHTS.get(my_lane, {})
    raw = []
    for _, enemy_lane in enemies:
        key = ENEMY_WEIGHT_KEY.get(enemy_lane)
        raw.append(role_weights.get(key, 0.0) if key else 0.0)
    total = sum(raw)
    if total <= 0:
        # Fallback: equal weights
        return [1.0 / len(enemies)] * len(enemies) if enemies else []
    return [w / total for w in raw]


def _shrunk_delta(wr, n, baseline):
    """Shrink wr toward baseline with prior strength SHRINK_K, then return delta."""
    if n <= 0:
        return 0.0
    return (wr * n + baseline * SHRINK_K) / (n + SHRINK_K) - baseline


def _modal_keystone(rune_stats, min_n=100):
    """Return the rune_id (str) of the keystone with highest PR in this matchup."""
    keystone_ids = [str(rid) for tree in RUNE_TREES.values() for rid in tree["rows"][0]]
    best_pr = -1.0
    best_id = None
    for kid in keystone_ids:
        info = rune_stats.get(kid)
        if not info:
            continue
        entry = info[0] if isinstance(info, list) and info else None
        if not entry or len(entry) < 3:
            continue
        pr, wr, n = entry
        if n >= min_n and pr > best_pr:
            best_pr = pr
            best_id = kid
    return best_id


def combine_rune_stats(build_results, enemy_weights, unconditioned_data,
                       champion_name=None, picking_lane=None):
    """
    Score each rune as: intrinsic_delta + Σ w_i × shrunk_matchup_shift_i

    intrinsic_delta = shrink(wr − baseline)
    matchup_shift_i = shrink(wr_i − wr_intrinsic)
    """
    rune_info = {"pri": {}, "sec": {}}
    if not unconditioned_data:
        return rune_info

    keystone_set = {str(rid) for tree in RUNE_TREES.values() for rid in tree["rows"][0]}
    intrinsic_baseline = unconditioned_data.get("avgWr", 50.0)
    intrinsic_stats = unconditioned_data.get("runes", {}).get("stats", {})

    # Riot's recommendation status — informational only, used by hidden_gems
    # analysis to flag "our pick differs from Riot's".
    if champion_name and picking_lane:
        recommended_slots = get_recommended_perk_slots(champion_name, picking_lane)
    else:
        recommended_slots = {}

    rune_intrinsic = {"pri": {}, "sec": {}}
    for rune_id, entries in intrinsic_stats.items():
        is_keystone = rune_id in keystone_set
        min_pr = MIN_KEYSTONE_PR if is_keystone else MIN_RUNE_PR
        for i, entry in enumerate(entries):
            slot = "pri" if i == 0 else "sec"
            pr, wr, n = entry
            if n < MIN_RUNE_N:
                continue
            riot_recommended = slot in recommended_slots.get(rune_id, ())
            rune_intrinsic[slot][rune_id] = {
                "wr": wr,
                "pr": pr,
                "n": n,
                "eligible": pr >= min_pr,
                "recommended": riot_recommended,
            }

    rune_matchup = {"pri": {}, "sec": {}}
    for result, weight in zip(build_results, enemy_weights):
        data = result.get("data")
        if not data or weight <= 0:
            continue
        rune_stats = data.get("runes", {}).get("stats", {})
        for rune_id, entries in rune_stats.items():
            for i, entry in enumerate(entries):
                slot = "pri" if i == 0 else "sec"
                if rune_id not in rune_intrinsic[slot]:
                    continue
                pr, wr, n = entry
                if n < MIN_RUNE_N:
                    continue
                intrinsic_wr = rune_intrinsic[slot][rune_id]["wr"]
                shrinkage = n / (n + SHRINK_K)
                shrunk_shift = shrinkage * (wr - intrinsic_wr)
                cell = rune_matchup[slot].setdefault(
                    rune_id, {"shift": 0.0, "n": 0, "pr": 0.0, "weight": 0.0}
                )
                cell["shift"] += weight * shrunk_shift
                cell["n"] += n
                cell["pr"] += weight * pr
                cell["weight"] += weight

    for slot in ("pri", "sec"):
        for rune_id, intr in rune_intrinsic[slot].items():
            raw_intrinsic_delta = intr["wr"] - intrinsic_baseline
            intr_shrinkage = intr["n"] / (intr["n"] + SHRINK_K)
            intrinsic_delta = intr_shrinkage * raw_intrinsic_delta
            matchup = rune_matchup[slot].get(rune_id, {"shift": 0.0, "n": 0, "pr": 0.0})
            total_delta = intrinsic_delta + matchup["shift"]
            rune_info[slot][rune_id] = {
                "delta": total_delta,
                "intrinsic_delta": intrinsic_delta,
                "matchup_shift": matchup["shift"],
                "wr": intr["wr"],
                "n": intr["n"] + matchup["n"],
                "pr": intr["pr"],
                "eligible": intr["eligible"],
                "recommended": intr["recommended"],
            }
    return rune_info


def combine_summoner_pairs(build_results, enemy_weights):
    """
    Aggregate summoner-spell pair stats across enemy matchups.
    Each build response has a `spells` list of [pair_id, wr, pr, n].
    Returns {pair_id: {delta, wr, n, pr, pair: [sid1, sid2]}}.
    """
    pairs = {}  # pair_id -> list of (weight, pr, wr, n, baseline)
    for result, weight in zip(build_results, enemy_weights):
        data = result.get("data")
        if not data or weight <= 0:
            continue
        baseline = data.get("avgWr", 50.0)
        for entry in data.get("spells", []) or []:
            if not isinstance(entry, (list, tuple)) or len(entry) < 4:
                continue
            pair_id, wr, pr, n = entry[0], entry[1], entry[2], entry[3]
            if n <= 0:
                continue
            pairs.setdefault(pair_id, []).append((weight, pr, wr, n, baseline))

    result = {}
    for pair_id, rows in pairs.items():
        total_delta = 0.0
        total_pr = 0.0
        total_n = 0
        total_wr_n = 0.0
        for w, pr, wr, n, base in rows:
            total_delta += w * _shrunk_delta(wr, n, base)
            total_pr += w * pr
            total_n += n
            total_wr_n += wr * n
        try:
            sid1, sid2 = (int(x) for x in str(pair_id).split("_"))
        except ValueError:
            continue
        result[pair_id] = {
            "pair": [sid1, sid2],
            "delta": total_delta,
            "wr": total_wr_n / total_n if total_n > 0 else 0.0,
            "n": total_n,
            "pr": total_pr,
        }
    return result


def pick_optimal_summoner_pair(pair_stats):
    """
    Choose the best summoner pair: filter to pairs meeting MIN_PAIR_PR, rank by delta.
    Returns top pick + a short list of top alternatives.
    """
    eligible = [p for p in pair_stats.values() if p["pr"] >= MIN_PAIR_PR]
    if not eligible:
        # Fallback: most-used pair wins, even if it's under the threshold
        eligible = list(pair_stats.values())
    if not eligible:
        return None, []
    eligible.sort(key=lambda p: -p["delta"])
    best = eligible[0]
    top_n = [
        {
            "pair": p["pair"],
            "delta": round(p["delta"], 2),
            "wr": round(p["wr"], 2),
            "pr": round(p["pr"], 2),
            "n": p["n"],
        }
        for p in eligible[:5]
    ]
    return best, top_n


ITEM_SLOTS = ("item1", "item2", "item3", "item4", "item5")


def combine_item_slots(build_results, enemy_weights):
    """
    Per-slot item aggregation across enemy matchups.
    Each field (boots, item1..item5) is a list of [item_id, wr, pr, n, avg_time].
    Returns {slot_key: {item_id: {delta, wr, n, pr, avg_time}}}.
    """
    slots = {k: {} for k in ("boots",) + ITEM_SLOTS}
    agg_time = {k: {} for k in slots}  # slot -> item_id -> (total w*pr*time, total w*pr)

    for result, weight in zip(build_results, enemy_weights):
        data = result.get("data")
        if not data or weight <= 0:
            continue
        baseline = data.get("avgWr", 50.0)
        for slot_key in slots:
            entries = data.get(slot_key) or []
            if not isinstance(entries, list):
                continue
            bucket = slots[slot_key]
            tbucket = agg_time[slot_key]
            for entry in entries:
                if not isinstance(entry, (list, tuple)) or len(entry) < 4:
                    continue
                item_id = entry[0]
                wr = entry[1]
                pr = entry[2]
                n = entry[3]
                time_min = entry[4] if len(entry) >= 5 else 0
                if n <= 0:
                    continue
                cell = bucket.setdefault(item_id,
                                         {"delta": 0.0, "pr": 0.0, "n": 0, "wr_n": 0.0})
                cell["delta"] += weight * _shrunk_delta(wr, n, baseline)
                cell["pr"] += weight * pr
                cell["n"] += n
                cell["wr_n"] += wr * n
                if time_min:
                    t = tbucket.setdefault(item_id, [0.0, 0.0])
                    t[0] += weight * pr * time_min
                    t[1] += weight * pr

    out = {}
    for slot_key, bucket in slots.items():
        out[slot_key] = {}
        for iid, cell in bucket.items():
            t = agg_time[slot_key].get(iid)
            avg_time = (t[0] / t[1]) if t and t[1] > 0 else 0.0
            out[slot_key][iid] = {
                "delta": cell["delta"],
                "wr": cell["wr_n"] / cell["n"] if cell["n"] > 0 else 0.0,
                "n": cell["n"],
                "pr": cell["pr"],
                "avg_time": avg_time,
            }
    return out


def combine_start_sets(build_results, enemy_weights):
    """
    Aggregate starting item SETS (e.g. '1055_2003' = Doran's Blade + Refillable).
    Returns {set_id: {items: [ids...], delta, wr, n, pr}}.
    """
    sets = {}
    for result, weight in zip(build_results, enemy_weights):
        data = result.get("data")
        if not data or weight <= 0:
            continue
        baseline = data.get("avgWr", 50.0)
        for entry in data.get("startSet") or []:
            if not isinstance(entry, (list, tuple)) or len(entry) < 4:
                continue
            set_id, wr, pr, n = entry[0], entry[1], entry[2], entry[3]
            if n <= 0:
                continue
            try:
                items = [int(x) for x in str(set_id).split("_")]
            except ValueError:
                continue
            cell = sets.setdefault(set_id,
                                   {"items": items, "delta": 0.0, "pr": 0.0,
                                    "n": 0, "wr_n": 0.0})
            cell["delta"] += weight * _shrunk_delta(wr, n, baseline)
            cell["pr"] += weight * pr
            cell["n"] += n
            cell["wr_n"] += wr * n

    out = {sid: {
        "items": cell["items"],
        "delta": cell["delta"],
        "wr": cell["wr_n"] / cell["n"] if cell["n"] > 0 else 0.0,
        "n": cell["n"],
        "pr": cell["pr"],
    } for sid, cell in sets.items()}
    return out


MIN_ITEM_PR = 3.0  # min aggregate pick rate for an item to be considered


def _rank_by_delta(stats_dict, min_pr):
    """Return list of (id, stats) sorted by delta desc, filtered by min_pr (with fallback)."""
    eligible = [(k, v) for k, v in stats_dict.items() if v["pr"] >= min_pr]
    if not eligible:
        eligible = list(stats_dict.items())
    eligible.sort(key=lambda x: -x[1]["delta"])
    return eligible


def _format_item(iid, cell):
    out = {
        "id": iid,
        "wr": round(cell["wr"], 2),
        "pr": round(cell["pr"], 2),
        "n": cell["n"],
        "delta": round(cell["delta"], 2),
    }
    if "avg_time" in cell and cell["avg_time"]:
        out["avg_time"] = round(cell["avg_time"], 1)
    return out


def pick_optimal_build(slot_stats, start_sets, alts_per_slot=3):
    """
    Build a structured recommendation:
      - start: best start set (2-3 items) by delta
      - boots: best boots
      - core: items 1..5, each with primary (globally deduped) + up to N alts
    """
    # Start set: rank by delta with MIN_ITEM_PR floor
    start = None
    start_alts = []
    ranked_ss = _rank_by_delta(start_sets, MIN_ITEM_PR)
    if ranked_ss:
        def _format_set(sid, cell):
            return {
                "id": sid,
                "items": cell["items"],
                "wr": round(cell["wr"], 2),
                "pr": round(cell["pr"], 2),
                "n": cell["n"],
                "delta": round(cell["delta"], 2),
            }
        start = _format_set(*ranked_ss[0])
        start_alts = [_format_set(sid, cell) for sid, cell in ranked_ss[1:3]]

    # Boots: rank, take best
    boots = None
    boots_alts = []
    ranked_boots = _rank_by_delta(slot_stats.get("boots", {}), MIN_ITEM_PR)
    if ranked_boots:
        boots = _format_item(*ranked_boots[0])
        boots_alts = [_format_item(i, c) for i, c in ranked_boots[1:1 + alts_per_slot]]

    # Core items 1..5 with global dedup on the primary pick
    chosen = set()
    if boots is not None:
        chosen.add(boots["id"])
    core = []
    for slot_key in ITEM_SLOTS:
        ranked = _rank_by_delta(slot_stats.get(slot_key, {}), MIN_ITEM_PR)
        # Primary: highest-delta item not already picked
        primary = None
        for iid, cell in ranked:
            if iid in chosen:
                continue
            primary = _format_item(iid, cell)
            chosen.add(iid)
            break
        # Alternatives: top-N by delta in this slot, NOT deduped
        alts = [_format_item(iid, cell) for iid, cell in ranked[:alts_per_slot + 1]
                if primary is None or iid != primary["id"]][:alts_per_slot]
        core.append({"slot": slot_key, "primary": primary, "alternatives": alts})

    return {
        "start": start,
        "start_alternatives": start_alts,
        "boots": boots,
        "boots_alternatives": boots_alts,
        "core": core,
    }


RUNE_TREES = {
    1: {"name": "Precision", "rows": [
        [8005, 8008, 8021, 8010],
        [9101, 9111, 8009],
        [9104, 9105, 9103],
        [8014, 8017, 8299],
    ]},
    2: {"name": "Sorcery", "rows": [
        [8214, 8229, 8230],
        [8224, 8226, 8275],
        [8210, 8234, 8233],
        [8237, 8232, 8236],
    ]},
    3: {"name": "Resolve", "rows": [
        [8437, 8439, 8465],
        [8446, 8463, 8401],
        [8429, 8444, 8473],
        [8451, 8453, 8242],
    ]},
    4: {"name": "Inspiration", "rows": [
        [8351, 8360, 8369],
        [8306, 8304, 8313],
        [8321, 8316, 8345],
        [8347, 8352, 8410],
    ]},
    5: {"name": "Domination", "rows": [
        [8112, 8128, 9923],
        [8126, 8139, 8143],
        [8137, 8140, 8141],
        [8135, 8105, 8106],
    ]},
}

RUNE_SHARDS = {
    "row1": [5008, 5005, 5007],
    # Row 2 (flex slot) shards are tracked separately from row 1 with an
    # 'f' suffix on the key (e.g. 5008f vs 5008). We look up stats by the
    # suffixed key but display/icon use the bare integer ID.
    "row2": ["5008f", "5010f", "5001f"],
    "row3": [5011, 5013, 5001],
}


def _shard_display_id(rid):
    """Strip the 'f' flex-slot suffix on shard keys (e.g. '5010f' -> 5010)."""
    if isinstance(rid, str) and rid.endswith("f") and rid[:-1].isdigit():
        return int(rid[:-1])
    return rid


def _rune_summary(rid, stats):
    """Return {id, delta, intrinsic_delta, matchup_shift, wr, pr, n, eligible} for a rune id."""
    info = stats.get(str(rid)) if rid is not None else None
    display = _shard_display_id(rid)
    if info is None:
        return {"id": display, "delta": 0.0, "intrinsic_delta": 0.0,
                "matchup_shift": 0.0, "wr": 0.0, "pr": 0.0, "n": 0,
                "eligible": False}
    return {
        "id": display,
        "delta": round(info["delta"], 2),
        "intrinsic_delta": round(info.get("intrinsic_delta", 0.0), 2),
        "matchup_shift": round(info.get("matchup_shift", 0.0), 2),
        "wr": round(info["wr"], 2),
        "pr": round(info["pr"], 2),
        "n": info["n"],
        "eligible": info.get("eligible", True),
    }


def _rune_alternatives(rune_ids, chosen_id, stats, top_n):
    """Top N alternatives from a set of rune IDs, excluding chosen_id.
    Eligible (passes PR threshold) alts come first, then ineligible ones, each
    group sorted by delta. This way the frontend can show meta alternatives
    before niche curiosities, and dim/flag the latter as 'filtered'.
    """
    alts = [_rune_summary(rid, stats) for rid in rune_ids if rid != chosen_id]
    alts.sort(key=lambda x: (not x["eligible"], -x["delta"]))
    return alts[:top_n]


def build_optimal_rune_page(rune_info):
    """
    Holistic rune-page search: evaluate (primary_tree × keystone × secondary_tree)
    jointly so a slightly-worse keystone in a tree with much-better minors can win.

    Each component's contribution is scaled by an attribution factor that reflects
    "how much of this signal's data actually came from users of THIS keystone":
    - Primary minor of tree T: scale by tree_share(Y, T) = pr(Y) / Σ pr(K in T).
      Otherwise a niche keystone in a popular tree would free-ride on the modal
      keystone's minor stats (e.g. FleetFW Sylas inheriting Conq Sylas's PoM δ).
    - Secondary minor of tree S: scale by pr(Y)/100 — Y's overall share among
      all keystone choices (uniform-secondary-choice approximation).
    - Shards: scale by pr(Y)/100 too — same reasoning as secondary minors.
    """
    pri = rune_info.get("pri", {})
    sec = rune_info.get("sec", {})

    def best_in_row(rune_ids, stats):
        best_rid, best_delta, best_n = None, -1e9, 0
        for rid in rune_ids:
            info = stats.get(str(rid))
            if info is None or not info.get("eligible", True):
                continue
            if info["delta"] > best_delta:
                best_delta = info["delta"]
                best_rid = rid
                best_n = info["n"]
        if best_rid is None:
            return None, 0.0, 0
        return best_rid, best_delta, best_n

    # Per-tree total keystone PR (denominator for tree_share)
    tree_total_pr = {}
    for tid, tree in RUNE_TREES.items():
        total = 0.0
        for ks in tree["rows"][0]:
            info = pri.get(str(ks))
            if info:
                total += info.get("pr", 0)
        tree_total_pr[tid] = total

    # Per-tree primary minor picks (best in each row, independent of keystone)
    tree_minor = {}
    for tree_id, tree in RUNE_TREES.items():
        runes = []
        total = 0.0
        for row in tree["rows"][1:]:
            rid, d, _ = best_in_row(row, pri)
            runes.append(rid)
            total += d if rid is not None else 0.0
        tree_minor[tree_id] = {"runes": runes, "delta": total}

    # Per-tree best 2 secondary runes (from different rows)
    tree_secondary = {}
    for tree_id, tree in RUNE_TREES.items():
        row_bests = []
        for idx, row in enumerate(tree["rows"][1:], start=1):
            rid, d, _ = best_in_row(row, sec)
            if rid is not None:
                row_bests.append((idx, rid, d))
        row_bests.sort(key=lambda x: -x[2])
        if len(row_bests) >= 2:
            tree_secondary[tree_id] = {
                "runes": [row_bests[0][1], row_bests[1][1]],
                "rows":  [row_bests[0][0], row_bests[1][0]],
                "delta": row_bests[0][2] + row_bests[1][2],
            }

    # Shards: best in each row, summed (scaled per-keystone via attribution below)
    shard_total = 0.0
    for row_ids in (RUNE_SHARDS["row1"], RUNE_SHARDS["row2"], RUNE_SHARDS["row3"]):
        _rid, d, _ = best_in_row(row_ids, pri)
        shard_total += d

    # Enumerate (primary_tree, keystone, secondary_tree).
    best = None
    fallback = None
    for p_tree_id, p_tree in RUNE_TREES.items():
        minor = tree_minor[p_tree_id]
        tree_total = tree_total_pr.get(p_tree_id, 0)
        for ks in p_tree["rows"][0]:
            ks_info = pri.get(str(ks))
            if ks_info is None or not ks_info.get("eligible", True):
                continue
            ks_delta = ks_info["delta"]
            ks_n = ks_info["n"]
            ks_pr = ks_info.get("pr", 0.0)
            # Attribution factors
            ts_primary = (ks_pr / tree_total) if tree_total > 0 else 0.0
            attr_other = ks_pr / 100.0
            for s_tree_id in RUNE_TREES:
                if s_tree_id == p_tree_id:
                    continue
                sec_info = tree_secondary.get(s_tree_id)
                if sec_info is None:
                    continue
                page_score = (
                    ks_delta
                    + minor["delta"] * ts_primary
                    + sec_info["delta"] * attr_other
                    + shard_total * attr_other
                )
                candidate = (
                    page_score, ks_n, p_tree_id, ks, ks_delta,
                    minor["runes"], s_tree_id, sec_info["runes"], sec_info["rows"],
                )
                if best is None or candidate[0] > best[0]:
                    best = candidate
                if fallback is None or candidate[1] > fallback[1]:
                    fallback = candidate

    chosen = best or fallback
    if chosen is None:
        return None

    (page_score, _ks_n, p_tree_id, ks, _ks_delta, minor_runes,
     s_tree_id, sec_runes, sec_rows) = chosen

    # --- Enrich primary (keystone + 3 minors) with alternatives ---
    all_keystones = [rid for t in RUNE_TREES.values() for rid in t["rows"][0]]
    primary_runes_out = [
        {**_rune_summary(ks, pri),
         "alternatives": _rune_alternatives(all_keystones, ks, pri, top_n=4)}
    ]
    for row_idx, row_ids in enumerate(RUNE_TREES[p_tree_id]["rows"][1:], start=1):
        chosen_rid = minor_runes[row_idx - 1]
        primary_runes_out.append({
            **_rune_summary(chosen_rid, pri),
            "alternatives": _rune_alternatives(row_ids, chosen_rid, pri, top_n=2),
        })

    # --- Enrich secondary (2 runes, from two distinct rows) with alternatives ---
    secondary_runes_out = []
    for pick_rid, pick_row in zip(sec_runes, sec_rows):
        row_ids = RUNE_TREES[s_tree_id]["rows"][pick_row]
        secondary_runes_out.append({
            **_rune_summary(pick_rid, sec),
            "alternatives": _rune_alternatives(row_ids, pick_rid, sec, top_n=2),
        })

    # --- Shards: best per row by delta, with 2 alternatives per row ---
    shards_out = []
    for row_ids in (RUNE_SHARDS["row1"], RUNE_SHARDS["row2"], RUNE_SHARDS["row3"]):
        rid, _, _ = best_in_row(row_ids, pri)
        shards_out.append({
            **_rune_summary(rid, pri),
            "alternatives": _rune_alternatives(row_ids, rid, pri, top_n=2),
        })

    return {
        "primary_tree": p_tree_id,
        "primary_tree_name": RUNE_TREES[p_tree_id]["name"],
        "primary_runes": primary_runes_out,
        "secondary_tree": s_tree_id,
        "secondary_tree_name": RUNE_TREES[s_tree_id]["name"],
        "secondary_runes": secondary_runes_out,
        "shards": shards_out,
        "page_delta": round(page_score, 2),
    }


@app.route("/build-calc")
def build_calc():
    """
    Fetch matchup-specific builds for a champion vs all enemies, combine to find optimal runes.
    Called on-demand when all enemies + our champion are confirmed.
    """
    my_champ = request.args.get("champion", "").strip()
    my_lane = request.args.get("lane", "support").strip()
    tier = request.args.get("tier", "emerald_plus")
    patch = "30"

    # Collect enemies with their predicted lanes
    enemies = []
    for i in range(1, 6):
        name = request.args.get(f"enemy_{i}", "").strip()
        lane = request.args.get(f"enemy_{i}_lane", "").strip()
        if name and lane:
            enemies.append((name, lane))

    if not my_champ or not enemies:
        return jsonify({"error": "Need champion and at least one enemy"}), 400

    # Fetch unconditioned + per-enemy builds in parallel
    with ThreadPoolExecutor(max_workers=6) as pool:
        uncond_future = pool.submit(fetch_unconditioned_build, my_champ, my_lane, tier, patch)
        matchup_futures = [
            pool.submit(fetch_vs_build, my_champ, my_lane, enemy_name, enemy_lane, tier, patch)
            for enemy_name, enemy_lane in enemies
        ]
        unconditioned = uncond_future.result()
        build_results = [f.result() for f in matchup_futures]

    # Renormalized enemy weights (from DEFAULT_WEIGHTS for picking lane)
    enemy_weights = build_enemy_weights(my_lane, enemies)

    # Rune page — split scoring into intrinsic + matchup_shift
    rune_info = combine_rune_stats(build_results, enemy_weights, unconditioned.get("data"),
                                   champion_name=my_champ, picking_lane=my_lane)
    optimal = build_optimal_rune_page(rune_info)

    # Summoner spell pair
    pair_stats = combine_summoner_pairs(build_results, enemy_weights)
    best_pair, top_pairs = pick_optimal_summoner_pair(pair_stats)
    optimal_summoners = None
    if best_pair:
        optimal_summoners = {
            "pair": best_pair["pair"],
            "delta": round(best_pair["delta"], 2),
            "wr": round(best_pair["wr"], 2),
            "pr": round(best_pair["pr"], 2),
            "n": best_pair["n"],
            "alternatives": top_pairs[1:],
        }

    # Item build
    slot_stats = combine_item_slots(build_results, enemy_weights)
    start_sets = combine_start_sets(build_results, enemy_weights)
    optimal_items = pick_optimal_build(slot_stats, start_sets)

    # Extract per-enemy summary (display-only)
    per_enemy = []
    for result, weight in zip(build_results, enemy_weights):
        data = result.get("data")
        entry = {
            "enemy": result["enemy"],
            "enemy_lane": result["enemy_lane"],
            "weight": round(weight, 3),
        }
        if data and data.get("summary"):
            win = data["summary"].get("win", {})
            entry.update({
                "n": data.get("n", 0),
                "runes": win.get("runes", {}),
                "items": win.get("items", {}),
                "spells": win.get("sums", {}),
                "skill": win.get("skillpriority", {}),
            })
        else:
            entry["n"] = 0
        per_enemy.append(entry)

    return jsonify({
        "champion": my_champ,
        "lane": my_lane,
        "optimal_runes": optimal,
        "optimal_summoners": optimal_summoners,
        "optimal_items": optimal_items,
        "per_enemy_builds": per_enemy,
        "enemy_weights": [round(w, 3) for w in enemy_weights],
    })


GITHUB_DATA_BASE = "https://raw.githubusercontent.com/viktor-weinkauf/lol-picker/main/data"
GITHUB_TIERS_INDEX = "https://api.github.com/repos/viktor-weinkauf/lol-picker/contents/data"
DATA_FILES = ["champions", "counters", "synergy", "overall", "lane_dist", "meta"]


def sync_data_from_github(timeout=10):
    """Download the latest data/ JSON files from GitHub. Keeps bundled copies on failure."""
    try:
        resp = http_requests.get(GITHUB_TIERS_INDEX, timeout=timeout)
        resp.raise_for_status()
        tiers = [item["name"] for item in resp.json()
                 if item["type"] == "dir" and item["name"] != "build_cache"]
    except Exception as e:
        print(f"[sync] could not reach GitHub ({e}); using bundled data")
        return

    os.makedirs(DATA_ROOT, exist_ok=True)
    for tier in tiers:
        tier_dir = os.path.join(DATA_ROOT, tier)
        os.makedirs(tier_dir, exist_ok=True)
        for key in DATA_FILES:
            url = f"{GITHUB_DATA_BASE}/{tier}/{key}.json"
            try:
                r = http_requests.get(url, timeout=timeout)
                r.raise_for_status()
                with open(os.path.join(tier_dir, f"{key}.json"), "w", encoding="utf-8") as f:
                    f.write(r.text)
            except Exception as e:
                print(f"[sync] {tier}/{key}.json failed ({e}); keeping local copy")
    _data_cache.clear()
    print(f"[sync] refreshed {len(tiers)} tier(s) from GitHub")


def seed_bundled_data():
    """First run of the exe: copy bundled data/ next to the exe so it's writable."""
    if not getattr(sys, "frozen", False):
        return
    bundled = os.path.join(BUNDLE_DIR, "data")
    if os.path.isdir(bundled) and not os.path.isdir(DATA_ROOT):
        shutil.copytree(bundled, DATA_ROOT)
        print(f"[seed] copied bundled data to {DATA_ROOT}")


def redirect_output_to_logfile():
    """When running as a windowed exe, send stdout/stderr to a log file
    next to the exe so errors aren't silently lost."""
    if not getattr(sys, "frozen", False):
        return
    log_path = os.path.join(BASE_DIR, "lolpicker.log")
    log = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = log
    sys.stderr = log


def open_browser_delayed(url="http://127.0.0.1:5000", delay=1.2):
    """Open the default browser once the Flask server has had time to bind."""
    import threading
    import webbrowser
    threading.Timer(delay, lambda: webbrowser.open(url)).start()


def _make_tray_image():
    """Load the tray icon from bundled assets, with a minimal fallback."""
    from PIL import Image, ImageDraw
    path = os.path.join(BUNDLE_DIR, "assets", "tray.png")
    if os.path.exists(path):
        return Image.open(path)
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    ImageDraw.Draw(img).rectangle([8, 8, 56, 56], outline="white", width=3)
    return img


def run_tray_icon(url="http://127.0.0.1:5000"):
    """Blocking: show a system tray icon with Open/Quit actions until Quit is clicked."""
    import webbrowser
    import pystray

    def on_open(icon, item):
        webbrowser.open(url)

    def on_quit(icon, item):
        icon.stop()
        os._exit(0)

    icon = pystray.Icon(
        "LoLPicker",
        _make_tray_image(),
        "LoL Picker",
        menu=pystray.Menu(
            pystray.MenuItem("Open picker", on_open, default=True),
            pystray.MenuItem("Quit", on_quit),
        ),
    )
    icon.run()


if __name__ == "__main__":
    redirect_output_to_logfile()
    seed_bundled_data()
    sync_data_from_github()
    tiers = available_tiers()
    if tiers:
        d = load_data(tiers[0])
        if d:
            print(f"Data: {tiers[0]} | {d['meta']['patch']} days | {d['meta'].get('viable_pairs', '?')} pairs")
    else:
        print("No data. Run: python scrape_data.py")

    frozen = getattr(sys, "frozen", False)
    if frozen:
        import threading
        threading.Thread(
            target=lambda: app.run(debug=False, port=5000, use_reloader=False),
            daemon=True,
        ).start()
        open_browser_delayed()
        run_tray_icon()
    else:
        app.run(debug=True, port=5000, use_reloader=False)
