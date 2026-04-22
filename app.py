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

    # For 2-5 champions: find the optimal assignment via brute force
    # (at most 5! = 120 permutations, very fast)
    best_assignment = None
    best_score = -1

    champs = list(enemy_champs)
    n = len(champs)
    available_roles = ROLES[:n] if n <= 5 else ROLES

    for perm in permutations(ROLES, n):
        score = 1.0
        for champ, role in zip(champs, perm):
            p = champ_lanes[champ].get(role, 0)
            score *= max(p, 0.001)  # avoid zero
        if score > best_score:
            best_score = score
            best_assignment = dict(zip(champs, perm))

    # Also compute per-champion role probabilities (marginals)
    # by summing over all valid assignments weighted by their probability
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
    Applied independently per enemy role (each role is a separate decomposition
    of the overall WR), then averaged.

    Per role: adjusted_wr_role = (wr - norm_pr * wr_vs) / (1 - norm_pr)
    where norm_pr = champion's PR / sum(all PRs in that role) = true encounter rate
    """
    champ_overall = data.get("overall", {}).get(picking_role, {}).get(champ_name, {})
    base_wr = champ_overall.get("wr", 50)
    if not bans:
        return base_wr

    champ_counters = data["counters"].get(picking_role, {}).get(champ_name, {})
    all_overall = data.get("overall", {})
    ban_set = set(bans)

    # Apply ban adjustment independently per enemy role
    total_adjustment = 0.0
    n_adjustments = 0

    for role in ROLES:
        role_overall = all_overall.get(role, {})
        # Total PR in this role (sums to ~200%, both teams)
        total_role_pr = sum(s.get("pr", 0) for s in role_overall.values())
        if total_role_pr <= 0:
            continue

        # Sum normalized PR of banned champs in this role
        removed_pr = 0.0
        removed_wr_contribution = 0.0

        for banned in ban_set:
            banned_stats = role_overall.get(banned, {})
            raw_pr = banned_stats.get("pr", 0)
            if raw_pr <= 0:
                continue

            norm_pr = raw_pr / total_role_pr  # true encounter rate
            matchup = champ_counters.get(role, {}).get(banned, {})
            if matchup.get("n", 0) >= MIN_GAMES:
                wr_vs = matchup["vsWr"]
            else:
                wr_vs = base_wr  # no data = no effect

            removed_pr += norm_pr
            removed_wr_contribution += norm_pr * wr_vs

        if removed_pr > 0 and removed_pr < 1.0:
            # This role's contribution to WR adjustment
            role_adjusted = (base_wr - removed_wr_contribution) / (1.0 - removed_pr)
            total_adjustment += role_adjusted - base_wr
            n_adjustments += 1

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
                n = fwd_n + rev_n
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
    enemy_weight_map = {"top": "enemy_top", "jungle": "enemy_jungle", "middle": "enemy_mid",
                        "bottom": "enemy_adc", "support": "enemy_support"}

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
            weight_key = enemy_weight_map[enemy_role]
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
                    wr = fwd_wr
                    n = fwd_n + rev_n
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
    avg_wr = next(iter(overall.values()), {}).get("avgWr", 50) if overall else 50

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


def combine_rune_stats(build_results):
    """
    Combine per-rune stats across multiple enemy matchups.
    For each rune slot, find the rune with the highest average WR
    weighted by game count across all matchups.
    """
    # Collect per-rune stats: rune_id -> {total_wr_weighted, total_n, appearances}
    # Separate by slot (primary vs secondary usage)
    combined = {"pri": {}, "sec": {}}

    for result in build_results:
        data = result.get("data")
        if not data:
            continue
        rune_stats = data.get("runes", {}).get("stats", {})
        for rune_id, entries in rune_stats.items():
            for i, entry in enumerate(entries):
                slot = "pri" if i == 0 else "sec"
                pr, wr, n = entry
                if n < 10:
                    continue
                if rune_id not in combined[slot]:
                    combined[slot][rune_id] = {"total_wr_n": 0, "total_n": 0}
                combined[slot][rune_id]["total_wr_n"] += wr * n
                combined[slot][rune_id]["total_n"] += n

    # Calculate weighted average WR per rune per slot
    rune_wrs = {"pri": {}, "sec": {}}
    for slot in ["pri", "sec"]:
        for rune_id, stats in combined[slot].items():
            if stats["total_n"] > 0:
                rune_wrs[slot][rune_id] = {
                    "wr": round(stats["total_wr_n"] / stats["total_n"], 2),
                    "n": stats["total_n"],
                }

    return rune_wrs


def build_optimal_rune_page(rune_wrs):
    """
    Given per-rune weighted WRs, construct the optimal rune page.
    Uses knowledge of rune tree structure to pick the best keystone,
    then best runes per row, then best secondary tree + runes, then shards.
    """
    # Rune tree structure: tree_id -> [row0_keystones, row1, row2, row3]
    TREES = {
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

    SHARDS = {
        "row1": [5008, 5005, 5007],
        "row2": [5008, 5010, 5001],  # 5008f, 5010f, 5001f in API
        "row3": [5011, 5013, 5001],
    }

    pri_stats = rune_wrs.get("pri", {})
    sec_stats = rune_wrs.get("sec", {})

    def best_rune(rune_ids, stats, min_n=1000):
        """Pick the rune with highest WR from a list, requiring sufficient games."""
        best = None
        best_wr = -1
        for rid in rune_ids:
            info = stats.get(str(rid))
            if info and info["wr"] > best_wr and info["n"] >= min_n:
                best_wr = info["wr"]
                best = rid
        # Fallback: if nothing meets min_n, use the one with most games
        if not best:
            most_games = -1
            for rid in rune_ids:
                info = stats.get(str(rid))
                if info and info["n"] > most_games:
                    most_games = info["n"]
                    best = rid
                    best_wr = info["wr"]
        return best, best_wr

    # 1. Find best primary tree (by best keystone WR across ALL trees)
    all_keystones = []
    for tree_id, tree in TREES.items():
        for rid in tree["rows"][0]:
            info = pri_stats.get(str(rid))
            if info:
                all_keystones.append((tree_id, rid, info["wr"], info["n"]))

    # Sort by WR but require min 1000 games; fallback to most games
    reliable = [k for k in all_keystones if k[3] >= 1000]
    if reliable:
        reliable.sort(key=lambda x: -x[2])
        best_tree, best_ks, best_ks_wr, _ = reliable[0]
    elif all_keystones:
        all_keystones.sort(key=lambda x: -x[3])  # most games
        best_tree, best_ks, best_ks_wr, _ = all_keystones[0]
    else:
        best_tree = None
        best_ks = None
        best_ks_wr = -1

    if not best_tree:
        return None

    # 2. Pick best rune per row in primary tree
    pri_tree = TREES[best_tree]
    pri_runes = [best_ks]
    for row in pri_tree["rows"][1:]:
        rune, _ = best_rune(row, pri_stats)
        pri_runes.append(rune)

    # 3. Find best secondary tree (highest combined WR of 2 runes from different rows)
    best_sec_tree = None
    best_sec_runes = []
    best_sec_score = -1

    for tree_id, tree in TREES.items():
        if tree_id == best_tree:
            continue
        # Pick best rune from each row (rows 1-3, skip keystones)
        row_bests = []
        for row in tree["rows"][1:]:
            rune, wr = best_rune(row, sec_stats)
            if rune:
                row_bests.append((rune, wr))
        # Pick top 2 by WR
        row_bests.sort(key=lambda x: -x[1])
        if len(row_bests) >= 2:
            score = row_bests[0][1] + row_bests[1][1]
            if score > best_sec_score:
                best_sec_score = score
                best_sec_tree = tree_id
                best_sec_runes = [row_bests[0][0], row_bests[1][0]]

    # 4. Stat shards
    shard_stats = pri_stats  # shards use primary slot stats
    mod = []
    for shard_row_ids in [SHARDS["row1"], SHARDS["row2"], SHARDS["row3"]]:
        s, _ = best_rune(shard_row_ids, shard_stats)
        mod.append(s)

    return {
        "primary_tree": best_tree,
        "primary_tree_name": TREES[best_tree]["name"],
        "primary_runes": pri_runes,
        "secondary_tree": best_sec_tree,
        "secondary_tree_name": TREES[best_sec_tree]["name"] if best_sec_tree else None,
        "secondary_runes": best_sec_runes,
        "shards": mod,
        "keystone_wr": best_ks_wr,
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

    # Fetch builds vs all enemies in parallel (no delay needed for 5 requests)
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [
            pool.submit(fetch_vs_build, my_champ, my_lane, enemy_name, enemy_lane, tier, patch)
            for enemy_name, enemy_lane in enemies
        ]
        build_results = [f.result() for f in futures]

    # Combine per-rune stats and find optimal page
    rune_wrs = combine_rune_stats(build_results)
    optimal = build_optimal_rune_page(rune_wrs)

    # Also extract items and spells from each matchup's summary
    per_enemy = []
    for result in build_results:
        data = result.get("data")
        if data and data.get("summary"):
            win = data["summary"].get("win", {})
            per_enemy.append({
                "enemy": result["enemy"],
                "enemy_lane": result["enemy_lane"],
                "n": data.get("n", 0),
                "runes": win.get("runes", {}),
                "items": win.get("items", {}),
                "spells": win.get("sums", {}),
                "skill": win.get("skillpriority", {}),
            })
        else:
            per_enemy.append({"enemy": result["enemy"], "enemy_lane": result["enemy_lane"], "n": 0})

    return jsonify({
        "champion": my_champ,
        "lane": my_lane,
        "optimal_runes": optimal,
        "per_enemy_builds": per_enemy,
        "rune_wrs": {slot: {k: v for k, v in stats.items() if v.get("n", 0) >= 100}
                     for slot, stats in rune_wrs.items()},
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


if __name__ == "__main__":
    seed_bundled_data()
    sync_data_from_github()
    tiers = available_tiers()
    if tiers:
        d = load_data(tiers[0])
        if d:
            print(f"Data: {tiers[0]} | {d['meta']['patch']} days | {d['meta'].get('viable_pairs', '?')} pairs")
    else:
        print("No data. Run: python scrape_data.py")
    debug_mode = not getattr(sys, "frozen", False)
    app.run(debug=debug_mode, port=5000, use_reloader=False)
