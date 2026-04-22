"""
Download champion matchup and synergy data from lolalytics for all viable role-champion pairs.

Phase 1: Fetch lane distribution for every champion (172 calls, ~4 min)
Phase 2: For each viable pair (>0.5% pick rate in that role), fetch cross-lane counter data
         and synergy data (5 vslane + 1 synergy = 6 calls per pair, ~20-30 min total)

Usage:
    python scrape_data.py                           # default: emerald_plus, 30 days
    python scrape_data.py --tier diamond_plus       # specific tier
    python scrape_data.py --tier gold --patch 14    # gold, last 14 days
"""

import argparse
import json
import os
import time
import requests
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
API_BASE = "https://a1.lolalytics.com/mega/"
DDRAGON_VERSIONS = "https://ddragon.leagueoflegends.com/api/versions.json"
DDRAGON_CHAMPS = "https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"
DELAY = 1.5
MIN_LANE_PCT = 0.5  # minimum lane % to scrape a champion-role pair

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Referer": "https://lolalytics.com/",
    "Origin": "https://lolalytics.com",
}

VALID_TIERS = [
    "all", "challenger", "grandmaster_plus", "grandmaster",
    "master_plus", "master", "diamond_plus", "diamond",
    "emerald_plus", "emerald", "platinum_plus", "platinum",
    "gold_plus", "gold", "silver", "bronze", "iron",
]

ROLES = ["top", "jungle", "middle", "bottom", "support"]


def get_champion_mapping():
    print("Fetching champion data from Data Dragon...")
    versions = requests.get(DDRAGON_VERSIONS, timeout=10).json()
    version = versions[0]
    print(f"  Latest version: {version}")

    champ_data = requests.get(DDRAGON_CHAMPS.format(version=version), timeout=10).json()

    id_to_name = {}
    name_to_id = {}
    id_to_image = {}
    for champ_id, info in champ_data["data"].items():
        cid = int(info["key"])
        id_to_name[cid] = info["name"]
        name_to_id[info["name"]] = cid
        id_to_image[cid] = champ_id  # DDragon key e.g. "KaiSa"

    patch = version.rsplit(".", 1)[0]
    return id_to_name, name_to_id, id_to_image, version, patch


def api_name_from_image(image_key):
    return image_key.lower()


def fetch_lane_distribution(aname, tier, patch):
    """Fetch lane distribution for a champion. Returns {role: pct} and overall stats."""
    # Query with the champion's most likely lane - but lanes data is the same regardless
    # Use 'top' as default; if it 404s we'll try others
    for try_lane in ROLES:
        params = {"ep": "counter", "v": 1, "c": aname, "lane": try_lane, "tier": tier, "patch": patch}
        try:
            r = requests.get(API_BASE, params=params, headers=HEADERS, timeout=15)
            data = r.json()
            if "stats" in data and "lanes" in data.get("stats", {}):
                stats = data["stats"]
                return stats.get("lanes", {}), {
                    "wr": float(stats.get("wr", 0)),
                    "pr": float(stats.get("pr", 0)),
                    "br": float(stats.get("br", 0)),
                    "avgWr": float(stats.get("avgWr", 50)),
                }
        except Exception:
            pass
    return None, None


def fetch_counter_vslane(aname, lane, vslane, tier, patch):
    params = {"ep": "counter", "v": 1, "c": aname, "lane": lane, "vslane": vslane, "tier": tier, "patch": patch}
    r = requests.get(API_BASE, params=params, headers=HEADERS, timeout=15)
    data = r.json()
    if "counters" not in data:
        return {}, None
    matchups = {
        c["cid"]: {"vsWr": c["vsWr"], "d1": c["d1"], "d2": c["d2"], "n": c["n"], "role": c.get("defaultLane", "")}
        for c in data["counters"]
    }
    stats = data.get("stats", {})
    overall = {
        "wr": float(stats.get("wr", 0)),
        "pr": float(stats.get("pr", 0)),
        "br": float(stats.get("br", 0)),
        "avgWr": float(stats.get("avgWr", 50)),
    }
    return matchups, overall


def fetch_synergy(aname, lane, tier, patch):
    params = {"ep": "build-team", "v": 1, "c": aname, "lane": lane, "tier": tier, "patch": patch}
    r = requests.get(API_BASE, params=params, headers=HEADERS, timeout=15)
    data = r.json()
    if "team" not in data:
        return None
    result = {}
    for role, entries in data["team"].items():
        role_data = {}
        for entry in entries:
            ally_cid, wr, d1, d2, pr, n = entry
            if n >= 20:
                role_data[ally_cid] = {"wr": wr, "d1": d1, "d2": d2, "n": n}
        result[role] = role_data
    return result


def scrape_all(tier, patch):
    data_dir = os.path.join(BASE_DIR, "data", tier)
    os.makedirs(data_dir, exist_ok=True)

    id_to_name, name_to_id, id_to_image, ddragon_version, current_patch = get_champion_mapping()

    # =========================================
    # Phase 1: Lane distributions for ALL champs
    # =========================================
    print(f"\n--- Phase 1: Lane distributions for {len(id_to_name)} champions ---")
    lane_dist = {}  # cid -> {role: pct}
    all_champs = sorted(id_to_name.items(), key=lambda x: x[1])

    for idx, (cid, name) in enumerate(all_champs):
        aname = api_name_from_image(id_to_image.get(cid, name))
        lanes, _ = fetch_lane_distribution(aname, tier, patch)
        if lanes:
            lane_dist[cid] = lanes
            roles_str = ", ".join(f"{r}={lanes.get(r,0)}%" for r in ROLES if lanes.get(r, 0) >= MIN_LANE_PCT)
            print(f"[{idx+1}/{len(all_champs)}] {name}: {roles_str}")
        else:
            print(f"[{idx+1}/{len(all_champs)}] {name}: FAILED")
        time.sleep(DELAY)

    # Determine viable pairs
    viable_pairs = []  # (role, cid, name)
    for cid, lanes in lane_dist.items():
        for role in ROLES:
            if lanes.get(role, 0) >= MIN_LANE_PCT:
                viable_pairs.append((role, cid, id_to_name[cid]))

    print(f"\nViable role-champion pairs (>{MIN_LANE_PCT}%): {len(viable_pairs)}")
    for role in ROLES:
        count = sum(1 for r, _, _ in viable_pairs if r == role)
        print(f"  {role}: {count}")

    # =========================================
    # Phase 2: Counter + synergy for viable pairs
    # =========================================
    print(f"\n--- Phase 2: Scraping {len(viable_pairs)} viable pairs ---")

    counters = {role: {} for role in ROLES}
    synergy = {role: {} for role in ROLES}
    overall = {role: {} for role in ROLES}

    for idx, (role, cid, name) in enumerate(viable_pairs):
        aname = api_name_from_image(id_to_image.get(cid, name))
        print(f"[{idx+1}/{len(viable_pairs)}] {role}/{name}...", end=" ", flush=True)

        # Counter data per enemy lane
        champ_counters = {}
        total_c = 0
        for vslane in ROLES:
            try:
                vs_data, stats = fetch_counter_vslane(aname, role, vslane, tier, patch)
                if vs_data:
                    champ_counters[vslane] = vs_data
                    total_c += len(vs_data)
                if stats and vslane == role:
                    # Use same-lane stats for overall (most relevant)
                    overall[role][cid] = stats
            except Exception:
                pass
            time.sleep(DELAY)

        counters[role][cid] = champ_counters
        print(f"c={total_c}", end=" ", flush=True)

        # Synergy
        try:
            syn = fetch_synergy(aname, role, tier, patch)
            if syn:
                synergy[role][cid] = syn
                total_syn = sum(len(v) for v in syn.values())
                print(f"s={total_syn}")
            else:
                print("s=FAIL")
        except Exception:
            print("s=ERR")
        time.sleep(DELAY)

    # =========================================
    # Save
    # =========================================
    def cid_to_name_recursive(d):
        if not isinstance(d, dict):
            return d
        result = {}
        for k, v in d.items():
            key = id_to_name[k] if isinstance(k, int) and k in id_to_name else k
            result[key] = cid_to_name_recursive(v) if isinstance(v, dict) else v
        return result

    counters_named = cid_to_name_recursive(counters)
    synergy_named = cid_to_name_recursive(synergy)
    overall_named = cid_to_name_recursive(overall)
    lane_dist_named = {id_to_name[cid]: lanes for cid, lanes in lane_dist.items() if cid in id_to_name}

    # Build pools: per-role champion lists (all champs with >MIN_LANE_PCT in that role)
    pools = {role: {} for role in ROLES}
    for role, cid, name in viable_pairs:
        pools[role][str(cid)] = name

    champions_data = {
        "id_to_name": {str(k): v for k, v in id_to_name.items()},
        "name_to_id": name_to_id,
        "id_to_image": {str(k): v for k, v in id_to_image.items()},
        "pools": {role: {str(k): v for k, v in champs.items()} for role, champs in pools.items()},
    }

    meta = {
        "patch": patch,
        "current_patch": current_patch,
        "ddragon_version": ddragon_version,
        "tier": tier,
        "scraped_at": datetime.now().isoformat(),
        "viable_pairs": len(viable_pairs),
        "min_lane_pct": MIN_LANE_PCT,
    }

    files = {
        "champions.json": champions_data,
        "counters.json": counters_named,
        "synergy.json": synergy_named,
        "overall.json": overall_named,
        "lane_dist.json": lane_dist_named,
        "meta.json": meta,
    }

    for filename, content in files.items():
        path = os.path.join(data_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2, ensure_ascii=False)
        size_kb = os.path.getsize(path) / 1024
        print(f"Saved {filename} ({size_kb:.0f} KB)")

    print(f"\nDone! Patch: {patch} | Tier: {tier} | Pairs: {len(viable_pairs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download LoL matchup data from lolalytics")
    parser.add_argument("--tier", default="emerald_plus", choices=VALID_TIERS)
    parser.add_argument("--patch", default="30")
    args = parser.parse_args()
    scrape_all(args.tier, args.patch)
