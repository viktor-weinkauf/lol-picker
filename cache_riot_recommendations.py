"""
Cache Riot's recommended rune pages for every (champion, lane) pair.

Reads the patch version + recommended pages from the local League client (LCU)
and saves them to data/riot_recommended/{patch}/runes.json. Run while the
client is open. Re-run each patch — we keep prior patches so we can diff
recommendation changes against observed WR shifts.
"""
import sys
import os
import json
import datetime
from concurrent.futures import ThreadPoolExecutor

import requests
import urllib3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import app  # for get_lcu_connection + champion data

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# Our lane name -> Riot's LCU position string
LANE_TO_RIOT = {
    "top":     "TOP",
    "jungle":  "JUNGLE",
    "middle":  "MIDDLE",
    "bottom":  "BOTTOM",
    "support": "UTILITY",
}


def patch_version(base, auth):
    r = requests.get(f"{base}/system/v1/builds", auth=auth, verify=False, timeout=5)
    full = r.json().get("version", "unknown")
    # "16.8.768.3546" -> "16.8"
    return ".".join(full.split(".")[:2])


def fetch_pages(base, auth, champ_id, riot_lane):
    url = (f"{base}/lol-perks/v1/recommended-pages/champion/"
           f"{champ_id}/position/{riot_lane}/map/11")
    try:
        r = requests.get(url, auth=auth, verify=False, timeout=5)
        if r.status_code != 200:
            return []
        out = []
        for page in r.json():
            keystone = page.get("keystone") or {}
            perks = page.get("perks") or []
            out.append({
                "keystone_id": keystone.get("id"),
                "keystone_name": keystone.get("name"),
                "primary_style_id": page.get("primaryPerkStyleId"),
                "secondary_style_id": page.get("secondaryPerkStyleId"),
                # The 9-element list: [keystone, 3 primary minors, 2 secondary
                # minors, 3 shards]. Stored as Riot returns them; the runtime
                # lookup applies the 'f' suffix for the row-2 shard (perks[7]).
                "perk_ids": [pk.get("id") for pk in perks if pk.get("id")],
                "primary_attribute": page.get("primaryRecommendationAttribute"),
                "secondary_attribute": page.get("secondaryRecommendationAttribute"),
            })
        return out
    except Exception:
        return []


def main():
    port, token = app.get_lcu_connection()
    if not port:
        print("League client not running. Open the client and try again.")
        sys.exit(1)
    auth = ("riot", token)
    base = f"https://127.0.0.1:{port}"

    patch = patch_version(base, auth)
    print(f"Patch detected: {patch}")

    # Load champion pools from our scraped tier data
    tier = "emerald_plus"
    champ_path = os.path.join("data", tier, "champions.json")
    with open(champ_path, "r", encoding="utf-8") as f:
        champ_data = json.load(f)
    name_to_id = champ_data["name_to_id"]
    pools = champ_data["pools"]  # role -> {champ_id_str: name}

    # Build (champ_id, name, our_lane, riot_lane) work list (deduped)
    seen = set()
    work = []
    for our_lane, role_pool in pools.items():
        riot_lane = LANE_TO_RIOT.get(our_lane)
        if not riot_lane:
            continue
        for cid_str, name in role_pool.items():
            cid = int(cid_str)
            if (cid, our_lane) in seen:
                continue
            seen.add((cid, our_lane))
            work.append((cid, name, our_lane, riot_lane))

    print(f"Fetching {len(work)} (champion, lane) pairs...")

    # Output structure: pages[champion_name][our_lane] = [pages...]
    pages_by_name = {}

    def task(args):
        cid, name, our_lane, riot_lane = args
        return name, our_lane, fetch_pages(base, auth, cid, riot_lane)

    completed = 0
    with ThreadPoolExecutor(max_workers=10) as pool:
        for name, our_lane, pages in pool.map(task, work):
            if pages:
                pages_by_name.setdefault(name, {})[our_lane] = pages
            completed += 1
            if completed % 50 == 0:
                print(f"  {completed}/{len(work)}...")

    out = {
        "patch": patch,
        "fetched_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "pages": pages_by_name,
    }
    out_dir = os.path.join("data", "riot_recommended", patch)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "runes.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    n_champs = len(pages_by_name)
    n_lanes = sum(len(v) for v in pages_by_name.values())
    print(f"\nSaved {n_champs} champions × {n_lanes} total (champion, lane) pages to:")
    print(f"  {out_path}")


if __name__ == "__main__":
    main()
