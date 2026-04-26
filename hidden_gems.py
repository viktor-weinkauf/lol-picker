"""
Find rune picks that beat Riot's recommendations.

For every (champion, lane) in our scraped pool, run our scoring with no
specific enemies (intrinsic-only) and report any rune our optimal page
picks that does NOT appear in Riot's recommended pages for that combo.

These are "hidden gems" — runes the data prefers despite the +2pp Phreak
boost we give to all of Riot's recommended runes. Worth countercheck on
lolalytics to validate.
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import app


# Disable PR threshold so we see every signal the data carries
app.MIN_KEYSTONE_PR = 0.0
app.MIN_RUNE_PR = 0.0

TIER = "emerald_plus"
PATCH = "30"
# Filtering thresholds — only report findings backed by substantial sample.
MIN_DELTA_GAIN = 1.0  # gain over Riot's best pick (in WR pp)
MIN_PR = 5.0          # at least this much pickrate (filters off-meta lanes)
MIN_N = 1000          # at least this many sample games

NAMES = {
    "8005": "Press the Attack", "8008": "Lethal Tempo", "8021": "Fleet Footwork",
    "8010": "Conqueror", "8214": "Summon Aery", "8229": "Arcane Comet",
    "8230": "Phase Rush", "8437": "Grasp of the Undying", "8439": "Aftershock",
    "8465": "Guardian", "8351": "Glacial Augment", "8360": "Unsealed Spellbook",
    "8369": "First Strike", "8112": "Electrocute", "8128": "Dark Harvest",
    "9923": "Hail of Blades",
    "9111": "Triumph", "8009": "Presence of Mind", "9101": "Overheal",
    "9104": "Legend: Alacrity", "9105": "Legend: Tenacity", "9103": "Legend: Bloodline",
    "8014": "Coup de Grace", "8017": "Cut Down", "8299": "Last Stand",
    "8224": "Nullifying Orb", "8226": "Manaflow Band", "8275": "Nimbus Cloak",
    "8210": "Transcendence", "8234": "Celerity", "8233": "Absolute Focus",
    "8237": "Scorch", "8232": "Waterwalking", "8236": "Gathering Storm",
    "8446": "Demolish", "8463": "Font of Life", "8401": "Mirror Shell",
    "8429": "Conditioning", "8444": "Second Wind", "8473": "Bone Plating",
    "8451": "Overgrowth", "8453": "Revitalize", "8242": "Unflinching",
    "8351": "Glacial Augment", "8360": "Unsealed Spellbook", "8369": "First Strike",
    "8306": "Hextech Flashtraption", "8304": "Magical Footwear", "8313": "Triple Tonic",
    "8321": "Cash Back", "8316": "Minion Dematerializer", "8345": "Biscuit Delivery",
    "8347": "Cosmic Insight", "8352": "Time Warp Tonic", "8410": "Approach Velocity",
    "8126": "Cheap Shot", "8139": "Taste of Blood", "8143": "Sudden Impact",
    "8137": "Grisly Mementos", "8140": "Eyeball Collection", "8141": "Zombie Ward",
    "8135": "Treasure Hunter", "8105": "Relentless Hunter", "8106": "Ultimate Hunter",
    "5008": "Adaptive Force", "5005": "Attack Speed", "5007": "Ability Haste",
    "5008f": "Adaptive Force (flex)", "5010": "Movement Speed",
    "5010f": "Movement Speed (flex)", "5001": "Health Scaling",
    "5001f": "Health Scaling (flex)", "5011": "Health", "5013": "Tenacity",
}


def name_of(rid):
    return NAMES.get(str(rid), str(rid))


def best_riot_delta(rid_set, rune_info_slot):
    """Best delta among the set of rune IDs that ARE in Riot's recommendations."""
    best = None
    for rid in rid_set:
        info = rune_info_slot.get(str(rid))
        if info is None:
            continue
        if best is None or info["delta"] > best["delta"]:
            best = info
            best["_id"] = rid
    return best


def analyze_champion_lane(champ, lane):
    """Run optimization with no enemies and find divergences from Riot's recs."""
    uncond = app.fetch_unconditioned_build(champ, lane, TIER, PATCH)
    if not uncond.get("data"):
        return None
    rune_info = app.combine_rune_stats([], [], uncond.get("data"),
                                       champion_name=champ, picking_lane=lane)
    optimal = app.build_optimal_rune_page(rune_info)
    if not optimal:
        return None

    # Riot's recommended sets per slot+row
    recommended_slots = app.get_recommended_perk_slots(champ, lane)
    pages = (app._load_riot_recommended()
             .get("pages", {}).get(champ, {}).get(lane, []))

    # Build per-slot recommended ID sets so we can find the *best* Riot
    # rune per slot for delta-gain comparison.
    rec_keystones = set()
    rec_primary_minors = [set(), set(), set()]   # per row
    rec_secondary_minors = set()
    rec_shards = [set(), set(), set()]            # per row
    for page in pages:
        perks = page.get("perk_ids") or []
        for i, pid in enumerate(perks):
            if pid is None:
                continue
            if i == 0:
                rec_keystones.add(str(pid))
            elif 1 <= i <= 3:
                rec_primary_minors[i - 1].add(str(pid))
            elif 4 <= i <= 5:
                rec_secondary_minors.add(str(pid))
            elif i == 6:
                rec_shards[0].add(str(pid))
            elif i == 7:
                rec_shards[1].add(f"{pid}f")
            elif i == 8:
                rec_shards[2].add(str(pid))

    findings = []
    pri = rune_info.get("pri", {})
    sec = rune_info.get("sec", {})

    # Keystone divergence
    chosen_ks = optimal["primary_runes"][0]
    if str(chosen_ks["id"]) not in rec_keystones:
        # Find best Riot keystone for comparison
        best_riot = None
        for rid in rec_keystones:
            info = pri.get(rid)
            if info and (best_riot is None or info["delta"] > best_riot["delta"]):
                best_riot = {**info, "_id": rid}
        gain = chosen_ks["delta"] - (best_riot["delta"] if best_riot else 0)
        if (gain >= MIN_DELTA_GAIN
                and chosen_ks["pr"] >= MIN_PR
                and chosen_ks["n"] >= MIN_N):
            findings.append({
                "type": "keystone",
                "ours": chosen_ks["id"], "ours_name": name_of(chosen_ks["id"]),
                "ours_delta": chosen_ks["delta"],
                "ours_pr": chosen_ks["pr"], "ours_n": chosen_ks["n"],
                "riot": best_riot["_id"] if best_riot else None,
                "riot_name": name_of(best_riot["_id"]) if best_riot else "?",
                "riot_delta": best_riot["delta"] if best_riot else None,
                "gain": gain,
            })

    # Primary minors (3 rows of the chosen primary tree)
    p_tree = app.RUNE_TREES[optimal["primary_tree"]]
    for row_idx, minor in enumerate(optimal["primary_runes"][1:], start=1):
        chosen_id = str(minor["id"])
        # All rec primary minors merged across pages for this row position
        # (matching by row index of the chosen primary tree might not align
        # if Riot's recommended primary tree differs; instead just check the
        # full union of recommended primary minors in case the rune is in
        # any recommended position).
        # Simpler check: is it in ANY recommended primary minor row (1-3)?
        all_rec_primary_minors = (rec_primary_minors[0] | rec_primary_minors[1]
                                  | rec_primary_minors[2])
        if chosen_id not in all_rec_primary_minors:
            # Find best of the rec primary minors with data
            best_riot = None
            for rid in all_rec_primary_minors:
                info = pri.get(rid)
                if info and (best_riot is None or info["delta"] > best_riot["delta"]):
                    best_riot = {**info, "_id": rid}
            gain = minor["delta"] - (best_riot["delta"] if best_riot else 0)
            if (gain >= MIN_DELTA_GAIN
                    and minor["pr"] >= MIN_PR
                    and minor["n"] >= MIN_N):
                findings.append({
                    "type": f"primary_minor_row_{row_idx}",
                    "ours": minor["id"], "ours_name": name_of(minor["id"]),
                    "ours_delta": minor["delta"],
                    "ours_pr": minor["pr"], "ours_n": minor["n"],
                    "riot": best_riot["_id"] if best_riot else None,
                    "riot_name": name_of(best_riot["_id"]) if best_riot else "?",
                    "riot_delta": best_riot["delta"] if best_riot else None,
                    "gain": gain,
                })

    # Secondary minors
    for j, secm in enumerate(optimal["secondary_runes"], 1):
        chosen_id = str(secm["id"])
        if chosen_id not in rec_secondary_minors:
            best_riot = None
            for rid in rec_secondary_minors:
                info = sec.get(rid)
                if info and (best_riot is None or info["delta"] > best_riot["delta"]):
                    best_riot = {**info, "_id": rid}
            gain = secm["delta"] - (best_riot["delta"] if best_riot else 0)
            if (gain >= MIN_DELTA_GAIN
                    and secm["pr"] >= MIN_PR
                    and secm["n"] >= MIN_N):
                findings.append({
                    "type": f"secondary_pick_{j}",
                    "ours": secm["id"], "ours_name": name_of(secm["id"]),
                    "ours_delta": secm["delta"],
                    "ours_pr": secm["pr"], "ours_n": secm["n"],
                    "riot": best_riot["_id"] if best_riot else None,
                    "riot_name": name_of(best_riot["_id"]) if best_riot else "?",
                    "riot_delta": best_riot["delta"] if best_riot else None,
                    "gain": gain,
                })

    # Shards (per row, including row-2 'f' suffix handling)
    for row_idx, shard in enumerate(optimal["shards"], 1):
        # The shard id we display is bare (we strip 'f'). Match against the
        # recommended set for that row (which uses 'f' for row 2).
        chosen_id_str = str(shard["id"])
        # For row 2, our chosen ID has been stripped of 'f' for display, but
        # its key in stats is "{id}f". Compare against rec_shards[1] which
        # also uses 'f' suffix.
        if row_idx == 2:
            chosen_key = f"{chosen_id_str}f"
        else:
            chosen_key = chosen_id_str
        if chosen_key not in rec_shards[row_idx - 1]:
            best_riot = None
            for rid in rec_shards[row_idx - 1]:
                info = pri.get(rid)
                if info and (best_riot is None or info["delta"] > best_riot["delta"]):
                    best_riot = {**info, "_id": rid}
            gain = shard["delta"] - (best_riot["delta"] if best_riot else 0)
            if (gain >= MIN_DELTA_GAIN
                    and shard["pr"] >= MIN_PR
                    and shard["n"] >= MIN_N):
                findings.append({
                    "type": f"shard_row_{row_idx}",
                    "ours": shard["id"], "ours_name": name_of(shard["id"]),
                    "ours_delta": shard["delta"],
                    "ours_pr": shard["pr"], "ours_n": shard["n"],
                    "riot": best_riot["_id"] if best_riot else None,
                    "riot_name": name_of(best_riot["_id"]) if best_riot else "?",
                    "riot_delta": best_riot["delta"] if best_riot else None,
                    "gain": gain,
                })

    return findings


def main():
    # Load champion pools
    pool_path = os.path.join("data", TIER, "champions.json")
    with open(pool_path, "r", encoding="utf-8") as f:
        champ_data = json.load(f)
    pools = champ_data["pools"]

    # Build (champ_name, lane) work list, deduped
    work = []
    seen = set()
    for lane, role_pool in pools.items():
        for cid_str, name in role_pool.items():
            if (name, lane) in seen:
                continue
            seen.add((name, lane))
            work.append((name, lane))
    work.sort()
    print(f"Analyzing {len(work)} (champion, lane) pairs (intrinsic-only, no enemies)...")

    all_findings = []
    for i, (champ, lane) in enumerate(work):
        try:
            findings = analyze_champion_lane(champ, lane)
        except Exception as e:
            print(f"  {champ}/{lane}: error {e}")
            continue
        if findings:
            for f in findings:
                f["champion"] = champ
                f["lane"] = lane
                all_findings.append(f)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(work)}...")

    print(f"\nTotal findings: {len(all_findings)}")
    # Sort by gain
    all_findings.sort(key=lambda f: -f["gain"])

    # Group by type for readability
    print("\n" + "=" * 100)
    print(f"HIDDEN GEMS (gain >= {MIN_DELTA_GAIN}pp; intrinsic-only scoring)")
    print("=" * 100)
    for category in ("keystone", "primary_minor", "secondary_pick", "shard_row"):
        relevant = [f for f in all_findings if f["type"].startswith(category)]
        if not relevant:
            continue
        print(f"\n--- {category.upper()} ({len(relevant)}) ---")
        for f in relevant:
            print(f"  {f['champion']:<14} {f['lane']:<8} {f['type']:<20} "
                  f"ours: {f['ours_name']:<22} (Δ={f['ours_delta']:+.2f}, pr={f['ours_pr']:.1f}%, n={f['ours_n']:,}) "
                  f"vs Riot's best: {f['riot_name']:<22} (Δ={f['riot_delta']:+.2f}) "
                  f"→ gain {f['gain']:+.2f}pp")


if __name__ == "__main__":
    main()
