"""Test Soraka with dynamic gap-based Phreak (PR-gap → deflation magnitude)."""
import sys
sys.path.insert(0, '.')
import app

NAMES = {
    "8005": "Press the Attack", "8008": "Lethal Tempo", "8021": "Fleet Footwork",
    "8010": "Conqueror", "8214": "Summon Aery", "8229": "Arcane Comet",
    "8230": "Phase Rush", "8437": "Grasp of the Undying", "8439": "Aftershock",
    "8465": "Guardian", "8351": "Glacial Augment", "8360": "Unsealed Spellbook",
    "8369": "First Strike", "8112": "Electrocute", "8128": "Dark Harvest",
    "9923": "Hail of Blades",
    "9111": "Triumph", "8009": "Presence of Mind", "9101": "Absorb Life",
    "9104": "Legend: Alacrity", "9105": "Legend: Haste", "9103": "Legend: Bloodline",
    "8014": "Coup de Grace", "8017": "Cut Down", "8299": "Last Stand",
    "8224": "Axiom Arcanist", "8226": "Manaflow Band", "8275": "Nimbus Cloak",
    "8210": "Transcendence", "8234": "Celerity", "8233": "Absolute Focus",
    "8237": "Scorch", "8232": "Waterwalking", "8236": "Gathering Storm",
    "8446": "Demolish", "8463": "Font of Life", "8401": "Shield Bash",
    "8429": "Conditioning", "8444": "Second Wind", "8473": "Bone Plating",
    "8451": "Overgrowth", "8453": "Revitalize", "8242": "Unflinching",
    "8306": "Hextech Flashtraption", "8304": "Magical Footwear", "8321": "Cash Back",
    "8313": "Triple Tonic", "8352": "Time Warp Tonic", "8345": "Biscuit Delivery",
    "8347": "Cosmic Insight", "8410": "Approach Velocity", "8316": "Jack Of All Trades",
    "8126": "Cheap Shot", "8139": "Taste of Blood", "8143": "Sudden Impact",
    "8137": "Sixth Sense", "8140": "Grisly Mementos", "8141": "Deep Ward",
    "8135": "Treasure Hunter", "8105": "Relentless Hunter", "8106": "Ultimate Hunter",
    "5008": "Adaptive Force", "5005": "Attack Speed", "5007": "Ability Haste",
    "5010": "Movement Speed", "5001": "Health Scaling", "5011": "Health", "5013": "Tenacity",
}


def n(rid):
    s = str(rid)
    if s.endswith("f"):
        s = s[:-1]
    return NAMES.get(s, s)


PHREAK_REF_GAP = 60.0
PHREAK_REF_BIAS = 2.0


def combine_dynamic_phreak(unconditioned_data, champion_name, picking_lane):
    """combine_rune_stats but with dynamic Phreak based on PR gap within row."""
    rune_info = {"pri": {}, "sec": {}}
    if not unconditioned_data:
        return rune_info
    keystone_set = {str(rid) for tree in app.RUNE_TREES.values() for rid in tree["rows"][0]}
    intrinsic_baseline = unconditioned_data.get("avgWr", 50.0)
    intrinsic_stats = unconditioned_data.get("runes", {}).get("stats", {})

    recommended_slots = app.get_recommended_perk_slots(champion_name, picking_lane)

    # Build row-membership map for each (rune_id, slot) tuple
    row_of = {}  # (rune_id, slot) -> list of rune ids in same row
    for tree in app.RUNE_TREES.values():
        # Keystones use ALL keystones across all trees as the choice set
        # (a player choosing the keystone picks among 16 globally)
        all_keystones = [str(r) for tt in app.RUNE_TREES.values() for r in tt["rows"][0]]
        for rid in tree["rows"][0]:
            row_of[(str(rid), 'pri')] = all_keystones
        # Minors — within the row of the same tree
        for row in tree["rows"][1:]:
            row_keys = [str(r) for r in row]
            for rid in row:
                row_of[(str(rid), 'pri')] = row_keys
                row_of[(str(rid), 'sec')] = row_keys
    # Shards — within the same shard row.
    # Note: RUNE_SHARDS["row2"] already contains strings with 'f' suffix; just
    # str() them all (handles int → "5008" and "5008f" → "5008f").
    for row_key, ids in app.RUNE_SHARDS.items():
        keys = [str(i) for i in ids]
        for k in keys:
            row_of[(k, 'pri')] = keys

    def gap_based_phreak(rune_id, slot):
        if slot not in recommended_slots.get(rune_id, set()):
            return 0.0
        peers = row_of.get((rune_id, slot))
        if not peers:
            return 0.0
        prs = []
        for rk in peers:
            info = intrinsic_stats.get(rk)
            if not info:
                continue
            idx = 0 if slot == 'pri' else 1
            entries = info if isinstance(info, list) else []
            if idx < len(entries):
                pr, _, n_val = entries[idx]
                if n_val >= 30:
                    prs.append((rk, pr))
        if len(prs) < 2:
            return 0.0
        prs.sort(key=lambda x: -x[1])
        modal_id, modal_pr = prs[0]
        second_pr = prs[1][1]
        if modal_id != rune_id:
            return 0.0
        gap = max(0.0, modal_pr - second_pr)
        return PHREAK_REF_BIAS * gap / PHREAK_REF_GAP

    rune_intrinsic = {"pri": {}, "sec": {}}
    for rune_id, entries in intrinsic_stats.items():
        is_keystone = rune_id in keystone_set
        min_pr_t = app.MIN_KEYSTONE_PR if is_keystone else app.MIN_RUNE_PR
        for i, entry in enumerate(entries):
            slot = "pri" if i == 0 else "sec"
            pr, wr, n_val = entry
            if n_val < app.MIN_RUNE_N:
                continue
            phreak = gap_based_phreak(rune_id, slot)
            rune_intrinsic[slot][rune_id] = {
                "wr": wr, "pr": pr, "n": n_val,
                "eligible": pr >= min_pr_t, "phreak": phreak,
            }

    for slot in ("pri", "sec"):
        for rune_id, intr in rune_intrinsic[slot].items():
            raw = intr["wr"] - intrinsic_baseline
            shrink = intr["n"] / (intr["n"] + app.SHRINK_K)
            d = shrink * raw + intr["phreak"]
            rune_info[slot][rune_id] = {
                "delta": d, "intrinsic_delta": d, "matchup_shift": 0.0,
                "wr": intr["wr"] + intr["phreak"],
                "n": intr["n"], "pr": intr["pr"],
                "eligible": intr["eligible"],
                "recommended": intr["phreak"] > 0,
                "phreak": intr["phreak"],
            }
    return rune_info


# Disable thresholds
app.MIN_KEYSTONE_PR = 0.0
app.MIN_RUNE_PR = 0.0

result = app.fetch_unconditioned_build('Soraka', 'support', 'emerald_plus', '30')
rune_info = combine_dynamic_phreak(result['data'], 'Soraka', 'support')
optimal = app.build_optimal_rune_page(rune_info)

print("=== Soraka SUPPORT — DYNAMIC PHREAK (60pp gap → 2pp deflation) ===\n")
print(f"Primary tree: {optimal['primary_tree_name']}")
for i, r in enumerate(optimal['primary_runes']):
    label = "Keystone" if i == 0 else f"Row {i}"
    full = rune_info['pri'].get(str(r['id']), {})
    phr = full.get('phreak', 0)
    print(f"  {label:<10} {n(r['id']):<22} (intr={r['intrinsic_delta']:+.2f}, "
          f"Phreak={phr:+.2f}, pr={r['pr']:.1f}%)")
print(f"\nSecondary tree: {optimal['secondary_tree_name']}")
for r in optimal['secondary_runes']:
    full = rune_info['sec'].get(str(r['id']), {})
    phr = full.get('phreak', 0)
    print(f"  Sec        {n(r['id']):<22} (intr={r['intrinsic_delta']:+.2f}, "
          f"Phreak={phr:+.2f}, pr={r['pr']:.1f}%)")
print(f"\nShards:")
for i, s in enumerate(optimal['shards'], 1):
    key = str(s['id']) + ('f' if i == 2 else '')
    full = rune_info['pri'].get(key, {})
    phr = full.get('phreak', 0)
    print(f"  Row {i}      {n(s['id']):<22} (intr={s['intrinsic_delta']:+.2f}, "
          f"Phreak={phr:+.2f}, pr={s['pr']:.1f}%)")

# Show row-by-row Sorcery row 1 with both runes
print("\n=== Sorcery row 1 (Axiom vs Manaflow vs Nimbus) ===")
for rid in ('8224', '8226', '8275'):
    info = rune_info['pri'].get(rid)
    if info:
        raw_wr = info['wr'] - info['phreak']
        print(f"  {n(rid):<20} pr={info['pr']:5.1f}%  raw_wr={raw_wr:5.2f}%  +Phreak={info['phreak']:+5.2f}"
              f"  → delta={info['delta']:+5.2f}")
