"""
How much does each enemy role actually move the rune-page choice?

For each (test_champ, test_lane), compute:
  - "Vacuum" page (no matchups)
  - "Role-only" page for each enemy role: ONLY a representative enemy in that
    role, weight = 1.0 (max signal from that role alone, no others diluting).

Then count how often each role's addition flips the page (keystone, tree,
minor, secondary, shard). Roles that flip the page more often carry more
rune-relevant signal — that's evidence for weighting them higher.
"""
import sys
import os
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import app


# Disable PR threshold so we see every signal
app.MIN_KEYSTONE_PR = 0.0
app.MIN_RUNE_PR = 0.0

TIER = "emerald_plus"
PATCH = "30"

# Pick 3 representative popular enemies per role. The signal is averaged
# across these 3 within each role to reduce per-enemy noise.
REP_ENEMIES = {
    "top":     ["Garen",   "Sett",    "Mordekaiser"],
    "jungle":  ["Lee Sin", "Graves",  "Vi"],
    "middle":  ["Ahri",    "Yasuo",   "Annie"],
    "bottom":  ["Jinx",    "Caitlyn", "Ezreal"],
    "support": ["Nautilus","Thresh",  "Lulu"],
}

# Test set: 4 champions per picking lane, picked across keystone-flexibility
# (some monolithic, some multi-keystone).
TEST = {
    "top":     ["Aatrox", "Camille", "Garen",  "Yasuo"],
    "jungle":  ["Lee Sin", "Graves", "Kindred", "Vi"],
    "middle":  ["Ahri",   "Sylas",   "Yasuo",   "Akali"],
    "bottom":  ["Jinx",   "Caitlyn", "Ezreal",  "Lucian"],
    "support": ["Nautilus", "Lulu",  "Thresh",  "Pyke"],
}


def compute_page(champ, lane, enemies):
    """enemies: list of (name, role_lane) pairs. Empty list => vacuum."""
    with ThreadPoolExecutor(max_workers=6) as pool:
        uf = pool.submit(app.fetch_unconditioned_build, champ, lane, TIER, PATCH)
        if enemies:
            mfs = [pool.submit(app.fetch_vs_build, champ, lane, e[0], e[1], TIER, PATCH)
                   for e in enemies]
            results = [f.result() for f in mfs]
        else:
            results = []
        uncond = uf.result()
    if not uncond.get("data"):
        return None
    if len(enemies) == 1:
        weights = [1.0]
    elif len(enemies) > 1:
        weights = app.build_enemy_weights(lane, enemies)
    else:
        weights = []
    rune_info = app.combine_rune_stats(results, weights, uncond.get("data"),
                                       champion_name=champ, picking_lane=lane)
    return app.build_optimal_rune_page(rune_info)


def page_sig(p):
    if not p:
        return None
    return {
        "tree":       p["primary_tree"],
        "keystone":   p["primary_runes"][0]["id"],
        "minors":     tuple(r["id"] for r in p["primary_runes"][1:]),
        "sec_tree":   p["secondary_tree"],
        "sec_runes":  frozenset(r["id"] for r in p["secondary_runes"]),
        "shards":     tuple(s["id"] for s in p["shards"]),
    }


def diff(a, b):
    """Set of components that differ between page signatures."""
    if a is None or b is None:
        return {"missing"}
    out = set()
    if a["keystone"] != b["keystone"]: out.add("keystone")
    elif a["tree"]   != b["tree"]:     out.add("tree")
    if a["minors"]   != b["minors"]:   out.add("primary_minors")
    if a["sec_tree"] != b["sec_tree"]: out.add("sec_tree")
    elif a["sec_runes"] != b["sec_runes"]: out.add("sec_runes")
    if a["shards"]   != b["shards"]:   out.add("shards")
    return out


# Counters
flip_count = defaultdict(lambda: defaultdict(int))   # role → component → count
total_per_role = defaultdict(int)                    # role → trials
flip_examples = defaultdict(list)                    # role → list of (champ, lane, diff_set)

for picking_lane, champs in TEST.items():
    for champ in champs:
        vacuum = compute_page(champ, picking_lane, [])
        if vacuum is None:
            continue
        vacuum_sig = page_sig(vacuum)

        for enemy_role, reps in REP_ENEMIES.items():
            for rep in reps:
                # Skip self-matchup edge cases
                if rep.lower() == champ.lower():
                    continue
                page = compute_page(champ, picking_lane, [(rep, enemy_role)])
                if page is None:
                    continue
                d = diff(vacuum_sig, page_sig(page))
                total_per_role[enemy_role] += 1
                for component in d:
                    flip_count[enemy_role][component] += 1
                if d:
                    flip_examples[enemy_role].append((champ, picking_lane, rep, d))

# Report
print("=" * 80)
print("Per-role page-flip rates (vacuum → +1 enemy in role)")
print("=" * 80)
print(f"{'Role':<10} {'Trials':<8} {'AnyFlip':<10} {'Keystone':<10} {'Tree':<8} "
      f"{'Minors':<8} {'SecTree':<10} {'Shards':<8}")
print("-" * 80)
for role in ("top", "jungle", "middle", "bottom", "support"):
    n = total_per_role[role]
    if n == 0:
        continue
    counts = flip_count[role]
    # "AnyFlip" = number of trials where AT LEAST ONE component differs
    # (we don't have that directly so approx by max of component counts)
    keystone = counts.get("keystone", 0)
    tree = counts.get("tree", 0)
    minors = counts.get("primary_minors", 0)
    sec_tree = counts.get("sec_tree", 0)
    sec_runes = counts.get("sec_runes", 0)
    shards = counts.get("shards", 0)
    any_flip = sum(1 for ex in flip_examples[role]) if flip_examples[role] else 0
    print(f"{role:<10} {n:<8} {any_flip}/{n:<8} "
          f"{keystone:<10} {tree:<8} {minors:<8} {sec_tree+sec_runes:<10} {shards:<8}")

print()
print("Sample of flips (which champion changed which component when conditioned on role):")
for role in ("top", "jungle", "middle", "bottom", "support"):
    if flip_examples[role]:
        print(f"\n  vs {role}:")
        for champ, lane, rep, d in flip_examples[role][:6]:
            print(f"    {champ} ({lane}) vs {rep}: {sorted(d)}")
