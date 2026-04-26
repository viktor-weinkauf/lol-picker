"""
Calibration script: derive data-driven rune weights per picking lane.

Methodology:
  For each (champion, picking_lane), fetch builds vs N enemies in each enemy role.
  Build a "rune-WR vector" per matchup (WR for each rune with adequate samples).
  For each enemy role, compute mean (1 - Spearman correlation) across matchup pairs.
  Higher value = enemy identity in that role moves rune ordering more = more weight.
  Normalize per lane to sum to 1. Aggregate across champions in each picking lane.
"""
import sys
import random
import statistics
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, ".")
import app  # uses cache layer

# ---------- test configuration ----------
TEST_CHAMPS = {
    # 5 champs per picking lane, prioritizing those with viable multi-keystone choice
    "top":     ["Camille", "Irelia", "Yasuo", "Gnar", "Sett"],
    "jungle":  ["Graves", "Kindred", "Lillia", "Diana", "Wukong"],
    "middle":  ["Sylas", "Akali", "Ekko", "Yone", "Yasuo"],
    "bottom":  ["Caitlyn", "Ezreal", "Lucian", "Jhin", "Aphelios"],
    "support": ["Bard", "Pyke", "Karma", "Maokai", "Senna"],
}

ENEMY_POOLS = {
    "top":     ["Aatrox", "Camille", "Darius", "Fiora", "Garen", "Gnar", "Gwen",
                "Irelia", "Jax", "Kayle", "Mordekaiser", "Nasus", "Pantheon",
                "Renekton", "Riven", "Sett", "Shen", "Sion", "Teemo",
                "Tryndamere", "Volibear", "Yasuo", "Yone", "Yorick"],
    "jungle":  ["Diana", "Ekko", "Elise", "Graves", "Hecarim", "Jarvan IV",
                "Kha'Zix", "Kindred", "Lee Sin", "Lillia", "Master Yi", "Nidalee",
                "Nocturne", "Rengar", "Sejuani", "Shaco", "Vi", "Viego",
                "Volibear", "Warwick", "Wukong", "Xin Zhao", "Zac"],
    "middle":  ["Ahri", "Akali", "Anivia", "Annie", "Diana", "Ekko", "Fizz",
                "Galio", "Hwei", "Katarina", "LeBlanc", "Lissandra", "Lux",
                "Orianna", "Qiyana", "Sylas", "Syndra", "Vex", "Veigar",
                "Viktor", "Vladimir", "Yasuo", "Yone", "Zed", "Ziggs"],
    "bottom":  ["Aphelios", "Ashe", "Caitlyn", "Draven", "Ezreal", "Jhin",
                "Jinx", "Kai'Sa", "Kalista", "Lucian", "Miss Fortune", "Nilah",
                "Senna", "Sivir", "Smolder", "Tristana", "Twitch", "Varus",
                "Vayne", "Xayah", "Zeri"],
    "support": ["Alistar", "Bard", "Blitzcrank", "Braum", "Brand", "Janna",
                "Karma", "Leona", "Lulu", "Lux", "Maokai", "Milio", "Morgana",
                "Nami", "Nautilus", "Pyke", "Rakan", "Rell", "Renata Glasc",
                "Senna", "Seraphine", "Soraka", "Thresh", "Yuumi", "Zilean",
                "Zyra"],
}

LANES = ["top", "jungle", "middle", "bottom", "support"]
N_ENEMIES_PER_ROLE = 10
RUNE_MIN_N = 30

# All rune IDs from RUNE_TREES (keystones + minors), primary-slot stats
ALL_RUNES = []
KEYSTONE_SET = set()
for tree in app.RUNE_TREES.values():
    for i, row in enumerate(tree["rows"]):
        if i == 0:
            KEYSTONE_SET.update(str(r) for r in row)
        ALL_RUNES.extend(row)


# ---------- helpers ----------
def spearman(a, b):
    """Spearman rank correlation. Returns rho in [-1, 1] or None if undefined."""
    n = len(a)
    if n < 3:
        return None

    def ranks(arr):
        # average-rank for ties
        order = sorted(range(n), key=lambda i: arr[i])
        ranks_out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and arr[order[j + 1]] == arr[order[i]]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks_out[order[k]] = avg_rank
            i = j + 1
        return ranks_out

    ra, rb = ranks(a), ranks(b)
    mean_ra = sum(ra) / n
    mean_rb = sum(rb) / n
    num = sum((ra[i] - mean_ra) * (rb[i] - mean_rb) for i in range(n))
    var_a = sum((r - mean_ra) ** 2 for r in ra)
    var_b = sum((r - mean_rb) ** 2 for r in rb)
    if var_a == 0 or var_b == 0:
        return None
    return num / (var_a ** 0.5 * var_b ** 0.5)


def get_rune_vector(data):
    """Return dict {rune_id: wr} after applying the same biases the live system uses:
    - Phreak deflation on the modal keystone
    - MIN_KEYSTONE_PR / MIN_RUNE_PR cutoffs to drop selection-biased niche picks
    """
    stats = data.get("runes", {}).get("stats", {})
    modal_ks = app._modal_keystone(stats)
    out = {}
    for rid in ALL_RUNES:
        info = stats.get(str(rid))
        if not info:
            continue
        entry = info[0] if isinstance(info, list) else None
        if not entry or len(entry) < 3:
            continue
        pr, wr, n = entry
        is_keystone = str(rid) in KEYSTONE_SET
        min_pr = app.MIN_KEYSTONE_PR if is_keystone else app.MIN_RUNE_PR
        if n < RUNE_MIN_N or pr < min_pr:
            continue
        if is_keystone and str(rid) == modal_ks:
            wr = wr - app.PHREAK_BIAS
        out[rid] = wr
    return out


def gather(champ, lane, vs, vslane):
    return app.fetch_vs_build(champ, lane, vs, vslane, "emerald_plus", "30")


# ---------- run calibration ----------
random.seed(42)

n_total = sum(len(c) for c in TEST_CHAMPS.values()) * len(LANES) * N_ENEMIES_PER_ROLE
print(f"Calibrating: {sum(len(c) for c in TEST_CHAMPS.values())} champs × {len(LANES)} roles × {N_ENEMIES_PER_ROLE} enemies = {n_total} fetches")
print()

per_lane_per_role = {pl: {er: [] for er in LANES} for pl in LANES}

for picking_lane in LANES:
    for champ in TEST_CHAMPS[picking_lane]:
        print(f"[{picking_lane}/{champ}] fetching...", end=" ", flush=True)
        # Sample enemies per role (exclude self in the same role pool)
        per_role_results = {}
        for enemy_role in LANES:
            pool = [e for e in ENEMY_POOLS[enemy_role]
                    if e.lower() != champ.lower()]
            sample = random.sample(pool, min(N_ENEMIES_PER_ROLE, len(pool)))
            with ThreadPoolExecutor(max_workers=5) as tp:
                results = list(tp.map(
                    lambda e, er=enemy_role: gather(champ, picking_lane, e, er),
                    sample
                ))
            per_role_results[enemy_role] = [r.get("data") or {} for r in results]

        # Per role: pairwise Spearman, then mean (1 - rho)
        change_per_role = {}
        for enemy_role in LANES:
            datas = per_role_results[enemy_role]
            vecs = [get_rune_vector(d) for d in datas]
            distances = []
            for i in range(len(vecs)):
                for j in range(i + 1, len(vecs)):
                    common = sorted(set(vecs[i]) & set(vecs[j]))
                    if len(common) < 5:
                        continue
                    a = [vecs[i][r] for r in common]
                    b = [vecs[j][r] for r in common]
                    rho = spearman(a, b)
                    if rho is None:
                        continue
                    distances.append(1.0 - rho)
            if distances:
                change_per_role[enemy_role] = statistics.mean(distances)

        # Normalize within champ
        total = sum(change_per_role.values())
        if total <= 0:
            print("(no signal)")
            continue
        norm = {er: change_per_role.get(er, 0) / total for er in LANES}
        print("  " + "  ".join(f"{er[:3]}={norm[er]:.3f}" for er in LANES))
        for er in LANES:
            per_lane_per_role[picking_lane][er].append(norm[er])

print()
print("=" * 78)
print("Aggregate RUNE_WEIGHTS (mean across champions per picking lane)")
print("=" * 78)
for pl in LANES:
    avg = {er: (statistics.mean(per_lane_per_role[pl][er])
                if per_lane_per_role[pl][er] else 0.0)
           for er in LANES}
    total = sum(avg.values())
    norm = {er: avg[er] / total for er in LANES} if total > 0 else avg
    s = "  ".join(f"{er[:3]}={norm[er]:.3f}" for er in LANES)
    print(f"  {pl:8s} | {s}")
