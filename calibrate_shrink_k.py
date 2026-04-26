"""
Empirical-Bayes calibration of SHRINK_K.

Model:
  observed_shift_i = (wr_i - intrinsic_wr)
                   = true_shift_i + sample_noise_i
  sample_noise_var_i = wr_i*(100-wr_i)/n_i  (binomial in pp^2)
  true_shift ~ Normal(0, prior_var)

Solve for prior_var:
  prior_var ≈ mean(observed_shift^2) − mean(sample_noise_var_i)

Then for Beta-Binomial prior with strength K:
  prior_var = baseline*(100-baseline) / (K+1)
  K = baseline*(100-baseline) / prior_var − 1

We compute prior_var across many (champ, rune) pairs and average.
"""
import sys
import statistics
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, ".")
import app


CHAMPS = [
    ("Aatrox",  "top"),       ("Garen",   "top"),       ("Camille",  "top"),
    ("Yasuo",   "middle"),    ("Ahri",    "middle"),    ("Sylas",    "middle"),
    ("Lee Sin", "jungle"),    ("Graves",  "jungle"),    ("Vi",       "jungle"),
    ("Caitlyn", "bottom"),    ("Jinx",    "bottom"),    ("Ezreal",   "bottom"),
    ("Nautilus","support"),   ("Lulu",    "support"),   ("Thresh",   "support"),
]

ENEMIES_BY_LANE = {
    "top":     ["Garen", "Sett", "Mordekaiser", "Riven", "Camille", "Darius", "Fiora", "Jax"],
    "jungle":  ["Lee Sin", "Graves", "Shaco", "Vi", "Viego", "Master Yi", "Wukong", "Diana"],
    "middle":  ["Ahri", "Yasuo", "Annie", "Vex", "Yone", "Akali", "Sylas", "Zed"],
    "bottom":  ["Jinx", "Caitlyn", "Vayne", "Ezreal", "Kai'Sa", "Lucian", "Jhin", "Aphelios"],
    "support": ["Nautilus", "Thresh", "Lulu", "Soraka", "Pyke", "Karma", "Leona", "Bard"],
}

ALL_LANES = list(ENEMIES_BY_LANE.keys())
MIN_N = 50  # require ≥50 samples to use a per-matchup observation

# Iterate over runes that pass our PR threshold globally (so we don't fold in
# selection-biased noise from 1% picks)
keystone_set = {str(rid) for tree in app.RUNE_TREES.values() for rid in tree["rows"][0]}


def collect_observations(champ, lane):
    """For each rune that passes our threshold globally, gather per-matchup
    (wr_i, n_i, intrinsic_wr) tuples. Returns list of (rune_id, intrinsic_wr, list of (wr_i, n_i))."""
    uncond = app.fetch_unconditioned_build(champ, lane, "emerald_plus", "30")
    uc_data = uncond.get("data")
    if not uc_data:
        return []
    intrinsic_stats = uc_data.get("runes", {}).get("stats", {})

    # Fetch matchups: 4 enemies per role × 5 roles = 20 per champ
    matchup_data = []
    for enemy_lane in ALL_LANES:
        pool = [e for e in ENEMIES_BY_LANE[enemy_lane] if e.lower() != champ.lower()]
        sample = pool[:4]  # first 4 popular ones, deterministic
        with ThreadPoolExecutor(max_workers=4) as tp:
            results = list(tp.map(
                lambda e, el=enemy_lane: app.fetch_vs_build(champ, lane, e, el, "emerald_plus", "30"),
                sample
            ))
        for r in results:
            d = r.get("data")
            if d:
                matchup_data.append(d)

    out = []
    for rune_id, entries in intrinsic_stats.items():
        is_ks = rune_id in keystone_set
        min_pr = app.MIN_KEYSTONE_PR if is_ks else app.MIN_RUNE_PR
        for i, entry in enumerate(entries):
            if i != 0:
                continue  # primary slot only — keystones are primary
            pr, wr, n = entry
            if n < 1000 or pr < min_pr:
                continue
            intrinsic_wr = wr
            per_matchup = []
            for d in matchup_data:
                stats = d.get("runes", {}).get("stats", {})
                info = stats.get(rune_id)
                if not info:
                    continue
                m_entry = info[0] if isinstance(info, list) and info else None
                if not m_entry or len(m_entry) < 3:
                    continue
                m_pr, m_wr, m_n = m_entry
                if m_n < MIN_N:
                    continue
                per_matchup.append((m_wr, m_n))
            if len(per_matchup) >= 3:
                out.append((rune_id, intrinsic_wr, per_matchup))
    return out


# ---------- collect observations across all champs ----------
print(f"Collecting observations across {len(CHAMPS)} champions...")
all_observations = []
for champ, lane in CHAMPS:
    obs = collect_observations(champ, lane)
    print(f"  {champ:13s} ({lane}): {len(obs)} runes with multi-matchup data")
    all_observations.extend(obs)

print(f"\nTotal: {len(all_observations)} (champ, rune) pairs")
print()

# ---------- estimate prior_var ----------
print("Per-(champ, rune) variance decomposition:")
print(f"{'rune':>6}  {'intrinsic':>10}  {'n_obs':>5}  {'shift²':>8}  {'noise':>8}  {'true_var':>8}")
print("-" * 60)

per_champ_var = []
for rune_id, intrinsic_wr, matchups in all_observations:
    shifts = [wr - intrinsic_wr for wr, n in matchups]
    sample_noises = [wr * (100 - wr) / n for wr, n in matchups]
    obs_var = statistics.mean([s**2 for s in shifts])
    noise_var = statistics.mean(sample_noises)
    true_var = max(0, obs_var - noise_var)
    per_champ_var.append(true_var)
    if len(per_champ_var) <= 30:
        print(f"  {rune_id:>5}  {intrinsic_wr:>10.2f}  {len(matchups):>5}  "
              f"{obs_var:>8.2f}  {noise_var:>8.3f}  {true_var:>8.2f}")

print()
mean_true_var = statistics.mean(per_champ_var)
median_true_var = statistics.median(per_champ_var)
print(f"Across {len(per_champ_var)} (champ, rune) pairs:")
print(f"  mean true_var = {mean_true_var:.3f} pp²")
print(f"  median true_var = {median_true_var:.3f} pp²")
print()

# ---------- compute optimal K ----------
print("Optimal K from K = baseline*(100-baseline)/true_var − 1, baseline=51.8:")
b = 51.8
denom = b * (100 - b)  # ≈ 2497.96
for label, tv in [("mean", mean_true_var), ("median", median_true_var)]:
    if tv > 0:
        k = denom / tv - 1
        print(f"  {label:6s} (true_var={tv:.2f}): K = {k:.0f}")
    else:
        print(f"  {label}: true_var = 0 → K → ∞ (no real shift signal)")
