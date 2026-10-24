# Rosella Carrier-Appetite Call List

> **Task**: Generate an ordered list of trucking companies to call, prioritised by how appropriately each is placed with their current insurance carrier.

Deliverables (built by `rosella investigation.ipynb`):

| File | Shape | Purpose |
|---|---|---|
| `rosella_appetite_outliers_v3.csv` | 10K rows × ~26 cols | **The call list.** Wide format — top-20 recommended carriers per DOT per line, sorted by appetite-outlier priority score. |
| `rosella_rankings_v3_long.csv` | 23M rows × 9 cols | Full long-format rankings for programmatic use. One (DOT, line, state, carrier) per row with fit score, rank, pool size, and best-fit sub-entity. |
| `carrier_consolidation_report.csv` | ~2.1K rows | Auxiliary — maps raw insurer strings to consolidated carrier groups with per-mapping provenance (data-provider map / rule tier / raw fallback). |
| `rule_vs_provider_disagreements.csv` | ~10 rows | Residual label-disagreement log after the carrier-consolidation reconciliation. |

---

## 1. TL;DR

- **Scope**: 4 insurance lines modelled per DOT: `BIPD` (auto liability), `cargo`, `wc` (workers comp), `surety_bond` (BMC-84 broker bond). `gl` and `surety` dropped due to systematically NULL event-status in the source data.
- **Approach**: Binary contrastive-fit LightGBM model per line. For each DOT, rank the carriers in their state's pool by predicted "appetite fit." Low-ranking current placements become appetite outliers — high-priority call candidates.
- **Features** (170+ total): fleet shape, risk record, SMS BASIC scores, authority filings, insurance history, 30 commodity flags, timing/recency, derived ratios, 30 carrier-level aggregates (power/crash/commodity mix per carrier × state cell), service composite (tenure, exit, rejection), carrier momentum (6m vs 6-18m).
- **Carrier consolidation**: hybrid — data provider's `insurer_group` first (2.7% of events have it), then 60+ rule-based regexes, then raw-fallback. Reconciled: ~1,400 consolidated groups from ~2,100 distinct insurer strings.
- **Headline results** (model vs popularity-by-fleet-band baseline, top-5 hit rate):
  - BIPD: **87.8%** vs 68.7% baseline — **+19 pts**
  - WC: **85.3%** vs 51.8% — **+34 pts**
  - cargo / surety_bond: metrics saturate near 100% due to small pools (acknowledged as pool-size artefact, not pure quality)
- **Validation suite** (Phase 4): feature importance, error analysis, known-good/known-bad audit, calibration, pool-size scatter, synthetic-DOT sanity, temporal drift check, weighted-vs-base labelling comparison. All pass or produce documented caveats.

---

## 2. Problem framing

The deliverable is an **ordered call list for insurance brokers**. The ordering is "how appropriately placed is this DOT with their current carrier?" — DOTs placed with a carrier that *shouldn't* be writing their risk are high-priority calls (opportunity to pitch a better fit).

Reformulated as a ranking problem: for each DOT, rank the carriers in their state's eligible pool. The DOT's actual carrier is somewhere in that ranking. If it ranks near the top → placement looks appropriate. If it ranks near the bottom → appetite outlier → prioritise for the broker's call list.

Signal source: observed voluntary placements from ~4.5M DOTs across 24 months. Positives are "carrier Y was willing to voluntarily insure DOT X" — a proxy for appetite that aggregates across the market. We don't have ground-truth labels for "optimal placement" (would need retention/churn data), so appropriateness is inferred from the aggregate market pattern of which carriers take which types of DOTs.

---

## 3. Data

- **Source**: `usdot_insurance_events_2026-10-12.csv` (43 GB CSV, 12.24M events, 370 columns).
- **Grain**: insurance events over DOT history, each with policy_line, event_status, effective_date, cancellation_or_end_date, residual_market flags, carrier identity, and extensive DOT census/safety/authority joined columns.
- **Vintage anchoring**: `MAX(effective_date)` sanitised against outliers (year 9001 entries in raw data). All 24-month windows are vintage-relative, not today-relative, so re-runs are deterministic.
- **Snapshot columns**: three "as-of 2026-05-26" columns provide the authoritative current BIPD / cargo / WC placements per DOT. For `surety_bond`, placements are derived from events with `event_status IN ('active','active (cancellation pending)')`.
- **Data-quality cleaning** (cell 2E):
  - Dates past today or before 1990 → NaT (nulled, not clipped — honest handling).
  - Numerics past plausible bounds (e.g. `power_units > 100k`, `years_in_business > 150`) → NaN (nulled). LightGBM handles NaN natively; DuckDB `SUM`/`AVG` skip NULL so LOO aggregates stay clean.
  - `effective_date` capped at `CURRENT_DATE` (past-only), expiration/cancellation dates capped at year 2100 (allows multi-decade surety bonds).

### 3.1 Lines excluded from modelling (`gl`, `surety`)

Part 2F diagnostic found that GL (7,659 DOTs) and non-BMC-84 surety (3,274 DOTs) events have `event_status` **systematically NULL** in the source. With no status column, "latest event per DOT" cannot distinguish current voluntary placements from historical/cancelled ones. Any imputation would silently mislabel ex-customers as positives. Dropped from scope. `surety_bond` (BMC-84 broker bond — 12,466 active DOTs) has clean status and is retained.

### 3.2 Carrier consolidation

Approximately 2,100 distinct insurer strings reduce to ~1,400 carrier groups via a hybrid pipeline (`consolidate_one`):

1. **Data provider's `insurer_group` map** (highest confidence — populated for 2.7% of events). ~1,000 raw → group entries.
2. **Rule-based regexes** (60+ patterns, tiered T1-T4 by confidence) — covers acquisitions, brand variants, state subsidiaries. Example: `GEICO *` → `BERKSHIRE HATHAWAY GRP`.
3. **Raw-fallback** — if neither matches, the raw insurer string is its own group.

Rule-vs-provider disagreements were reviewed in `rule_vs_provider_disagreements.csv` and 35 label mismatches reconciled (rules re-aligned to provider's canonical labels, several over-matching rules tightened or removed).

**Provider wins on conflicts** — they have the most recent snapshot of carrier ownership; our rules are secondary.

---

## 4. Pipeline

Phase structure in the notebook. Each phase has caching so a kernel restart replays in ~2 min from on-disk artefacts rather than ~45 min from scratch.

| Phase | Cells | Produces | Cache | Time (cold) |
|---|---|---|---|---|
| 1 — Setup, Carrier Consolidation & Feature Build | 1A-1C, 1E | config, `CARRIER_RULES`, `_DATA_VINTAGE`, `DATA_PROVIDER_MAP`, `dot_features` DuckDB table | `vintage_and_provider_map.pkl` + DuckDB file | ~10 min |
| 2 — Aggregates, Centroids & Scorable DOTs | 2A-2E | `carrier_lookup`, `recent_events`, `service_composite`, `carrier_state_agg`, `dot_cell_contrib`, `sub_entity_centroids`, `dot_subent_contrib`, `dot_df` (pandas) | DuckDB tables + `dot_df.parquet` | ~15 min |
| **3 — EDA & Data Quality (new)** | 1D sanity check, 2F placement counts | print-only diagnostics | — | <1 min |
| 4 — Model Training | 3A-3C | `MODELS` (4 LightGBM boosters, one per line) | `models.pkl` (joblib) | ~5 min |
| 5 — Scoring & Validation | 4A-4K | `SCORED`, `OBS_RANK`, feature importance, error analysis, known-good/known-bad audit, weighted-tier retrain + comparison, calibration, pool-size scatter, synthetic DOT, temporal validation (approximate), true temporal train/test split | `scored_obs_rank.pkl`, `models_weighted.pkl`, `scored_obs_rank_weighted.pkl`, `line_tenure.pkl`, `models_temporal.pkl`, `results_temporal.pkl` | ~50 min |
| 6 — Deliverables | 5A-5B | `long_ranked`, Option B sub-entity attached, `rosella_*_v3.csv` output | — | ~10 min |
| 7 — Market analysis | 6A-6C | CA/TX market structure, carrier momentum, appetite-outlier call sheets | — | ~1 min |
| 8 — Recommendation hygiene & multi-line analysis | 7A, 7B | Multi-line bundling analysis + hygiene post-processing (`*_hygiene.csv`) | — | ~2 min |

### 4.1 Memory architecture

- **DuckDB with `memory_limit = '6GB'`, `threads = 2`, `preserve_insertion_order = false`** — tuned for a laptop with limited RAM.
- **Cell 1E (`dot_features`) chunked into two passes** (core + feature expansion) because a single 163-aggregate GROUP BY over 12M events exceeded 6 GB. Each pass has ~80 aggregates; join at the end.
- **Cell 4A (streaming scoring) uses intra-state DOT batching** (`SCORING_BATCH_DOTS = 3000`) — big states (TX, CA, 50K+ DOTs × 70 carriers = 3.5M-row cross-join) blew past 6 GB otherwise.
- **File-backed DuckDB** (`cache/analysis.duckdb`) — tables persist across kernel restarts.

---

## 5. Feature engineering

### 5.1 DOT-level features (170+)

Numeric (`DOT_NUM_FEATURES`, 133):
- **Fleet shape**: `power_units`, `total_drivers`, `total_cdl`, `truck_units`, radius mix (interstate/intrastate × within/beyond 100mi)
- **Risk record**: `crash_count`, `crash_count_24m`, `crash_fatalities`, `inspection_count`, `inspection_oos_viol`, `oos_order_count`, `revocation_count`, `prior_revoke_flag`
- **SMS BASIC scores**: unsafe-driving, HOS, driver-fit, vehicle-maintenance, controlled-substances (measures + inspection totals + violation counts + alert categories)
- **Insurance history**: `ins_event_count`, `ins_distinct_insurers`, cancelled/replaced/rejected counts, years-with-coverage
- **Coverage**: `coverage_amount_usd`, `underlying_limit_usd`, `min_required_coverage_usd`, `coverage_vs_minimum_ratio`
- **Timing / recency**: `days_since_bipd_effective`, `years_with_current_carrier`, `days_since_last_crash/inspection/oos/revocation`, `days_since_insurance_latest`, `days_since_mcs150_update`
- **Derived ratios**: `crash_rate_per_pu`, `oos_rate_per_pu`, `violation_rate`, `driver_oos_rate`, `vehicle_oos_rate`, `hazmat_oos_rate`, `insurance_cancel_rate`, `insurance_rejection_rate`, `carrier_switch_rate`, `driver_per_pu`
- **Rare-event flags**: `had_any_fatal_crash`, `had_any_revocation`, `had_any_oos_order`, `had_any_rejection`
- **30 commodity flags** (`crgo_genfreight`, `crgo_hazmat`, `crgo_passengers`, etc.)

Categorical (`DOT_CAT_FEATURES`, 41):
- `state`, `phy_state`, `authority_bus_state_code`
- `naics_sector`, `naics_subsector`, `co_lifecycle_status`, `org_type`
- `fleetsize_band`, `carrier_operation`, `classdef`, `hazmat_ind`, `fleet_band`
- Authority stat/filing/req/pend/chk columns (18)
- SMS alert-category columns (5)
- `co_match_confidence`, `census_status_code`, `safety_rating`, `carship`

### 5.2 Carrier-level features (30)

Computed per `(carrier_group, state, line)` cell with `MIN_CARRIER_PLACEMENTS` per line (BIPD/cargo/WC=20, surety_bond=15):

- **Portfolio profile**: `carr_mean_power`, `carr_median_power`, `carr_p90_power`, `carr_mean_crash`, `carr_mean_insp_oos`, `carr_mean_revoc`, `carr_mean_oos`, `carr_pct_hazmat`, 10 commodity shares, radius mix means
- **Service composite** (per carrier): `svc_short_tenure_cancel_rate`, `svc_rejection_rate`, `svc_voluntary_exit_ratio`, `svc_mean_tenure_days`
- **Momentum**: `carr_placements_6m`, `carr_placements_6to18m`, `carr_state_momentum` (6m / (6-18m/2 + 1))
- **Residual market share**: `carr_pct_residual_market` — carriers with ≥50% get filtered from the recommendation pool

### 5.3 Leave-one-out (LOO) adjustment

A naive issue: `carr_mean_power` for carrier Y in state S is computed over all DOTs placed with Y — including the DOT we're trying to score. That's leakage.

Fix (cell 3A `apply_loo_adjustment`): maintain `_sum` columns alongside the `_mean/_pct` features. At merge time, for each (DOT, carrier) pair where the DOT is actually in that cell, subtract the DOT's contribution:

```
carr_mean_power_loo = (carr_power_sum - dot_n_events * dot_power_units) / (carr_n_events - dot_n_events)
```

18 mean/pct features get this treatment. For DOTs not in a given cell (soft negatives), the LOO adjustment is a no-op. `carr_n_placements` and the 6m/6-18m momentum counts get analogous DOT-subtraction (`dot_in_6m`, `dot_in_6to18m` indicators). Median/p90 can't be cheaply LOO-adjusted and are left as-is (minor residual leakage bounded by `MIN_CARRIER_PLACEMENTS`).

### 5.4 Sub-entity centroids (Option B distance)

Each consolidated carrier group often has multiple raw sub-entities (e.g. `PROGRESSIVE GRP` → `PROGRESSIVE COUNTY MUTUAL`, `PROGRESSIVE DIRECT`, etc.). After we recommend a group, we also recommend the specific sub-entity whose typical client profile matches the DOT best, via a variance-weighted Euclidean distance on 10 appetite features (`power_units_log`, `crash_count`, `inspection_oos_viol`, 7 commodity flags). Weights are proportional to within-cell variance across sub-entities — high-variance features *are* the discriminative ones.

Also LOO-adjusted (if DOT places with a specific sub-entity, that sub-entity's centroid excludes the DOT's contribution — multi-event-aware).

---

## 6. Modelling

### 6.1 Labelling scheme (Option C)

Per training example `(DOT, carrier)`:

- **Clean positive** (`y=1`): DOT's actual voluntary placement for this line, where `is_residual_market == 0 AND is_cancellation_pending == 0`. Residual-market and cancellation-pending placements are **not** voluntary-market signal and get excluded.
- **Hard negative** (`y=0`): same DOT paired with their *actual residual-market* carrier (meaning the DOT was flagged `is_residual_market=1`), OR a `(DOT, carrier)` pair where `end_reason='revoked'` in `revoked_pairs`.
- **Soft negative** (`y=0`): random `K=3` carriers per positive drawn from the DOT's state carrier pool, deduped against the DOT's actual placements. Deliberately samples per-positive so DOTs with multiple positives get proportional soft negatives.

### 6.2 Training

- **Model**: LightGBM (`n_estimators=200, num_leaves=31, learning_rate=0.08, reg_lambda=1.0, min_child_samples=20`), with HGBT fallback if LightGBM import fails.
- **CV**: 3-fold `GroupKFold` with `groups=usdot` so a DOT isn't split across train/val.
- **Final model**: retrained on full training set after CV.
- **Separate model per line** (`bipd`, `cargo`, `wc`, `surety_bond`).

### 6.3 Alternative labelling: weighted-tier (sensitivity check in Phase 4E/4F)

A stricter interpretation of "appropriately placed" would require placements to have *persisted*. Phase 4E trains a parallel model with sample weights by positive tier:

| Tier | Condition | Weight |
|---|---|---|
| Strong | clean + `line_tenure >= STRICT_TENURE_MIN_PER_LINE[line]` + `insurance_cancel_rate < 0.3` | 1.0 |
| Medium | clean + `line_tenure in [0.5y, threshold)` | 0.6 |
| Weak | clean + `line_tenure < 0.5y` (included, not excluded) | 0.3 |
| Hard neg | residual market, revoked | 1.0 |
| Soft neg | random | 0.3 |

Per-line tenure thresholds: BIPD 1.5y, WC 1.0y, cargo 0.5y, surety_bond 0.5y. Line-specific tenure computed from `recent_events` (`cache/line_tenure.pkl`).

**Verdict**: Base model (Option C) beats weighted on all metrics, including the "ground-truth proxy" of long-tenure placements (≥1.5y). See §7.2.

### 6.4 Residual-market carrier filtering

Carriers with `carr_pct_residual_market >= 0.5` (half their events are residual-market) are filtered **from the recommendation pool** in cell 4A — brokers shouldn't be pitching DOTs toward residual-market carriers. They **stay in training** as hard negatives, so the model still learns to down-score them for voluntary DOTs.

---

## 7. Validation suite (Phase 4)

### 7.1 Headline metrics (base model, Option C labelling)

Model vs two baselines:
- **B1 random**: random carrier from state's pool
- **B2 popularity**: popularity-by-state-×-fleet-band rank

Reported on observed placements (actual DOT → actual carrier), top-K hit rate = % of DOTs where actual carrier is in model's top-K.

| Line | n | Metric | **Model** | B1 random | B2 popularity | **Model − B2** |
|---|---|---|---|---|---|---|
| **BIPD** | 320,329 | top-1 | 57.3% | 4.6% | 32.1% | **+25.2 pts** |
| | | top-5 | **87.8%** | 18.4% | 68.7% | **+19.1 pts** |
| | | top-10 | **94.2%** | 32.6% | 80.9% | **+13.3 pts** |
| | | MRR | 0.706 | 0.136 | 0.487 | **+0.22** |
| **WC** | 83,910 | top-1 | 58.8% | 2.9% | 28.6% | **+30.1 pts** |
| | | top-5 | **85.3%** | 13.9% | 51.8% | **+33.5 pts** |
| | | top-10 | **92.4%** | 25.3% | 65.6% | **+26.8 pts** |
| | | MRR | 0.706 | 0.100 | 0.400 | **+0.31** |
| cargo | 3,304 | top-5 | 100% | 83.8% | 68.1% | *saturated* |
| surety_bond | 10,451 | top-5 | 100% | 60.7% | 87.9% | *saturated* |

**BIPD and WC**: significant, real lift over popularity. Model learns genuine appetite patterns beyond coarse fleet-size matching.

**cargo and surety_bond**: metrics saturate at ~100% because the carrier pools are tiny (cargo: 64 cells total ⇒ 2-5 carriers per state; surety_bond: 206 cells ⇒ 3-10 carriers per state). Hit rates are misleadingly high — document as a pool-size artefact rather than genuine model quality.

Segmented BIPD (by fleet band) shows the expected monotonic degradation — easier for small-fleet DOTs (fewer carriers writing them, more concentrated market), harder for mega-fleets (bespoke relationships, more diverse carrier mix):

| Fleet band | n | MRR | top-5 |
|---|---|---|---|
| B1 (≤1 truck) | 176,390 | 0.759 | 0.913 |
| B2 (2-5) | 97,342 | 0.681 | 0.867 |
| B3 (6-20) | 33,084 | 0.576 | 0.788 |
| B4 (21-100) | 11,180 | 0.528 | 0.743 |
| B5 (100+) | 2,167 | 0.480 | 0.721 |

### 7.2 Weighted-tier labelling comparison (Phase 4F)

| Line | Metric | Base (Option C) | Weighted-tier | Delta |
|---|---|---|---|---|
| BIPD | top-5 (all) | 0.878 | 0.872 | −0.007 |
| BIPD | top-5 (tenure ≥ 1.5y) | 0.889 | 0.881 | −0.008 |
| WC | top-5 (all) | 0.853 | 0.826 | −0.028 |
| WC | top-5 (tenure ≥ 1.5y) | 0.851 | 0.837 | −0.014 |
| WC | MRR (tenure ≥ 1.5y) | 0.723 | 0.693 | −0.029 |

Base model outperforms weighted on **both** standard metrics and the long-tenure proxy for "appropriately placed." The weighted-tier approach is retained in the notebook as a sensitivity check but **the deliverable uses the base model** (set `MODEL_FOR_DELIVERABLE = 'base'` in cell 1A).

Methodologically weighted is more directly aligned with the task framing, but empirically it loses on all slices — likely because the tenure threshold introduces selection bias toward older DOTs rather than meaningfully improving the "appropriate" signal.

### 7.3 Feature importance (Phase 4B)

Per-line share of importance mass:

| Line | Carrier features | DOT-numeric | DOT-categorical |
|---|---|---|---|
| BIPD | 63.2% | 22.0% | 14.7% |
| WC | 82.0% | 7.1% | 10.9% |
| Surety_bond | 92.0% | 4.4% | 3.7% |
| Cargo | 51.4% | 45.9% | 2.6% |

**Carrier features dominate for 3 of 4 lines** — model is learning actual carrier appetite, not just coarse DOT-level patterns. Cargo's 46% DOT-numeric share is a yellow flag: with only 64 cells of carrier-side signal, the model falls back on DOT-level correlates.

Top BIPD features: `carr_n_placements`, `days_since_bipd_effective`, `svc_short_tenure_cancel_rate`, `authority_bipd_file`, `state`, `carr_mean_oos`, commodity-share features.

Top WC features: `carr_n_placements`, `carr_pct_livestock`, `naics_subsector`, `carr_mean_oos`, `carr_pct_meat`, `carr_state_momentum`, `carr_mean_power`.

### 7.4 Error analysis (Phase 4C)

Observed-rank distribution for BIPD:

| Rank bucket | % of DOTs |
|---|---|
| rank ≤ 1 | 57.3% |
| rank ≤ 5 | 87.8% |
| rank ≤ 10 | 94.2% |
| rank ≤ 20 | 97.9% |
| rank > 10 (hard cases) | 5.8% |

Hard cases — DOTs where actual carrier ranked >10 — are overwhelmingly **Texas county mutuals and specialty programs**:

- `MERCURY COUNTY MUTUAL`, `INFINITY COUNTY MUTUAL` — Texas's non-standard auto market structure
- `BENCHMARK INS`, `STAR INS`, `MSIG SPECIALTY`, `KNIGHT SPECIALTY`, `HOXBRIDGE` — specialty surplus programs
- `TEXAS FARM BUREAU` — regional affinity carrier

Model consistently recommends mainstream carriers (Progressive, BRK, Travelers, Canal, Great West) for these DOTs. This is the use case working as intended: a 1-truck non-hazmat TX DOT on Mercury County Mutual *is* a plausible appetite outlier for a broker to call — they'd likely qualify for mainstream-market pricing.

### 7.5 Known-good / known-bad audit (Phase 4D)

Industry-knowledge check independent of data patterns: hard-coded buckets of 12 mainstream carriers (Progressive, Travelers, BRK, Zurich, Chubb, CNA, Liberty, Nationwide, Great West, Canal, Hartford, AIG) vs 14 specialty/residual carriers (Texas Mutual, Mercury/Infinity County Mutual, Prime, Knight Specialty, Trisura, Southlake Specialty, etc.).

For voluntary-market DOTs, where does each bucket land in the model's ranking?

| Line | Mainstream median rank | Specialty median rank | Gap |
|---|---|---|---|
| BIPD | 7 | 29 | **+22** |
| WC | 12 | 52 | **+40** |

Model structurally separates mainstream from specialty for voluntary DOTs — PASS. Cargo and surety_bond: all specialty carriers were pre-filtered from the scoring pool by the residual-market threshold, so no comparison needed.

### 7.6 Fit score calibration (Phase 4G)

Fit score monotonically indicates placement likelihood. Observed-placement rate per decile of fit score:

| Line | Decile 0 | Decile 5 | Decile 9 (highest fit) |
|---|---|---|---|
| BIPD | 0.00% | 0.17% | **15.3%** |
| WC | 0.00% | 0.09% | **15.9%** |
| Cargo | 0.00% | 0.02% | **25.4%** |
| Surety_bond | 0.00% | 0.00% | **93.0%** |

Score is a usable confidence number, not just an ordinal rank. Broker can filter calls by threshold (e.g. actual carrier has fit score < 0.1 ⇒ strong appetite-outlier signal).

### 7.7 Pool-size caveat (Phase 4H)

Per-state scatter of pool size vs top-5 hit rate:

| Line | corr(pool_size, top-5) | Interpretation |
|---|---|---|
| BIPD | **−0.72** | Strong pool-size effect. CA (pool 84) → 0.91 vs TX (pool 179) → 0.80. |
| WC | −0.30 | Moderate pool-size effect. |
| Cargo | n/a (all states saturate at 1.0) | Documented as pool-size artefact. |
| Surety_bond | n/a (all states saturate at 1.0) | Documented as pool-size artefact. |

Confirms cargo/surety_bond caveat. BIPD still achieves 80% top-5 even in TX (largest pool) → genuine signal, not just pool-size inflation.

### 7.8 Synthetic DOT (Phase 4I)

Picked 5 canonical mainstream DOTs (mid-fleet, no crashes, non-hazmat, genfreight-only, voluntary) across different states. Checked whether model's top-5 includes any specialty / surplus carriers.

- 4/5 clean: top-5 are all mainstream (Progressive, Travelers, Great West, Erie, Sentry, Acuity, etc.)
- 1 yellow flag: a TX DOT returned Trisura and Cimarron (specialty) in top 5 alongside mainstream. Explained by TX's county-mutual market structure, not a general model failure.

### 7.9 Temporal validation (Phase 4J)

Partitioned observed placements by `latest_eff_date` at `vintage − 3 months`. Compared hit rate on recent vs older slices:

| Line | Slice | n | top-5 | top-10 | MRR |
|---|---|---|---|---|---|
| BIPD | recent (<3m) | 111 | 0.757 | 0.856 | 0.608 |
| BIPD | older (≥3m) | 206K | 0.870 | 0.931 | 0.711 |
| WC | recent (<3m) | 11.9K | **0.793** | 0.914 | 0.595 |
| WC | older (≥3m) | 69K | **0.870** | 0.931 | 0.730 |

**WC shows an 8-point top-5 drop on recent placements** (robust sample size). Model performs worse on bleeding-edge WC placements — may indicate temporal drift in carrier appetite or that very-recent placements don't yet have enough history to be well-ranked. Documented as a known limitation.

BIPD's 11-point drop is on only 111 DOTs — too small for a confident claim.

---

## 8. Market analysis (Phase 6 — CA / TX)

Focused analysis of the two largest state markets to demonstrate the derived features in a stakeholder-facing way.

### 8.1 Market structure

| State × Line | Carriers | Total placements | HHI | Top-5 share |
|---|---|---|---|---|
| BIPD / CA | 84 | 53,180 | 0.042 | 37.2% |
| BIPD / TX | 179 | 81,828 | 0.111 | 57.7% |
| WC / TX | 136 | 34,475 | **0.226** | 58.8% |

- BIPD / CA: competitive (HHI < 0.15), fragmented across many carriers including fronting entities (Southlake Specialty, MS Transverse).
- BIPD / TX: still competitive, Progressive + BRK (incl. GEICO) dominant.
- WC / TX: moderately concentrated (HHI > 0.15). Texas Mutual alone writes 47% of state's WC market — a state-authorised near-monopoly structure.

### 8.2 Carrier momentum

Top gainers / losers in each state × line by `carr_state_momentum` (6m vs 6-18m placements):

- **CA BIPD gainers**: Infinity Select, Commodore RRG, Universal Casualty RRG, Spinnaker Specialty — the state is shifting toward risk-retention groups and specialty surplus.
- **CA BIPD losers** (zero placements in last 6m): Imperium, Obsidian Specialty, Transverse Specialty — exiting the market.
- **TX WC gainers**: Stonetrust, Vanliner, Hanover American, PIE Insurance — mainstream carriers growing.
- Tied to `svc_voluntary_exit_ratio` to distinguish healthy growth from churn-driven decline.

### 8.3 Appetite-outlier call sheets

Top-25 DOTs per state × line where actual carrier ranks in the bottom decile of the state's pool. Each row has DOT profile (fleet band, hazmat, crashes, tenure) and model's top-3 recommendations.

Example (BIPD / CA):
```
DOT 859192 | rank 83/84 | actual=OBSIDIAN SPECIALTY INSURANCE COMPANY
           fleet=2 band=B2 hazmat=1 crashes=0 tenure=1.7y
           top-3 recs: GREAT WEST (0.81) | TRAVELERS GRP (0.78) | IAT GRP (0.39)
```

Interpretation: a small-fleet CA DOT with hazmat exposure but no crashes is placed with a specialty surplus insurer. Model identifies three mainstream alternatives with high fit scores — strong candidate for the broker's first call.

---

## 9. Known limitations

### 9.1 Positives label is "voluntary active placement," not "optimal placement"

We measure *carrier appetite* ("which carriers voluntarily insure which types of DOTs") as a proxy for appropriate placement. In many cases these align, but the labelling cannot distinguish:

1. Carrier actively wanted the DOT (true fit)
2. Broker preference / inertia placed them with the carrier
3. DOT took whatever was available in a hard market
4. Price-only decision ignoring underwriting fit

Our output flags **appetite outliers** (DOTs placed with carriers whose typical client profile they don't match) rather than true placement-quality outliers directly. For a production system, retention / renewal / profitability outcomes would be the gold-standard labels; absent those, aggregate market behaviour is the available proxy.

### 9.2 Pool-size inflation on cargo and surety_bond

With 2-5 carriers per state for cargo and 3-10 for surety_bond, top-5 hit rates saturate at ~100% regardless of model quality. Treat these lines as supplementary; primary deliverable value is BIPD and WC.

### 9.3 Peer leakage (bounded)

LOO adjustment removes a DOT's *own* contribution from carrier aggregates, but *peer DOTs* placed with the same carrier are still in the aggregate. This inflates CV ROC-AUC relative to true out-of-sample generalisation. Bounded by `MIN_CARRIER_PLACEMENTS` (5-10% self-correlation per cell).

### 9.4 Temporal drift

WC top-5 is 8 pts lower on recent (<3m) placements vs older. Model may underperform on bleeding-edge placements where carrier appetite is still shifting or where DOT history is thin.

### 9.5 Carrier consolidation inflates absolute metrics

Group-level recommendations ("call Progressive") rather than sub-entity ("call Progressive Direct specifically"). We mitigate with Option B sub-entity distance ranking in Phase 5A, but hit-rate numbers are measured at group level and would be lower at sub-entity level. All baselines are measured on the same consolidated pool, so the model-vs-baseline lift is apples-to-apples.

### 9.6 Known structural quirks accepted

- **GL / non-BMC-84 surety**: dropped from scope due to NULL `event_status`.
- **Texas Mutual** (TX WC state fund): included as a mainstream TX WC carrier because the data provider doesn't flag it as residual-market. Methodologically it IS the Texas residual-market option — treat the model's WC/TX ranking with that context.
- **Fronting carriers** (Southlake Specialty, Highlander Specialty, MS Transverse Specialty, Trisura): these are shells for MGUs/MGAs. The "real" risk-taker is behind them. Can't reliably un-front without an external map.

---

## 10. Deliverable interpretation

### 10.1 `rosella_appetite_outliers_v3.csv` (wide)

Top 10,000 DOTs ranked by `priority`. Columns:

| Column | Meaning |
|---|---|
| `usdot`, `company_name`, `state`, `power_units` | DOT identifier + minimal profile |
| `worst_line` | Which line (bipd/wc/cargo/surety_bond) has the worst-fit current placement |
| `observed_carrier` | Current carrier for `worst_line` |
| `observed_rank_in_line` | Model's rank of the observed carrier within state's pool |
| `pool_size` | How many carriers are in the state's pool for this line |
| `appetite_outlier_score` | `observed_rank / pool_size`. 1.0 = at bottom of pool, 0.0 = at top. |
| `top20_recommended` | Pipe-separated list of top-20 recs, format `"GROUP (best_subentity)"` |
| `priority` | Composite score combining appetite_outlier_score with scale (power_units log) |

Sort by `priority` DESC; the broker starts calling from the top.

### 10.2 `rosella_rankings_v3_long.csv` (long)

Full output for programmatic use. One (DOT, line, state, carrier) per row:

| Column | Meaning |
|---|---|
| `usdot`, `line`, `state`, `rank`, `carrier_group`, `fit_score` | ranking |
| `is_observed` | 1 if this is the DOT's actual carrier on this line |
| `pool_size` | state × line pool size |
| `best_subentity` | Option B closest sub-entity within the carrier group |
| `subentity_distance` | Variance-weighted Euclidean distance from DOT to that sub-entity's centroid |

### 10.3 How a broker uses this

1. Open `rosella_appetite_outliers_v3.csv`.
2. Starting from the top (highest priority):
   - Pull the DOT's profile: fleet, state, line of concern.
   - Note the actual (poorly-ranked) carrier.
   - Pitch the top-3 recommended carriers from `top20_recommended`.
3. For deeper research, cross-reference with the long CSV to see fit scores and sub-entity recommendations for all states / lines this DOT operates in.

---

## 11. Reproducibility

Pipeline is cached end-to-end. Fresh kernel + full warmup takes ~2 min from cached state, ~45 min from scratch.

### 11.1 Cache files (in `cache/`)

| File | Produced by | Load time |
|---|---|---|
| `analysis.duckdb` | cell 5 opens; cells 1E, 2A-2D populate | ~1 s |
| `vintage_and_provider_map.pkl` | cell 1C | ~1 s |
| `dot_df.parquet` | cell 2E | ~15-20 s |
| `models.pkl` (joblib) | cell 3C | ~5 s |
| `scored_obs_rank.pkl` | cell 4A | ~20 s |
| `line_tenure.pkl` | cell 4E | ~1 s |
| `models_weighted.pkl` | cell 4E | ~5 s |
| `scored_obs_rank_weighted.pkl` | cell 4F | ~20 s |

### 11.2 Invalidating caches

- Flip `FORCE_REBUILD = True` in cell 5 to nuke the cache dir on next run.
- To invalidate only consolidation-dependent tables (after CARRIER_RULES changes):
  ```python
  for tbl in ['carrier_lookup','recent_events','service_composite',
              'carrier_state_agg','dot_cell_contrib',
              'sub_entity_centroids','dot_subent_contrib']:
      con.execute(f"DROP TABLE IF EXISTS {tbl}")
  ```
  Then re-run cells 1C, 2A-2D. `dot_features` (cell 1E) and `dot_df` (cell 2E) stay cached.

### 11.3 Switching models for the deliverable

Set `MODEL_FOR_DELIVERABLE` in cell 1A to `'base'` or `'weighted'`. Re-run cell 1A, then cells 5A and 5B. **Current recommendation**: `'base'` (wins empirically; see §7.2).

---

## 12. Phase / cell map

| Cell | Phase | Purpose |
|---|---|---|
| 1-5 | — | Imports, DuckDB connect, cache helpers, raw CSV view |
| 6-19 | — | Earlier EDA / discovery (not required for v3) |
| 20 (md), 21-24 | **Phase 1** | Setup, carrier consolidation rules, data vintage, dot_features table (1A-1C, 1E) |
| 25 (md), 26-30 | **Phase 2** | Aggregates, centroids, scorable-DOTs, derived features (2A-2E) |
| 31 (md), 32-33 | **Phase 3** | **EDA & Data Quality** — sanity check (1D), placement count diagnostic (2F) |
| 34 (md), 35-37 | **Phase 4** | Feature definitions + training helpers, placement extraction, LightGBM training |
| 38 (md), 39-49 | **Phase 5** | Scoring, baselines, feature importance, error analysis, known-good/known-bad, weighted-tier retrain, weighted comparison, calibration, pool-size, synthetic DOT, approximate temporal, true temporal train/test split |
| 50 (md), 51-52 | **Phase 6** | Combine, Option B sub-entity, deliverable CSVs |
| 53 (md), 54-56 | **Phase 7** | CA / TX market structure, carrier momentum, appetite-outlier call sheets |
| 57 (md), 58-59 | **Phase 8** | Multi-line bundling analysis, recommendation-hygiene post-processing (hygiene CSVs) |

---

## 13. Future improvements

If this were a production system (not a take-home):

1. **Learning-to-rank (LambdaRank)** instead of binary classification — directly optimises NDCG@K.
2. **Temporal train/test split** — hold out most-recent 3m of placements as true out-of-sample test; current temporal check is approximate.
3. **Market-segmented models** — separate models for standard-risk vs non-standard-risk DOTs with routing on DOT features. Would improve the Texas-county-mutual / specialty-market segment specifically.
4. **Retention outcomes as labels** — if available (renewal/churn/cancellation data over time), supervised signal for "placement quality" rather than relying on the appetite proxy.
5. **Fronting carrier unwinding** — map Southlake / Highlander / MS Transverse / Trisura shells to their actual risk-takers via a fronting-relationship table (would need external data).
6. **Dynamic per-DOT scoring context** — instead of one static ranking, serve different top-K depending on broker intent (new placement vs renewal vs re-shop).
