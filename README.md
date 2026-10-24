# Rosella Carrier-Appetite Call List — Data Science Report

> **Task**: Generate an ordered call list of trucking companies, prioritised by how appropriately each is placed with their current insurance carrier.

**Deliverables** (produced by `rosella investigation.ipynb`):

| File | Shape | Purpose |
|---|---|---|
| `rosella_appetite_outliers.csv` | 10K rows × ~26 cols | **The call list.** Wide — top-20 recommended carriers per DOT per line, sorted by appetite-outlier priority. |
| `rosella_rankings_long.csv` | 23M rows × 9 cols | Long-format rankings — one `(DOT, line, state, carrier)` per row with fit score, rank, pool size, best-fit sub-entity. |
| `*_hygiene.csv` | 10K / 23M | Same as above after recommendation-hygiene post-processing (exiting-carrier filter, previously-rejected filter, bundling bonus). |
| `carrier_consolidation_report.csv` | ~2.1K rows | Raw-insurer → carrier-group map with per-mapping provenance (provider map / rule tier / raw). |
| `rule_vs_provider_disagreements.csv` | ~10 rows | Residual label-disagreement log after reconciliation. |

---

## 1. Executive Summary

- **Approach.** Binary contrastive LightGBM, one model per insurance line. For each DOT, rank the carriers in their state's eligible pool by predicted "appetite fit." A DOT whose actual carrier ranks poorly is an *appetite outlier* — a high-priority call candidate for the broker.
- **Lines modelled (4).** `bipd` (auto liability), `wc` (workers comp), `cargo`, `surety_bond` (BMC-84 broker bond). `gl` and `surety` dropped — raw `event_status` systematically NULL, no clean way to isolate current voluntary placements.
- **Features (170+).** 133 DOT numerics + 41 DOT categoricals + 30 carrier aggregates (per `(carrier, state, line)` cell). Full inventory in §5.3.
- **Carrier consolidation.** Hybrid pipeline — data provider's `insurer_group` (2.7% coverage, high confidence) → 78 rule-based regexes (tiered T1-T4) → raw fallback. ~2,100 distinct insurer strings reduce to ~1,400 carrier groups.
- **Headline results** (model vs popularity-by-fleet-band baseline, top-5 hit rate):

  | Line | Model top-5 | Baseline (popularity) | Lift | n |
  |---|---|---|---|---|
  | **BIPD** | 87.8% | 68.7% | **+19.1 pts** | 320,329 |
  | **WC** | 85.3% | 51.8% | **+33.5 pts** | 83,910 |
  | cargo | 100% | 68.1% | *saturated — pool-size artefact* | 3,304 |
  | surety_bond | 100% | 87.9% | *saturated — pool-size artefact* | 10,451 |

- **Validation passes.** 10 independent checks (feature importance, error analysis, known-good/known-bad audit, calibration, pool-size scatter, synthetic-DOT sanity, approximate temporal, true temporal train/test split, weighted-tier sensitivity, hygiene filters). All pass or produce documented caveats.
- **Model used in deliverable.** Base model with clean positives + hard/soft negatives — beat the weighted-tier alternative on every slice including the long-tenure proxy for "appropriately placed."

---

## 2. Problem Introduction

### 2.1 Business context

Rosella is an insurance broker for trucking risks. A broker's edge is identifying DOTs placed with a carrier that *shouldn't* be writing their risk — those DOTs are opportunities to pitch a better fit and win the account.

The question posed by the brief: given the full dataset of 4.5M+ DOTs and their insurance history, which ones should the broker call *first*? The ordering rule is "how appropriately placed is this DOT with its current carrier?"

### 2.2 Why this is a ranking problem, not a classification problem

The task has **no direct label** for "appropriately placed." Retention / churn / profitability data would be the ground truth; absent that, we have to infer appropriateness from aggregate market behaviour — *which carriers voluntarily insure which types of DOTs*. This translates naturally into a **contrastive ranking** formulation:

- For each DOT in a given line, rank the carriers in their state's pool.
- The DOT's actual carrier sits somewhere in that ranking.
  - High rank → placement looks appropriate (model agrees with the market).
  - Low rank → *appetite outlier* → call candidate.

This framing side-steps the missing label: we never predict "is this placement good?" directly — we predict "would carrier C willingly take DOT D?" and use the actual placement's rank as the outlier signal.

### 2.3 What "appropriate" means in this context (and what it doesn't)

The positive label is "voluntary active placement" — specifically, `is_residual_market = 0 AND is_cancellation_pending = 0`. This proxies carrier appetite but **conflates four regimes** that we can't disentangle from the data:

1. Carrier actively wanted the DOT (true fit).
2. Broker inertia placed them with the carrier.
3. DOT took whatever was available in a hard market.
4. Price-only decision ignoring underwriting fit.

The output is therefore honestly framed as **appetite outliers** — DOTs placed with carriers whose typical client profile they don't match — rather than placement-quality outliers directly. §9.1 covers this limitation explicitly.

---

## 3. Data Overview

- **Source**: `usdot_insurance_events_2026-10-12.csv` — 43 GB, 12.24M events, 370 columns.
- **Grain**: one row per insurance event over DOT history. Each event has `policy_line`, `event_status`, `effective_date`, `cancellation_or_end_date`, `is_residual_market`, carrier identity, and the DOT's census / safety / authority joined columns.
- **Snapshot columns**: "as-of 2026-05-26" active-placement columns (`active_bipd_raw`, `active_cargo_raw`, `active_wc_raw`) give the authoritative current BIPD/cargo/WC carriers per DOT. For `surety_bond`, placements are reconstructed from events with `event_status IN ('active', 'active (cancellation pending)')`.
- **Data vintage**: `MAX(effective_date)` sanitised against outliers (raw contains year-9001 entries). All 24-month windows are *vintage-relative* so re-runs are deterministic even as calendar time passes.

### 3.1 Scorable DOT universe

~462K DOTs with at least one active placement across the 6 insurance lines in scope. This is the scoring pool — the complement are DOTs with no active placement in the window, which can't be ranked against the market.

### 3.2 Line coverage (feeds EDA §4.1)

Observed via `Part 2F`:

| Line | DOTs with active | Model-eligible? | Notes |
|---|---|---|---|
| BIPD | ~388K | Yes | Pre-computed snapshot. Primary line. |
| WC | ~70K | Yes | Pre-computed snapshot. |
| Cargo | ~35K | Yes | Pre-computed snapshot. Thin per-state pools. |
| Surety_bond | ~12K | Yes | Event-reconstructed from BMC-84 broker bond status. |
| GL | ~7.7K | **No** | `event_status` NULL — can't isolate active placements. |
| Surety | ~3.3K | **No** | `event_status` NULL — same problem. |

---

## 4. Exploratory Data Analysis

The notebook's **Phase 3** cells (`Part 1D` sanity check, `Part 2F` placement counts) and the earlier discovery cells (6-19) surface the findings below. Those cells are retained but not required for scoring; the final model only consumes `Phase 1-2` artefacts.

### 4.1 GL and non-BMC-84 surety had unusable event-status

`Part 2F` dumped the distribution of `event_status` per line. For `gl` and `surety` (non-BMC-84), `event_status` was NULL on **100%** of events. With no status, "latest event per DOT" cannot distinguish current voluntary placements from historical/cancelled ones. Any imputation would silently mislabel ex-customers as positives. Both lines dropped from scope. `surety_bond` (BMC-84 broker bond) has clean status and is retained.

### 4.2 Carrier identity is messy — ~2,100 raw strings for a market with ~1,400 real entities

Direct count: `SELECT COUNT(DISTINCT insurer_raw) FROM events` returned ~2,100. But the data-provider map (`insurer_group`, populated on 2.7% of events) ties many of those strings together — `PROGRESSIVE DIRECT INS CO`, `PROGRESSIVE AMERICAN INS CO`, `PROGRESSIVE PREFERRED INS CO` all map to `PROGRESSIVE GRP`.

The 2.7% coverage is insufficient to use the provider map alone — it leaves 97% of events with a raw string. We built a hybrid pipeline (§5.2) to cover the remaining 97%.

### 4.3 Dates in the raw data are hostile

Discovery cells found:
- `9001-01-01` and `9999-12-31` sentinel entries scattered across date columns.
- Future `crash_latest_date` entries (dated 2028+) that should be impossible for a past-event column.
- Legitimate future-dated expirations on `current_bipd_expiration_date` (policies run 1-2 years ahead).

These demanded separate handling for past-only vs future-allowed columns — see §5.1.

### 4.4 Numeric outliers consistent with raw-CSV data-entry errors

Discovery found `power_units` values in the millions, `years_in_business > 500`. These are clearly impossible; the data-cleaning layer imposes plausible bounds (§5.1).

### 4.5 Snapshot vs event-reconstruction mismatch

For BIPD/cargo/WC the snapshot columns (as-of 2026-05-26) are the source of truth for *current* placement. Event reconstruction gave a different carrier in ~5% of cases — usually because an event dated after 2026-05-26 was in the raw events but hadn't been reflected in the snapshot yet, or because of residual-market event classification differences.

Resolution: **snapshot wins** for the three lines that have one. `surety_bond` has no snapshot column, so event reconstruction is unavoidable (and clean because BMC-84 has clean `event_status`).

### 4.6 Residual market is a 7% volume signal with outsized downstream impact

~7% of placements are flagged `is_residual_market = 1`. These are state-assigned, non-voluntary placements — the carrier didn't choose the DOT. Treating them as positives would teach the model that "a hazmat trucker in New Jersey is a good fit for the state's assigned-risk pool" — the opposite of appetite. They become the clearest *hard-negative* signal (§5.4).

### 4.7 Carrier state-footprint is dynamic

Over 24 months some carriers enter new states and others exit — e.g. CA BIPD saw Obsidian Specialty, Imperium, and Transverse Specialty go to zero recent placements. If we recommend a carrier that has stopped writing in a state, the recommendation is useless. The momentum features (§5.3.3) and the exiting-carrier hygiene filter (§10.4) address this.

### 4.8 Market concentration varies dramatically by state × line

`Part 6A` market structure analysis:

| State × Line | Carriers | HHI | Top-5 share |
|---|---|---|---|
| BIPD / CA | 84 | 0.042 (competitive) | 37.2% |
| BIPD / TX | 179 | 0.111 (competitive) | 57.7% |
| WC / TX | 136 | **0.226 (concentrated)** | 58.8% |

WC/TX concentration is driven by **Texas Mutual writing 47% alone** — a state-authorised near-monopoly. This has implications for the "specialty vs mainstream" audit (§7.3).

---

## 5. Methodology

### 5.1 Data cleaning

| Step | Action | Rationale |
|---|---|---|
| Dates (past-only) | Null to `NaT` if `< 1990-01-01` or `> today` | 9001-year entries and 2028-dated past crashes are data errors. Nulling (not clipping) propagates to downstream `days_since_*` as `NaN`, which LightGBM handles natively — honest "unknown" treatment vs false-precision clip. |
| Dates (future-allowed) | Null to `NaT` if `< 1990-01-01` or `> 2100-01-01` | Policies legitimately expire 1-2 years ahead; multi-decade surety bonds exist. 2100 is a loose "obvious data error" ceiling that catches 9999 entries. |
| Numerics (bounded) | Null if outside `[0, bound]` for each column | `power_units > 100k`, `years_in_business > 150`, etc. are impossible. LightGBM handles `NaN`, DuckDB aggregates skip `NULL`, so LOO aggregates stay clean. |
| USDOT dtype | Enforced `int64` end-to-end | Pandas int32/Int64 nullable round-trips through parquet caused merge-time dtype-mismatch errors — serial resolution was `.astype('int64')` at every ingress point. |
| State codes | Uppercase + trim via DuckDB `UPPER(TRIM(...))` | Raw has mixed-case and whitespace variants. Required for the join across `recent_events`, `carrier_state_agg`, `dot_df`. |
| Insurer strings | Uppercase + trim before consolidation lookup | Same reason — ensures regex rules match. |

**Nulls are preserved, not imputed.** Imputing (e.g. median-fill `crash_count`) would inject structural bias — a DOT with no record and a DOT with zero events are different and the model needs to distinguish them. LightGBM learns a split on `is_null` natively; HGBT falls back to the same behaviour.

#### 5.1.1 Why a 24-month window on `recent_events`?

All carrier aggregates (`carr_mean_power`, `carr_pct_hazmat`, momentum, …) and service composite features are built from `recent_events` — events with `effective_date ≥ vintage − 24 months`. The window length is a load-bearing choice: too short loses signal, too long leaks stale appetite patterns.

- **Why not use the full event history (10+ years)?** Carrier appetite shifts. A carrier that wrote heavy hazmat in 2015 may not today; feeding 2015 events into `carr_pct_hazmat` would mislabel current appetite. The point of the aggregates is "current typical client," not "lifetime typical client."
- **Why not shorter (12 months)?** BIPD renewal cycles are annual — one renewal cycle of signal is barely enough for a carrier that cycles slowly. Cargo and surety_bond carriers have sparser activity; 12 months would drop many carriers below `MIN_CARRIER_PLACEMENTS` and shrink the scoring pool.
- **Why not longer (36-60 months)?** Includes carriers that have since exited the state × line (dynamic footprint, §4.7). The momentum features (`carr_placements_6m` vs `carr_placements_6to18m`) partially correct for this at the recent end, but the mid-window stays stale.
- **24 months as the compromise.** Two renewal cycles for annual lines; enough events to stabilise aggregates for all four modelled lines. The momentum features let the model distinguish "carrier wrote 100 placements but 90 were 20 months ago" from "carrier wrote 100 and 50 are from last 6 months."

**Window is vintage-anchored, not today-anchored.** `_DATA_VINTAGE = MAX(effective_date)` after outlier sanitisation. All downstream `days_since_*`, momentum windows, and temporal cutoffs use the vintage, not the wall-clock `today`. This makes re-runs deterministic (same inputs → same outputs regardless of when the notebook is executed) and makes the model behave as if run on the data vintage date.

### 5.2 Carrier consolidation

Three-tier `consolidate_one(raw_insurer_name)`:

1. **Data-provider map** (highest confidence, authoritative). The provider's own `insurer_group` field, available on 2.7% of events. We extract the `raw → group` map from populated rows and look up any `raw` seen in that map. ~1,000 raw-string entries.
2. **Rule-based regexes**, 78 patterns organised by confidence tier (T1 = brand-exact e.g. `^STATE FARM`, T4 = generic family e.g. `COUNTY MUTUAL`). Covers acquisitions (`GEICO *` → `BERKSHIRE HATHAWAY GRP`), brand variants (`PROGRESSIVE *` → `PROGRESSIVE GRP`), and state subsidiaries. Tiered so a T1 match wins over a T4 match on ambiguous strings.
3. **Raw fallback**. If neither matches, the raw insurer string is its own group — avoids collapsing a genuinely new insurer into a bucket by accident.

**Rationale for provider-first ordering**: the provider sees the most recent ownership structure; our rules are hand-written and inevitably trail. Where provider and rule disagree, provider wins. `rule_vs_provider_disagreements.csv` logs residual mismatches after we re-aligned the rules to provider labels — <10 rows after reconciliation.

### 5.3 Feature engineering

Three feature groups: DOT-level numeric (133), DOT-level categorical (41), and carrier-level aggregates (30). Each listed exhaustively below with definition, transformation, and rationale.

#### 5.3.1 DOT numeric features (133)

| Feature | Definition / Source | Transformation | Rationale |
|---|---|---|---|
| **Fleet shape** | | | |
| `power_units` | Count of registered power units (tractors + straight trucks) | raw | Primary scale indicator — larger fleets have different carrier markets than owner-operators. |
| `power_units_log` | Same as above | `log1p(power_units)` | Fleet size is heavy-tailed (p99 ≫ p50; some fleets at 1000+). Log compresses mega-fleet leverage. **Both forms kept** so trees pick whichever splits better — raw gives exact counts, log gives scale-aware splits. Other heavy-tailed features (`mcs150_mileage`) are not logged because they enter only as predictors and the +1-smoothed ratio features (`crash_rate_per_pu` etc.) already compensate for scale. |
| `total_drivers` | Total driver count (CDL + non-CDL) | raw | Crew intensity, correlates with but distinct from PU. |
| `total_cdl` | CDL drivers only | raw | Interstate/heavy-truck proxy. |
| `truck_units` | Trucks (subset of PU) | raw | Separates trucks from tractors. |
| `mcs150_mileage` | Self-reported annual mileage | raw | Exposure measure. Highly skewed but no bounding — reported in miles. |
| `bus_units` | Bus units | raw | Separates passenger operations (different carrier appetite). |
| **Driver composition** | | | |
| `total_intrastate_drivers` | Intrastate-only drivers | raw | Operational scope. |
| `driver_inter_total` | Interstate-driver count | raw | Operational scope. |
| **Radius mix** | | | |
| `inter_beyond_100` | Interstate drivers beyond 100mi | raw | Long-haul signal — specialty carriers underwrite this differently. |
| `inter_within_100` | Interstate drivers within 100mi | raw | Short-haul interstate. |
| `intra_beyond_100` | Intrastate drivers beyond 100mi | raw | Intrastate long-haul. |
| `intra_within_100` | Intrastate drivers within 100mi | raw | Local operations — different carrier appetite. |
| **Crash record** | | | |
| `crash_count` | Lifetime crashes reported to FMCSA | raw | Primary risk signal. |
| `crash_count_24m` | Crashes in last 24 months | raw | Recent risk — more actionable than lifetime. |
| `crash_fatalities` | Lifetime fatal crashes | raw | Severity. Very rare but very informative. |
| `crash_injuries` | Lifetime injury crashes | raw | Severity. |
| `crash_tow_away_count` | Lifetime tow-away crashes | raw | Severity. |
| `crash_rate_per_mil` | Source-provided mileage-normalised rate | raw | Normalised crash exposure. |
| **Inspection record** | | | |
| `inspection_count` | Lifetime inspections | raw | Enforcement exposure. |
| `inspection_violations` | Violations found at inspections | raw | Violation volume. |
| `inspection_oos_viol` | Out-of-service violations | raw | Serious compliance failures. |
| `inspections_with_oos` | Count of inspections with ≥1 OOS | raw | OOS incidence rate. |
| `inspection_driver_oos` | Driver-caused OOS | raw | Attribution to driver behaviour. |
| `inspection_vehicle_oos` | Vehicle-caused OOS | raw | Attribution to equipment. |
| `inspection_hazmat_oos` | Hazmat-specific OOS | raw | Hazmat compliance. |
| **Enforcement actions** | | | |
| `oos_order_count` | DOT-level OOS orders | raw | Operation-shutting enforcement — rare, serious. |
| `revocation_count` | Authority revocations | raw | Nuclear enforcement. |
| `prior_revoke_flag` | Any prior revocation (binary) | raw | Binary form for coarse separation. |
| **SMS BASIC scores** | | | |
| `sms_insp_total` | Total SMS inspections | raw | Enforcement window volume. |
| `sms_driver_insp_total` / `sms_vehicle_insp_total` | Driver / vehicle SMS inspections | raw | Attribution. |
| `sms_driver_oos_insp_total` / `sms_vehicle_oos_insp_total` | OOS variants | raw | Attribution of OOS findings. |
| `sms_violation_count` / `sms_violation_oos_count` / `sms_violation_severity` | SMS violation metrics | raw | Severity-weighted violation history. |
| `sms_unsafe_driv` / `sms_hos_driv` / `sms_driv_fit` / `sms_veh_maint` / `sms_contr_subst` | FMCSA BASIC category percentile (0-100) | raw | Standard FMCSA safety indicator per category. |
| `sms_unsafe_driv_viol` / `sms_hos_driv_viol` / `sms_driv_fit_viol` / `sms_contr_subst_viol` / `sms_veh_maint_viol` | Violation counts per BASIC category | raw | Underlying violation volume behind each BASIC. |
| `has_sms` | Any SMS data present (binary) | raw | Data-presence flag — distinguishes "no record" from "clean record." |
| **Authority** | | | |
| `has_authority` | Any authority registration (binary) | raw | Scope indicator. |
| `authority_history_count` | Count of historical authority filings | raw | Authority churn — stable operators vs serial-register. |
| **Business profile** | | | |
| `years_in_business` | Years since authority first served | raw | Age proxy — younger operators have thinner history and different carrier markets. |
| `co_usdot_count` | USDOTs linked to same company entity | raw | Fleet-of-fleets indicator. |
| `co_phone_count` / `co_officer_count` / `co_source_count` / `co_resolved_company_count` | Business-registry breadth | raw | Information density — richer records correlate with established operators. |
| `co_has_retirement_plan` / `co_has_health_plan` | Benefits binary | raw | Operator maturity proxy. |
| `has_domain` | Owns a web domain (binary) | raw | Formality proxy. |
| `is_peo_flag` | Operates under PEO structure | raw | PEOs have special WC market dynamics. |
| `has_mcsip` | Enrolled in MCSIP (motor-carrier safety improvement) | raw | Flagged problem operator. |
| **Insurance history** | | | |
| `ins_event_count` | Lifetime insurance events on file | raw | Total insurance activity. |
| `ins_distinct_insurers` | Distinct carriers seen in DOT history | raw | Carrier-switching intensity. |
| `ins_cancelled_count` / `ins_replaced_count` / `ins_rejected_count` | Event-type counts | raw | Cancellation / replacement / rejection history — distinct carrier-side signals. |
| `ins_years_coverage` | Years of insurance history | raw | Tenure for normalisation. |
| **Commodity flags (30)** | `crgo_genfreight`, `crgo_bldgmat`, `crgo_construct`, `crgo_beverages`, `crgo_paperprod`, `crgo_produce`, `crgo_meat`, `crgo_drybulk`, `crgo_coldfood`, `crgo_intermodal`, `crgo_grainfeed`, `crgo_livestock`, `crgo_motoveh`, `crgo_household`, `crgo_passengers`, `crgo_oilfield`, `crgo_chem`, `crgo_logpole`, `crgo_cargoothr`, `crgo_machlrg`, `crgo_mobilehome`, `crgo_coalcoke`, `crgo_garbage`, `crgo_metalsheet`, `crgo_drivetow`, `crgo_waterwell`, `crgo_usmail`, `crgo_utility`, `crgo_farmsupp`, `crgo_liqgas` | binary from self-report | Commodity niche is a strong appetite driver. Hazmat/livestock/oilfield are specialty markets; genfreight is mainstream. |
| **Coverage / price proxies** | | | |
| `coverage_amount_usd` | Current BIPD coverage amount | raw | Over-insurance signal (if ≫ authority minimum). |
| `underlying_limit_usd` | Primary layer limit when policy is layered | raw | Umbrella/excess structure. |
| `min_required_coverage_usd` | Minimum required by authority type | raw | Baseline for the ratio feature. |
| `authority_min_cov_amount` | Authority-type minimum coverage | raw | Alternative baseline. |
| `coverage_vs_minimum_ratio` | `coverage_amount_usd / authority_min_cov_amount` | ratio | Captures over-insurance — DOTs with 10× minimum signal different market segment than DOTs at minimum. +1 denominator avoids div-by-zero. |
| `has_layered_coverage` | `underlying_limit_usd.notna()` | binary | Umbrella/excess structure is itself a market segment. |
| `rejected_filing_count` | Authority-filing rejections | raw | Compliance-friction signal. |
| `wc_rate_multiplier` | WC rate multiplier from source | raw | Direct price signal for WC. |
| **Timing / recency** (all anchored at `_DATA_VINTAGE`, not `today`, for determinism) | | | |
| `days_since_bipd_effective` | `vintage − current_bipd_effective_date` days | subtract-from-vintage | BIPD tenure. Primary appetite signal — long tenure means retention. |
| `days_until_bipd_expiration` | `current_bipd_expiration_date − vintage` days | subtract-from-vintage | Renewal urgency. |
| `years_with_current_carrier` | `days_since_bipd_effective / 365.25` | scale conversion | Interpretable tenure unit. Kept alongside the days form — trees pick whichever splits better. |
| `days_since_last_crash` | `vintage − crash_latest_date` days | subtract-from-vintage | Recency decay — recent crashes are more informative than lifetime count. |
| `days_since_last_inspection` | same pattern | subtract | Enforcement recency. |
| `days_since_last_oos` | same pattern | subtract | OOS recency. |
| `days_since_last_revocation` | same pattern | subtract | Revocation recency. |
| `days_since_insurance_first` | same pattern | subtract | Insurance tenure. |
| `days_since_insurance_latest` | same pattern | subtract | Insurance recency. |
| `days_since_authority_served` | same pattern | subtract | Business age. |
| `days_since_mcs150_update` | same pattern | subtract | Census freshness — stale MCS-150 means fleet counts may be out of date. |
| `has_insurance_event` | Any insurance event present (binary) | raw | Data-presence flag. |
| **Ratio features (12)** | all use `+1` denominator smoothing | | |
| `crash_rate_per_pu` | `crash_count / (power_units + 1)` | rate normalisation | Fleet-size-normalised crash exposure. +1 avoids div-by-zero and shrinks tiny fleets toward zero. |
| `oos_rate_per_pu` | `oos_order_count / (power_units + 1)` | rate normalisation | Fleet-size-normalised OOS. |
| `violation_rate` | `inspection_violations / (inspection_count + 1)` | inspection-normalised | Violations-per-inspection. +1 smoothing dampens zero-inspection outliers. |
| `oos_violation_rate` | `inspection_oos_viol / (inspection_count + 1)` | inspection-normalised | OOS-violations-per-inspection. |
| `driver_oos_rate` | `inspection_driver_oos / (inspection_count + 1)` | inspection-normalised | Driver-attribution rate. |
| `vehicle_oos_rate` | `inspection_vehicle_oos / (inspection_count + 1)` | inspection-normalised | Vehicle-attribution rate. |
| `hazmat_oos_rate` | `inspection_hazmat_oos / (inspection_count + 1)` | inspection-normalised | Hazmat-attribution rate. |
| `insurance_cancel_rate` | `ins_cancelled_count / (ins_event_count + 1)` | event-normalised | Cancellation propensity. |
| `insurance_rejection_rate` | `ins_rejected_count / (ins_event_count + 1)` | event-normalised | Rejection propensity — large signal for appetite outlier audit. |
| `carrier_switch_rate` | `ins_distinct_insurers / (ins_years_coverage + 1)` | tenure-normalised | Churn intensity. |
| `driver_per_pu` | `total_drivers / (power_units + 1)` | rate | Crew intensity ratio. |
| `insurance_history_years` | `ins_last_year − ins_first_year`, clipped to ≥ 0 | subtract + clip | History span. Clip handles edge cases where `ins_first_year > ins_last_year` in raw data. |
| **Rare-event flags (4)** | | | |
| `had_any_fatal_crash` | `crash_fatalities > 0` | binary indicator | Fatal crashes are rare (<2% of fleet) but strongly informative. Binary form prevents a 1-crash DOT from being swamped by lifetime-count features. |
| `had_any_revocation` | `revocation_count > 0` | binary indicator | Same reasoning — revocations are rare, informative, and binarising prevents dilution. |
| `had_any_oos_order` | `oos_order_count > 0` | binary indicator | Same reasoning. |
| `had_any_rejection` | `ins_rejected_count > 0` | binary indicator | Same reasoning. |

**Note on `is_residual_market` and `is_cancellation_pending`**: these columns exist in the data but are **not** features. They define positive/negative labels (§5.4) — using them as features would be direct label leakage.

#### 5.3.2 DOT categorical features (41)

| Feature | Definition / Source | Transformation | Rationale |
|---|---|---|---|
| `state` | DOT's primary state | label-encoded via `CAT_MAPS` (fit once on all scorable DOTs) | Primary geographic indicator — carrier markets are state-partitioned. LightGBM `categorical_feature` support used end-to-end (not one-hot). |
| `phy_state` | Physical operations state | label-encoded | Operations vs registration can differ (e.g. DE-incorporated TX operator). |
| `authority_bus_state_code` | Authority business state | label-encoded | Yet another state field — carriers appear differently in each view. |
| `naics_sector` | `primary_naics[:3]` | 3-digit truncation | Sector-level industry. Trees can't exploit 6-digit NAICS directly without overfitting; 3-digit gives useful granularity. |
| `naics_subsector` | `primary_naics[:4]` | 4-digit truncation | Finer than sector; still generalisable. |
| `co_lifecycle_status` | Entity lifecycle | label-encoded | Dormant / active / dissolved — relevant for carrier willingness. |
| `org_type` | LLC / Corp / Sole prop / etc. | label-encoded | Legal structure — WC carriers in particular treat sole props differently. |
| `fleetsize_band` | Source-provided fleet band | label-encoded | Alternative to custom `fleet_band`. |
| `fleet_band` | Custom bin: ≤1=B1, ≤5=B2, ≤20=B3, ≤100=B4, else B5 | custom binning on `power_units` | Finer-grained bands tuned for trucking carrier breakpoints. Kept alongside source band — LightGBM picks whichever splits better. |
| `carrier_operation` | Operation type | label-encoded | For-hire / private / passenger / etc. — different carrier markets. |
| `classdef` | Carrier classification | label-encoded | FMCSA classification. |
| `hazmat_ind` | Hazmat permission (`Y`/`X`/`N`) | label-encoded | Hazmat appetite is a strong binary split. |
| `carship` | Carrier/shipper designation | label-encoded | Operation-type signal. |
| `census_status_code` | DOT census status | label-encoded | Active / inactive-pending — data-quality dimension. |
| `safety_rating` | SATISFACTORY / CONDITIONAL / UNSATISFACTORY | label-encoded | Direct safety indicator — strong carrier filter. |
| `authority_common_stat` / `authority_contract_stat` / `authority_broker_stat` | Authority status per type | label-encoded (3) | Granted / revoked / pending by authority type. |
| `authority_bipd_file` / `authority_cargo_file` / `authority_bond_file` | Filing flags | label-encoded (3) | Filed vs not — regulatory baseline. |
| `authority_property_chk` / `authority_passenger_chk` / `authority_hhg_chk` / `authority_private_auth_chk` / `authority_enterprise_chk` | Authority scope checks | label-encoded (5) | What the DOT is authorised to haul. |
| `authority_cargo_req` / `authority_bond_req` | Insurance requirements | label-encoded (2) | Whether cargo / bond insurance is required for the DOT's authority types. |
| `authority_common_app_pend` / `authority_contract_app_pend` / `authority_broker_app_pend` | Pending applications | label-encoded (3) | Authority in flux. |
| `authority_common_rev_pend` / `authority_contract_rev_pend` / `authority_broker_rev_pend` | Pending revocations | label-encoded (3) | Authority at risk — carriers avoid. |
| `authority_undeliverable_mail` | Mail-return flag | label-encoded | Address/business-reality signal. |
| `sms_unsafe_driv_ac` / `sms_hos_driv_ac` / `sms_driv_fit_ac` / `sms_contr_subst_ac` / `sms_veh_maint_ac` | SMS BASIC alert category | label-encoded (5) | FMCSA's categorical "needs intervention" per BASIC. Qualitative version of the numeric `sms_*` features. |
| `co_match_confidence` | Business-registry match confidence | label-encoded | Data-quality indicator — low confidence means business records may be stale/wrong. |

Encoding details: fit once on all scorable DOTs (`CAT_MAPS`), nulls become `'UNK'` → their own encoded value. Unseen values at inference time map to `-1` (LightGBM treats this as a separate categorical level).

#### 5.3.3 Carrier aggregates (30, per `(carrier_group, state, line)` cell)

Only cells with `≥ MIN_CARRIER_PLACEMENTS` are included (BIPD/cargo/WC = 20, surety_bond = 15). This filters sparse-presence carriers.

| Feature | Definition / Source | Transformation | Rationale |
|---|---|---|---|
| **Volume** | | | |
| `carr_n_placements` | `COUNT(DISTINCT usdot)` in cell | aggregation + LOO | Carrier's volume signal in this state × line. LOO-adjusted (subtract 1 if the scored DOT is actually in this cell). |
| `carr_n_events` | `COUNT(*)` events in cell | aggregation | Event count (vs DOT count). |
| `carr_pct_residual_market` | `AVG(is_residual_market_event)` | aggregation | Share of cell events that are residual-market. **Dual use** — carriers with ≥ 0.5 are filtered from the scoring pool (§5.6); for the rest, it's a feature flagging appetite skew. |
| **Portfolio profile** (central tendency over DOTs in cell) | | | |
| `carr_mean_power` | `AVG(power_units)` | aggregation + **LOO** | Carrier's typical fleet size. LOO subtracts the scored DOT's contribution to avoid self-leakage. |
| `carr_median_power` | `MEDIAN(power_units)` | aggregation (no LOO) | Robust central tendency. Median can't be cheaply LOO-adjusted — residual leakage bounded by `MIN_CARRIER_PLACEMENTS`. |
| `carr_p90_power` | `QUANTILE(0.9)` | aggregation (no LOO) | Captures mega-fleet appetite — some carriers write a long right tail. |
| `carr_mean_crash` | `AVG(crash_count)` | aggregation + LOO | Carrier's crash tolerance. |
| `carr_mean_insp_oos` | `AVG(inspection_oos_viol)` | aggregation + LOO | Carrier's OOS tolerance. |
| `carr_mean_revoc` | `AVG(revocation_count)` | aggregation + LOO | Carrier's written-off-risk tolerance. |
| `carr_mean_oos` | `AVG(oos_order_count)` | aggregation + LOO | Carrier's DOT-OOS tolerance. |
| `carr_pct_hazmat` | `AVG(hazmat_bin)` | aggregation + LOO | Hazmat share of carrier's book. |
| **Commodity mix** (10, LOO-adjusted) | | | |
| `carr_pct_genfreight` / `carr_pct_bldgmat` / `carr_pct_construct` / `carr_pct_livestock` / `carr_pct_oilfield` / `carr_pct_passengers` / `carr_pct_chem` / `carr_pct_meat` / `carr_pct_intermodal` / `carr_pct_motoveh` | `AVG(crgo_<x>)` | aggregation + LOO | Composition of carrier's book by commodity. 10 of the 30 commodity classes — the ones that showed meaningful variation across carriers in EDA. |
| **Radius mix** | | | |
| `carr_mean_inter_long` | `AVG(inter_beyond_100)` | aggregation + LOO | Long-haul interstate appetite. |
| `carr_mean_intra_short` | `AVG(intra_within_100)` | aggregation + LOO | Local operations appetite. |
| **Service composite** (per carrier, not per cell — reuses carrier identity) | | | |
| `svc_short_tenure_cancel_rate` | `AVG(days_in_force < 90 AND status = 'cancelled')` over all carrier events | carrier-level aggregation | Early-cancellation rate — proxy for underwriting rigour / service quality. High = carrier bounces lots of DOTs quickly. |
| `svc_rejection_rate` | `AVG(status = 'rejected')` | carrier-level aggregation | Rejection frequency — strong appetite signal. |
| `svc_voluntary_exit_ratio` | `AVG(status = 'replaced')` over `IN ('cancelled', 'replaced')` | carrier-level ratio | Distinguishes "DOT voluntarily left" (replaced) from "carrier cancelled DOT" (cancelled). Default 0.5 when neither. |
| `svc_mean_tenure_days` | `AVG(days_in_force WHERE ≥ 0)` | carrier-level aggregation | Retention proxy. |
| **Momentum** (6m vs 6-18m — recent-growth indicator) | | | |
| `carr_placements_6m` | `COUNT(DISTINCT usdot WHERE eff_date ≥ vintage − 6m)` | windowed count + LOO | Recent-growth signal. LOO subtracts if scored DOT is in the 6m window. |
| `carr_placements_6to18m` | `COUNT(DISTINCT usdot WHERE eff_date IN [vintage − 18m, vintage − 6m))` | windowed count + LOO | 1-year baseline for comparison. |
| `carr_state_momentum` | `carr_placements_6m / (carr_placements_6to18m / 2 + 1)` | ratio with half-period scaling + Laplace smoothing | Trajectory metric: >1 = growing, <1 = shrinking. Half-period scaling makes the ratio comparable (6m vs 12m). +1 smoothing prevents division explosion for new carriers. **Dual use** — exiting-carrier hygiene filter (§10.4) requires `momentum ≥ 0.1`. |

#### 5.3.4 Leave-one-out (LOO) adjustment — rationale and mechanics

**Problem.** `carr_mean_power` for carrier Y in state S is `AVG(power_units)` over DOTs placed with Y. If we score `(DOT X, carrier Y)` where X is actually placed with Y, X's own `power_units` is in the aggregate — the model sees "DOT X looks exactly like this carrier's typical client" partly because the carrier's typical client is defined by X. That's leakage.

**Fix.** `apply_loo_adjustment` (cell `Part 3A`) stores `_sum` columns alongside `_mean/_pct` features in `carrier_state_agg`. At train/score time, for each `(DOT, carrier)` pair where the DOT is actually in that cell:

```
dot_n = dot_n_events_in_cell  # from dot_cell_contrib
carr_mean_power_loo = (carr_power_sum − dot_n × dot_power_units)
                   / (carr_n_events − dot_n)
```

For DOTs **not** in the cell (soft negatives, counterfactual rankings), `dot_n = 0` → the adjustment is a no-op.

**18 features get this treatment**: 6 mean-of-numeric (`carr_mean_power/crash/insp_oos/revoc/oos`), 10 commodity-share (`carr_pct_*`), 2 radius-mean (`carr_mean_inter_long/intra_short`). Volume features (`carr_n_placements`, `carr_placements_6m`, `carr_placements_6to18m`) also get analogous DOT-count subtraction. `carr_state_momentum` is recomputed from the LOO-adjusted counts.

**Residual peer-leakage.** Other DOTs placed with Y are still in the aggregate — the model sees Y's typical client excluding X, but including Y's other N−1 clients. CV ROC-AUC therefore overestimates true out-of-sample performance. Bounded by `MIN_CARRIER_PLACEMENTS`: the smaller the cell, the larger the per-peer influence (~5-10% self-correlation per cell). §9.3 documents this.

**Median/p90 not LOO-adjusted.** Quantiles can't be cheaply updated by subtracting one point. Minor residual leakage, bounded as above.

#### 5.3.5 Sub-entity centroids (for distance-based sub-entity selection)

Each consolidated carrier group (e.g. `PROGRESSIVE GRP`) often has multiple raw sub-entities (`PROGRESSIVE COUNTY MUTUAL`, `PROGRESSIVE DIRECT`, …). The model recommends at group level, but a broker calls a specific legal entity. After we recommend a group, we attach the sub-entity whose typical-client centroid is closest to the DOT.

**Appetite features** (10, same scale as DOT features for distance):

`power_units_log`, `crash_count`, `inspection_oos_viol`, `crgo_genfreight`, `crgo_bldgmat`, `crgo_livestock`, `crgo_oilfield`, `crgo_intermodal`, `crgo_chem`, `crgo_passengers`.

**Weights**: proportional to within-cell variance across sub-entities — high-variance features ARE the discriminative ones for sub-entity selection.

**Distance**: variance-weighted Euclidean on the 10-feature vector. LOO-adjusted analogous to above.

### 5.4 Labelling scheme

Each `(DOT, carrier)` pair gets a binary label `y ∈ {0, 1}` and feeds into the training set. Three sources:

| Label | Condition | Rationale |
|---|---|---|
| **Clean positive** (`y=1`) | DOT's actual voluntary placement for this line, where `is_residual_market = 0 AND is_cancellation_pending = 0` | Voluntary active placement is our proxy for appetite fit. |
| **Hard negative** (`y=0`) | DOT paired with its *actual residual-market* carrier (`is_residual_market = 1`), OR a pair where `end_reason = 'revoked'` | Residual-market = state assigned, not carrier choice. Revoked = carrier *rejected* the DOT. Both are explicit anti-appetite signal. |
| **Soft negative** (`y=0`) | K=3 random carriers per positive, drawn from the DOT's state carrier pool, deduped against the DOT's actual placements | Covers the "would carrier Z take DOT X" counterfactual. Random because any carrier in the pool should be *a priori* equiprobable until features are considered. |

#### 5.4.1 How each `event_status` is treated

The raw data has 6+ event statuses. Our labels only use the clearest cases — ambiguous statuses are excluded from training rather than best-guessed.

| Status | Treatment | Reason |
|---|---|---|
| `active` + `is_residual_market=0` + `is_cancellation_pending=0` | **Clean positive** | Current voluntary placement — the appetite signal we want. |
| `active (cancellation pending)` | **Excluded** | Placement is in transition — neither "voluntary active" nor "revoked." Including as positive overstates appetite; including as negative overstates rejection. |
| `is_residual_market=1` (any status) | **Hard negative** | State-assigned, not carrier choice. Explicit anti-appetite — if this pair looked plausible from features alone the model would be systematically wrong. |
| `end_reason='revoked'` | **Hard negative** | Carrier explicitly dropped the DOT. Strongest possible anti-appetite signal. |
| `cancelled` | **Excluded** | Mixed signal — can be voluntary (DOT left) or carrier-initiated non-renewal. Without a definitive end-reason, we can't attribute. Labelling as negative would overcount carrier rejection; as positive would mislabel non-renewals. |
| `replaced` | **Excluded** | Carrier accepted the DOT at the time (looks like positive) but DOT subsequently chose to leave. Feeds `svc_voluntary_exit_ratio` as a carrier-level feature instead of a direct label. |
| `rejected` | **Excluded** (feeds `svc_rejection_rate` instead) | Pre-placement rejection — DOT never actually placed with carrier. Important signal but at carrier-level, better captured as a feature than as an individual negative label. |

The exclusion-first philosophy: it's better to train on fewer, cleaner labels than more noisy ones. Soft negatives (K=3 random per positive) provide volume; label noise would hurt more.

#### 5.4.2 K=3, oversample ×2 — why these specific values

**K = 3 soft negatives per positive:**
- K=1 under-samples the counterfactual space — the model learns a trivial boundary where observed carriers are "positive" and anything else is "unknown."
- K=10 inflates the training set 10× for diminishing returns. Early ablation showed <1 pt metric lift beyond K=3.
- K=3 vs K=5 ablated: within-noise metric differences, K=3 trains ~1.5× faster.
- K=3 keeps the positive:negative balance at roughly 1:3 (positives + 3 soft negs + a few hard negs), which LightGBM handles comfortably without class-weighting.

**Oversample ×2 (draw 6 candidates, keep 3 after dedupe):**
- Soft negatives are drawn randomly from the state pool, then deduped against the DOT's observed placements (positives + hard negs). A DOT with 4 observed placements in a 20-carrier state pool has 20% dedupe collision — drawing exactly K would leave fewer than K soft negs for some positives.
- Drawing 2K and capping at K after dedupe guarantees we hit K soft negs per positive without over-sampling for the typical case.

**Why random not stratified?** The counterfactual we want is "any plausible alternative carrier." Stratifying (e.g. by fleet band) would bias soft negatives toward similar-looking carriers — the model would learn to distinguish fine-grained cases but miss coarse ones. Random reflects the actual market composition.

#### 5.4.3 Why not retention-as-label?

Retention / churn / profitability data would be the gold standard for "appropriately placed" — a DOT that stays with a carrier 3+ years and renews cleanly is a well-placed DOT. We don't have that data. "Voluntary placement observed" is the available proxy. §8.1 covers this limitation honestly: the output is framed as **appetite outliers**, not placement-quality outliers.

### 5.5 Model selection

#### 5.5.1 Modelling paradigm — binary contrastive, not time-series or collaborative filtering

The task is "rank carriers by appetite fit for each DOT." Several paradigms could plausibly address this; we picked binary contrastive, and the reasoning matters because the paradigm choice determines what signal the model can extract.

**Binary contrastive** (chosen). For each `(DOT, carrier)` pair, predict `P(real pair | features)`. The observed carrier's rank among all in-state candidates is the appetite-outlier score. This is the paradigm most aligned with the task — we're literally asking "would this carrier take this DOT?" and ranking by the answer.

Why this fits the data:
- **Side-steps the missing ground-truth label.** We never predict "is this placement optimal" directly, which is the label we don't have.
- **Features generalise** across new DOTs and new carriers (as long as both appear in state pools) — unlike interaction-matrix methods that need every entity seen in training.
- **Classification machinery is well-understood** — SHAP, calibration, GroupKFold all work out of the box.
- **Honest about the uncertainty.** The score is `P(this is a real pair)`, which is interpretable and calibratable (§7.4). A ranking-only model gives order without magnitude.

**Alternatives considered:**

| Paradigm | Why not |
|---|---|
| **Direct best-carrier regression / multiclass** (predict a carrier per DOT) | No clean label for "best carrier." Multiclass over 400+ carrier groups is severely under-powered per class — even BIPD's 320K positives give <1K per class. |
| **Time-series / sequence model** (predict next placement given history) | DOTs don't switch frequently — a 24-month window has ~1 placement event per DOT. "Next placement given history" has near-zero positive signal for most DOTs. Also conflates two things: *when* the DOT switches (temporal) and *which carrier they switch to* (ranking). We only need the second. |
| **Collaborative filtering / matrix factorisation** (DOT × carrier interaction matrix) | Carrier pool per state is 20-200 while DOT universe is 400K+. CF would learn DOT embeddings but miss the carrier-appetite structure that cross-cuts states (a carrier writing hazmat in TX usually writes hazmat elsewhere). Our features-based approach generalises; CF doesn't. |
| **Bipartite graph (GNN)** | DOTs and carriers form a bipartite graph with placement edges. GNN could propagate signal across the graph. Correct in principle but: engineering overhead not justified for a take-home, loses the clean per-cell carrier-aggregate structure, needs more data than we have for carrier embeddings in thin cells (cargo). |

The contrastive paradigm maps the ranking question into a classification problem the data can actually answer.

#### 5.5.2 Three training regimes (base / weighted / temporal)

The notebook trains three model variants on the same features. They differ in training data and labelling weights:

| Regime | Training data | Labels | Weights | Purpose | Deliverable? |
|---|---|---|---|---|---|
| **Base** | All observed placements in the 24-month window | Clean positives + hard negatives + K=3 soft negatives (§5.4) | None (uniform) | **Primary model.** Standard contrastive setup. | **Yes** (`MODEL_FOR_DELIVERABLE='base'`) |
| **Weighted-tier** | Same training data | Same labels, but positives tiered by tenure confidence (strong / medium / weak based on `line_tenure` + `insurance_cancel_rate`) | `sample_weight` passed to LightGBM: 1.0 (strong) / 0.6 (medium) / 0.3 (weak); hard neg 1.0, soft neg 0.3 | **Sensitivity check.** Does a stricter interpretation of "appropriately placed" (positives must have *persisted*) improve the signal? | No — loses on every slice (§6.3) |
| **Temporal holdout** | **Only pre-cutoff** placements: `effective_date < vintage − 3 months` | Base labels on pre-cutoff positives only | None | **Honest out-of-sample validation.** Retrains from scratch without the recent 3m, then scores on the held-out window. | No — diagnostic only (§6.5) |

**All three share:** features, hyperparameters, LightGBM backend, GroupKFold CV, LOO adjustment, residual-market filtering. The training helpers in `Part 3A` (`_build_hard_negatives`, `_sample_soft_negatives`, `_dedupe_and_cap_soft`, `_assemble_training_frame`, `_fit_cv_and_refit`) are reused across all three — only labelling, weights, and the effective-date filter differ. This guarantees apples-to-apples comparison.

**Why three not one:** they answer different questions.
- **Base** → "best model given all available data."
- **Weighted** → "would a stricter positive definition change the recommendation?" (methodological sensitivity — the stricter definition is defensible, so we actually test it rather than assume).
- **Temporal** → "does the model generalise to data it never saw?" (operational generalisation — Part 4J uses the base model as an approximate check, Part 4K is the honest retrain).

The weighted model informs the labelling choice; the temporal model informs the generalisation claim. Only base feeds the deliverable.

#### 5.5.3 LightGBM — the model, in depth

**Backend: LightGBM binary classifier** (`lgb.LGBMClassifier`). HGBT (`HistGradientBoostingClassifier`) is used as a fallback only if LightGBM fails to import — same hyperparameter philosophy, 2-3× slower.

Why LightGBM specifically:
- **Native categorical-feature support.** `state` has 50+ levels, `naics_subsector` has hundreds. One-hot would 2-3× total feature count and defeat split-structure. Label-encoding without native categorical is lossy — the model treats encoded integers as ordinal (so `state=15` is "between" `state=14` and `state=16`, which is wrong). LightGBM's `categorical_feature` argument handles categories as unordered sets, learning splits like `state ∈ {TX, CA, FL}` directly.
- **Native `NaN` handling.** Each split learns a default direction for missing values. No imputation, no `is_null` indicator features — honest treatment of missingness.
- **Fast on medium-size data.** Histogram-based split finding is O(n_bins) not O(n_rows) per split. BIPD's ~2M training rows × 180 features train in ~2 min on 2 cores.

**Hyperparameters** (`make_classifier` in `Part 3A`):

| Hyperparameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 200 | CV log-loss plateaus by ~150 rounds on all 4 lines; 200 gives a 25% safety margin without inflating training time. |
| `num_leaves` | 31 | LightGBM default. Deep enough to capture `state × fleet_band × commodity` interactions; shallow enough that `reg_lambda` can prevent memorisation at this depth. |
| `learning_rate` | 0.08 | Slightly below default 0.1. With 200 rounds, 0.08 gives smoother convergence without needing early-stopping (we retrain on the full set after CV, so early-stopping would discard information). |
| `reg_lambda` | 1.0 | L2 regularisation on leaf weights. With 170+ features, some high-cardinality (`naics_subsector`), regularisation prevents over-fit to rare feature combinations. |
| `min_child_samples` | 20 | Minimum rows per leaf. Deliberately matched to `MIN_CARRIER_PLACEMENTS = 20` — a leaf never represents "effectively one carrier cell." |
| `n_jobs` | -1 | All cores. |
| `random_state` | 42 | Deterministic across runs. |

The hyperparameters weren't tuned via grid search for this take-home (§9 lists this as a next step). The values are deliberate defaults consistent with the data shape; the headline results (87.8% BIPD top-5, +19 pts over baseline) make it clear the ceiling isn't hyperparameter-bound.

**Categorical encoding: label-encoded via `CAT_MAPS`.** Fit once on all scorable DOTs: each `(column, value)` → integer in `[0, n_levels)`. `NaN` becomes `'UNK'` → its own encoded value. Unseen levels at inference map to `-1` (LightGBM treats as a separate category).

Not one-hot (`state` → 50 columns) because that would 2-3× feature count and defeat LightGBM's categorical splits. Not target-encoding because it risks leakage — the DOT's own observation feeds its encoding — and LightGBM handles categorical splits natively anyway.

**One model per line (4 models total), not a shared multi-line model with `line` as a feature.** Reasons:

- **Label distributions differ by orders of magnitude.** BIPD has ~320K positives; cargo has ~3.3K. Shared training would be dominated by BIPD, degrading the thin lines.
- **Carrier pools differ structurally.** BIPD has 400+ carrier groups; cargo has 64. The carrier aggregates (`carr_mean_power`, `carr_pct_hazmat`, …) have different distributions per line and would compete for the same tree splits.
- **Feature-importance patterns differ sharply** (§7.1). BIPD at 63% carrier-features, cargo at 51% DOT-numeric — a shared model would compromise.
- **Operational flexibility.** If WC drift becomes a problem, we retrain WC alone. Shared model forces retraining all four lines.

Trade-off: no cross-line transfer learning. Could add multi-task learning if operationalised.

**Final-model retraining.** After GroupKFold CV, each model is retrained on the full training set — standard practice to use all available data for the deployed model.

#### 5.5.4 Why no explicit feature selection

170+ features into a tree ensemble, no selection step. The LightGBM + L2 setup handles high-D well; SHAP (§7.1) confirms the top 20 features explain 60-80% of mass. Explicit feature selection would:

- Reduce the model's ability to find rare-interaction splits (e.g. `hazmat × specific commodity × specific state`).
- Add a hyperparameter (how many features) that would need its own tuning.
- Give negligible speed-up — LightGBM's column-wise histogram is already O(n_features) with low constant.

Features at 0.01% importance cost almost nothing. "Throw it all in with regularisation" is the right default for tree ensembles.

#### 5.5.5 Thresholds and magic numbers — one-stop reference

| Threshold | Value(s) | Rationale |
|---|---|---|
| `MIN_CARRIER_PLACEMENTS` | bipd/cargo/wc = 20; surety_bond = 15 | Below this, carrier cell aggregates are too noisy to be useful features. 20 chosen by inspection of distribution — the step where aggregate std stabilises. Surety_bond lowered to 15 because the pool is smaller; 20 would drop usable cells by ~30%. |
| `MIN_SUB_ENTITY_PLACEMENTS` | 10 | Sub-entity centroids need less data than full carrier aggregates (fewer features; LOO-adjusted at distance time). |
| `RESIDUAL_MARKET_CARRIER_THRESHOLD` | 0.5 | Carrier with ≥50% residual-market events ⇒ excluded from scoring pool. 0.5 is the empty middle of a bimodal distribution (voluntary carriers cluster near 0%, residual carriers near 100%). |
| `N_NEGATIVES_PER_POSITIVE` (K) | 3 | See §5.4.2. |
| `RECENCY_WINDOW_MONTHS` | 24 | See §5.1.1. |
| `fleet_band` bins | ≤1=B1, ≤5=B2, ≤20=B3, ≤100=B4, >100=B5 | Anchored at trucking carrier breakpoints — owner-operator (1), small (2-5), mid (6-20), large (21-100), mega (100+). These are the segmentations carriers actually underwrite against. |
| `STRICT_TENURE_MIN_PER_LINE` | bipd 1.5y / wc 1.0y / cargo 0.5y / surety_bond 0.5y | Line-specific "placement has persisted" threshold for weighted labelling. BIPD annual renewal → 1.5y = "survived one renewal plus a 6m grace." WC retention is faster. Cargo/surety_bond have less data; 0.5y is the minimum where positive counts remain usable. |
| `POS_WEIGHT_STRONG / MEDIUM / WEAK` | 1.0 / 0.6 / 0.3 | Monotonic decay for tenure confidence. 0.3 for weak matches `NEG_WEIGHT_SOFT` — a weak positive is treated with similar confidence to a soft negative. Medium at 0.6 is linear interpolation. |
| Temporal cutoff | 3 months (`vintage − 3m`) | Short enough that recent-placement sample size is non-trivial; long enough that recent placements can "settle" (early renewals/cancellations usually happen within weeks). 6m would halve the held-out sample. |
| Hygiene — exiting carrier | `momentum < 0.1 AND placements_6m < 5` | Both conditions required to avoid misclassifying slow-growth carriers as exiting. Momentum < 0.1 means <10% of baseline rate; placements_6m < 5 is "actually small volume, not just a lull." |
| Hygiene — high-churn flag ⚠️ | `svc_voluntary_exit_ratio > 0.7` | 70% of exits DOT-initiated — unusual. 50% would flag the median carrier. |
| Hygiene — high-rejection flag ❌ | `svc_rejection_rate > 0.3` | 30% rejection is ~10× the median voluntary-market rejection rate. |
| Hygiene — low-confidence flag 💭 | `fit_score < 0.2` on top-5 | Model's own best recommendation under 0.2 ⇒ bottom-tail of the calibration curve, high noise. |
| Bundling bonus | `+10%` per line in top-5 of ≥2 lines | Rough rule-of-thumb for the multi-line discount — real discount varies by carrier. Additive-per-line so 2 bundling lines don't 2× priority. |
| `TOP_N_OUTPUT` | 10,000 | Business constraint from the brief ("call list"). Below this, quality deteriorates into the long tail. |
| `TOP_K_PER_DOT` | 20 | 20 recommendations per DOT = enough depth without overwhelming the call sheet. |
| `TOP_K_FULL_RANKING` | 100 | Long-format output cap. 100 is deep enough for programmatic use (filtering, ad-hoc analysis). |
| `SCORING_BATCH_DOTS` | 3,000 | Intra-state DOT batching for Part 4A streaming scoring. Big states (TX/CA 50K+ DOTs × 70 carriers = 3.5M-row cross-join) blew past DuckDB's 6 GB otherwise. |

### 5.6 Residual-market carrier filtering

Carriers with `carr_pct_residual_market ≥ 0.5` (half their events are state-assigned) are filtered **from the recommendation pool** at scoring time. They **stay in training as hard negatives** so the model still learns to down-score them for voluntary DOTs.

**Rationale.** The business output is a *call list* — brokers shouldn't be pitching DOTs toward residual-market carriers. Filtering at the output guarantees no residual-market recommendation reaches the final CSV, regardless of model prediction.

### 5.7 Validation design

#### 5.7.1 Cross-validation

**Choice: 3-fold `GroupKFold` with `groups = usdot`.**

Why grouped by `usdot`: a DOT often has multiple events (positives + hard negatives), and random K-fold would split those across train/val — leaking the DOT's identity. GroupKFold ensures a DOT is entirely in one fold.

Why 3-fold: 5-fold doubles training time for marginal CV-noise reduction given the sample sizes. 3-fold still has robust `n` per fold for all lines except cargo.

#### 5.7.2 Metric selection

| Metric | Why chosen | Why not alternatives |
|---|---|---|
| **Top-K hit rate** (K=1, 5, 10) | Directly measures "would the model's top-K include the DOT's actual carrier?" — matches broker use case (a call list is a top-K cut). | Precision/recall are label-threshold-based, which doesn't apply to a ranking task. |
| **MRR** (Mean Reciprocal Rank) | Rewards ranking the correct answer high. Smooth over the whole ranking, not just the K cut. | Rank-weighted; comparable across lines with different pool sizes. |
| **CV ROC-AUC + PR-AUC** | Standard classifier calibration check. Reported alongside hit-rate as a sanity signal. | We don't use AUC as the primary metric because it averages over all score thresholds — a model with poor top-K but good AUC is useless for a call list. |

#### 5.7.3 Baselines

- **B1 — random-in-state**: pick a random carrier from the state's eligible pool. Pure chance level, varies with pool size.
- **B2 — popularity by state × fleet-band**: rank carriers by their count of placements in `(state, fleet_band)`. This is the "no model, just rank by volume" baseline — the real bar to beat.

Any lift above B2 is attributable to the model's learned appetite signal beyond coarse pool statistics.

#### 5.7.4 Validation suite — ten diagnostics with explicit failure modes

Ten independent checks beyond CV metrics. Each probes a specific failure mode; the pass criterion was defined before running the check so results are honest rather than retro-fitted. Full results in §6-7.

| # | Diagnostic | Why we picked this check | Failure mode we were watching for | Result |
|---|---|---|---|---|
| 1 | **Headline hit-rate** (Part 4A) | Primary business metric — "would the model's top-K include the DOT's actual carrier?" | Model doesn't beat the popularity baseline on BIPD or WC → the model has no signal beyond coarse pool statistics. | **PASS** — +19 pts BIPD, +34 pts WC on top-5. |
| 2 | **SHAP feature importance** (Part 4B) | Verify the model is using carrier aggregates (which encode appetite) and not just DOT-level signals (which could overfit on peer leakage from LOO's residual limitation). | Carrier features at <20% importance → model learnt DOT patterns only. Would mean the carrier aggregates aren't contributing and metrics are inflated by peer-leakage. | **PASS** — 63% / 82% / 92% carrier-features for BIPD/WC/surety_bond. Cargo at 51% carrier flagged as yellow. |
| 3 | **Error analysis** (Part 4C) | Characterise the ~5% of placements the model ranks poorly. If random, signal is thin; if concentrated, model is coherent and the "errors" may be true appetite outliers. | Errors scatter across DOT types with no pattern → model is guessing on hard cases. | **PASS** — errors concentrate on Texas county mutuals + specialty surplus programs. Expected given TX market structure (§7.2); confirms model behaves coherently. |
| 4 | **Known-good / known-bad audit** (Part 4D) | Industry-knowledge check *independent of the data pattern* — hard-code known mainstream vs specialty carriers and verify the model separates them for voluntary DOTs. | Mainstream median rank ≥ specialty median rank → model miscalibrated for voluntary DOTs; recommending specialty for the mass market. | **PASS** — median rank gap +22 (BIPD), +40 (WC). Mainstream carriers consistently rank above specialty for voluntary DOTs. |
| 5 | **Fit score calibration** (Part 4G) | Score needs to be usable as a *confidence number*, not just an ordinal rank — brokers want to filter by threshold (`fit_score < 0.1`). | Flat or inverted deciles → score is noise beyond rank order; `priority` composite breaks. | **PASS** — monotonic. D9 shows 15-93% observation rate across lines; D0 is 0% for all. |
| 6 | **Pool-size scatter** (Part 4H) | Diagnose confounder — are high hit rates driven by small pools (small-N artefact) or genuine model quality? | `corr(pool_size, top-5) ≈ -1` → metrics are explained by pool size alone; the "lift" is spurious. | **PASS** for BIPD/WC — BIPD TX at 80% top-5 despite pool=179. Cargo/surety_bond flagged as pool-size artefact and documented. |
| 7 | **Synthetic DOT** (Part 4I) | Controlled test — hand-build 5 clean mainstream profiles and verify top-5 doesn't contain specialty. If model has learnt a spurious correlation (e.g. "small fleet in TX = specialty"), this test catches it. | Specialty in top-5 for a clean mainstream profile → spurious signal learned. | **PASS 4/5**; 1 yellow flag for TX (county-mutual structure makes specialty-adjacent recommendations legitimate in TX only). |
| 8 | **Approximate temporal** (Part 4J) | Cheapest temporal check — doesn't require retraining. Splits observed placements by `effective_date ≥ vintage − 3m` and compares the same model's hit-rate on recent vs older. | > 15 pt drop → severe temporal drift. 5-10 pt would indicate mild drift. | **PASS** — ~8 pt WC top-5 drop (robust n=11.9K), 11 pt BIPD (small n=111). |
| 9 | **True temporal train/test** (Part 4K) | Most honest test — retrain the model from scratch on pre-cutoff only, score the held-out window. Independent evidence vs #8. | > 15 pt drift → generalisation failure. Convergence with #8 → mild intrinsic drift, not an artefact of the approximate check. | **PASS** — ~7.5 pt WC drift matches the ~8 pt from Part 4J, triangulating that WC has genuine mild drift rather than catastrophic failure. BIPD n=111 too small for a confident claim; Part 4J ≈ Part 4K convergence on WC is the main evidence. |
| 10 | **Weighted-tier sensitivity** (Parts 4E/4F) | Does a stricter interpretation of "appropriately placed" (positives must have persisted ≥ 1.5y) improve the appetite signal? Methodologically defensible so we test it. | Weighted beats base on long-tenure slice → use weighted; label noise is the limiting factor. Base beats weighted → selection bias in weighted is more costly than base's label noise. | **PASS for base** — base wins on all slices including the long-tenure proxy (−0.014 on `top-5 (tenure ≥ 1.5y)`). Deliverable uses base. |

**Pass criteria were pre-registered** — each failure mode was written down before running the check. This matters: it's easy to find a story that explains any observed pattern after the fact. The convergence of Part 4J (approximate, 8 pt) and Part 4K (honest, 7.5 pt) on WC is the strongest single piece of evidence that the model generalises; it is also the single result most likely to invalidate the methodology, so an honest value here was the main thing to verify.

---

## 6. Results

### 6.1 Headline hit rate (base model, Part 4A)

Reported on observed placements (actual DOT → actual carrier). Top-K hit rate = % of DOTs where actual carrier is in model's top-K.

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

**BIPD and WC**: significant lift above popularity. Model learns genuine appetite patterns beyond coarse fleet-size matching.

**Cargo and surety_bond**: saturate at ~100% because carrier pools are tiny (cargo ~64 state-cells total ⇒ 2-5 carriers per state; surety_bond ~206 ⇒ 3-10 per state). Hit rates are misleadingly high — documented as pool-size artefact. See §7.5.

### 6.2 Segmentation — BIPD by fleet band

| Fleet band | n | MRR | top-5 |
|---|---|---|---|
| B1 (≤1 truck) | 176,390 | 0.759 | 0.913 |
| B2 (2-5) | 97,342 | 0.681 | 0.867 |
| B3 (6-20) | 33,084 | 0.576 | 0.788 |
| B4 (21-100) | 11,180 | 0.528 | 0.743 |
| B5 (100+) | 2,167 | 0.480 | 0.721 |

Monotonic degradation with fleet size — easier for small-fleet DOTs (fewer carriers writing them, more concentrated market), harder for mega-fleets (bespoke relationships, more diverse carrier mix). Expected and acceptable.

### 6.3 Weighted-tier sensitivity (Part 4F)

Alternative labelling: positives tiered by tenure confidence (strong = 1.5y+, medium = 0.5-1.5y, weak = <0.5y), with LightGBM `sample_weight` passed accordingly (1.0 / 0.6 / 0.3).

| Line | Metric | Base | Weighted-tier | Delta |
|---|---|---|---|---|
| BIPD | top-5 (all) | 0.878 | 0.872 | −0.007 |
| BIPD | top-5 (tenure ≥ 1.5y) | 0.889 | 0.881 | −0.008 |
| WC | top-5 (all) | 0.853 | 0.826 | −0.028 |
| WC | top-5 (tenure ≥ 1.5y) | 0.851 | 0.837 | −0.014 |
| WC | MRR (tenure ≥ 1.5y) | 0.723 | 0.693 | −0.029 |

**Base model wins on every slice, including the long-tenure proxy that most directly reflects "appropriately placed."** The weighted-tier approach is retained as a sensitivity check. **Deliverable uses base model** (`MODEL_FOR_DELIVERABLE = 'base'`).

Methodologically weighted is more aligned with the task framing; empirically it loses — likely because the tenure threshold introduces selection bias toward older DOTs rather than improving appetite signal.

### 6.4 Approximate temporal (Part 4J)

Partitioned observed placements at `vintage − 3 months`.

| Line | Slice | n | top-5 | top-10 | MRR |
|---|---|---|---|---|---|
| BIPD | recent (<3m) | 111 | 0.757 | 0.856 | 0.608 |
| BIPD | older (≥3m) | 206K | 0.870 | 0.931 | 0.711 |
| WC | recent (<3m) | 11.9K | **0.793** | 0.914 | 0.595 |
| WC | older (≥3m) | 69K | **0.870** | 0.931 | 0.730 |

**WC drops 8 pts on top-5 for recent placements** (robust `n`). BIPD drops 11 pts but on only 111 DOTs (sampling-noise dominant).

### 6.5 True temporal train/test (Part 4K)

Honest out-of-sample: hold out all placements with `effective_date ≥ vintage − 3 months` entirely from training, retrain from scratch, score held-out window.

| Line | n held-out | top-5 | Drift vs full-data CV |
|---|---|---|---|
| WC | 11,934 | 0.78 | −7.5 pts |
| BIPD | 111 | n/a | sample too small |

WC drift of 7.5 pts closely matches the approximate 4J finding (~8 pts), triangulating that WC has genuine temporal shift of ~7 pts rather than catastrophic generalisation failure. BIPD's sample is too small for a confident claim; the convergence of 4J ≈ 4K for WC is evidence that model generalises in principle.

---

## 7. Discussion

### 7.1 Feature importance (Part 4B)

| Line | Carrier features | DOT-numeric | DOT-categorical |
|---|---|---|---|
| BIPD | 63.2% | 22.0% | 14.7% |
| WC | 82.0% | 7.1% | 10.9% |
| Surety_bond | 92.0% | 4.4% | 3.7% |
| Cargo | 51.4% | 45.9% | 2.6% |

**Carrier features dominate 3 of 4 lines** — the model is learning actual carrier appetite, not just coarse DOT patterns. This validates the carrier-aggregate feature design.

**Cargo's 46% DOT-numeric share is a yellow flag**: with only 64 cells of carrier-side signal, the model falls back on DOT-level correlates. Combined with the saturated pool-size metric (§6.1), cargo's model is informationally thin.

**Top BIPD features**: `carr_n_placements` (dominant), `days_since_bipd_effective` (tenure), `svc_short_tenure_cancel_rate` (carrier service), `authority_bipd_file`, `state`, `carr_mean_oos`, commodity-share features.

**Top WC features**: `carr_n_placements`, `carr_pct_livestock`, `naics_subsector`, `carr_mean_oos`, `carr_pct_meat`, `carr_state_momentum`, `carr_mean_power`. Note how commodity shares matter more for WC — reflects that WC carriers specialise by industry (meat processing, livestock) in ways BIPD carriers don't.

### 7.2 Error analysis (Part 4C) — hard cases are Texas county-mutual + specialty market

Observed-rank distribution for BIPD: 94.2% of actual placements in top-10, 5.8% in rank 10+. The hard 5.8% are overwhelmingly:

- `MERCURY COUNTY MUTUAL`, `INFINITY COUNTY MUTUAL` — Texas's non-standard auto market structure (county mutuals write non-standard risks).
- `BENCHMARK INS`, `STAR INS`, `MSIG SPECIALTY`, `KNIGHT SPECIALTY`, `HOXBRIDGE` — specialty surplus programs.
- `TEXAS FARM BUREAU` — regional affinity carrier.

**Model consistently recommends mainstream (Progressive, BRK, Travelers, Canal, Great West) for these DOTs** — a 1-truck non-hazmat TX DOT on Mercury County Mutual *is* a plausible appetite outlier for the broker to call. The "error" is the use case working as intended.

### 7.3 Known-good / known-bad audit (Part 4D)

Hard-coded industry-knowledge check: 12 mainstream carriers (Progressive, Travelers, BRK, Zurich, Chubb, CNA, Liberty, Nationwide, Great West, Canal, Hartford, AIG) vs 14 specialty/residual (Texas Mutual, Mercury/Infinity County Mutual, Prime, Knight Specialty, Trisura, Southlake Specialty, …).

For voluntary-market DOTs, where does each bucket land in the model's ranking?

| Line | Mainstream median rank | Specialty median rank | Gap |
|---|---|---|---|
| BIPD | 7 | 29 | **+22** |
| WC | 12 | 52 | **+40** |

Model structurally separates mainstream from specialty for voluntary DOTs. PASS.

### 7.4 Fit score calibration (Part 4G)

Fit score monotonically indicates placement likelihood. Observed-placement rate per decile:

| Line | Decile 0 | Decile 5 | Decile 9 (highest fit) |
|---|---|---|---|
| BIPD | 0.00% | 0.17% | **15.3%** |
| WC | 0.00% | 0.09% | **15.9%** |
| Cargo | 0.00% | 0.02% | **25.4%** |
| Surety_bond | 0.00% | 0.00% | **93.0%** |

Score usable as confidence number, not just ordinal rank. Broker can filter calls by threshold (e.g. `fit_score < 0.1` on actual carrier → strong appetite-outlier signal).

### 7.5 Pool-size caveat (Part 4H)

Per-state scatter of pool size vs top-5 hit rate:

| Line | corr(pool_size, top-5) | Interpretation |
|---|---|---|
| BIPD | **−0.72** | Strong pool-size effect. CA (pool 84) → 0.91 vs TX (pool 179) → 0.80. |
| WC | −0.30 | Moderate pool-size effect. |
| Cargo | n/a (all saturate at 1.0) | Documented pool-size artefact. |
| Surety_bond | n/a (all saturate at 1.0) | Documented pool-size artefact. |

Confirms the cargo/surety_bond saturation is pool-size driven. BIPD still achieves 80% top-5 even in TX (largest pool) — genuine signal, not just pool-size inflation.

### 7.6 Synthetic DOT (Part 4I)

5 canonical mainstream DOTs (mid-fleet, no crashes, non-hazmat, genfreight-only, voluntary) across different states. Does model's top-5 include specialty/surplus?

- **4/5 clean**: top-5 are all mainstream (Progressive, Travelers, Great West, Erie, Sentry, Acuity).
- **1 yellow flag**: a TX DOT returned Trisura and Cimarron in top 5 alongside mainstream. Explained by TX's county-mutual market structure — in TX, specialty carriers write a wider share than elsewhere, so a mainstream TX profile legitimately has some specialty exposure.

### 7.7 Market analysis — CA / TX (Phase 9)

Focused analysis of the two largest state markets.

**Market structure.**

| State × Line | Carriers | HHI | Top-5 share |
|---|---|---|---|
| BIPD / CA | 84 | 0.042 | 37.2% |
| BIPD / TX | 179 | 0.111 | 57.7% |
| WC / TX | 136 | **0.226** | 58.8% |

**Carrier momentum.**

- **CA BIPD gainers**: Infinity Select, Commodore RRG, Universal Casualty RRG, Spinnaker Specialty — state is shifting toward risk-retention groups and specialty surplus.
- **CA BIPD exiters** (zero placements last 6m): Imperium, Obsidian Specialty, Transverse Specialty.
- **TX WC gainers**: Stonetrust, Vanliner, Hanover American, PIE Insurance — mainstream carriers growing.

Tied to `svc_voluntary_exit_ratio` to distinguish healthy growth from churn-driven decline.

**Appetite-outlier call sheet example** (BIPD / CA):

```
DOT 859192 | rank 83/84 | actual=OBSIDIAN SPECIALTY INSURANCE COMPANY
           fleet=2 band=B2 hazmat=1 crashes=0 tenure=1.7y
           top-3 recs: GREAT WEST (0.81) | TRAVELERS GRP (0.78) | IAT GRP (0.39)
```

Interpretation: small-fleet CA DOT with hazmat exposure but no crashes, placed with a specialty surplus insurer. Three mainstream alternatives with high fit scores — strong candidate for first call.

### 7.8 Multi-line bundling (Part 7A)

For DOTs scored on multiple lines, we count carriers appearing in top-5 across ≥2 lines. These are **bundling opportunities** — a broker can pitch all lines to the same carrier for discount leverage.

Bundling bonus applied as `+10%` to priority score per line that bundles with the top-recommended carrier.

### 7.9 Recommendation hygiene (Part 7B)

Post-processing filters to reduce false-positive recommendations.

| Filter | Logic | Reason |
|---|---|---|
| Exiting carriers | `momentum < 0.1 AND placements_6m < 5` | Carrier appears to be leaving the state × line — recommendation is useless. |
| Previously-rejected pairs | DOT has `event_status = 'rejected'` with this carrier | Already said no. Don't re-pitch. |
| High churn flag ⚠️ | `svc_voluntary_exit_ratio > 0.7` | Carrier retention is unusually bad. Soft flag. |
| High rejection flag ❌ | `svc_rejection_rate > 0.3` | Carrier rejects a lot. Soft flag. |
| Low-confidence DOT 💭 | `fit_score < 0.2` on top-5 recs | Model is uncertain. Soft flag. |

Hygiene version of outputs saved as `*_hygiene.csv`.

---

## 8. Known Limitations

### 8.1 Positives label is a noisy proxy for appropriate placement

We measure *carrier appetite* as a proxy for appropriateness. See §2.3 for the four regimes we can't disentangle. Output is honestly framed as **appetite outliers**, not placement-quality outliers directly. For production, retention / renewal / profitability data would be the gold-standard labels.

### 8.2 Pool-size inflation on cargo and surety_bond

With 2-5 carriers per state for cargo and 3-10 for surety_bond, top-5 hit rates saturate regardless of model quality. Treat these lines as supplementary — primary deliverable value is BIPD and WC.

### 8.3 Peer leakage (bounded)

LOO adjustment removes a DOT's own contribution from carrier aggregates. Peer DOTs placed with the same carrier are still in the aggregate. This inflates CV ROC-AUC vs true out-of-sample generalisation. Bounded by `MIN_CARRIER_PLACEMENTS`: ~5-10% self-correlation per cell.

### 8.4 Temporal drift

WC top-5 is ~7 pts lower on recent placements vs older — triangulated between the approximate (4J, 8 pts) and honest (4K, 7.5 pts) temporal checks. Model generalises in principle but underperforms on bleeding-edge WC placements.

### 8.5 Carrier consolidation inflates absolute metrics

Group-level recommendations ("call Progressive") rather than sub-entity ("call Progressive Direct specifically"). Mitigated with sub-entity distance ranking (§5.3.5), but headline hit-rate numbers are measured at group level — sub-entity-level numbers would be lower. Baselines are measured on the same consolidated pool, so **model-vs-baseline lift is apples-to-apples**.

### 8.6 Known structural quirks accepted

- **GL / non-BMC-84 surety**: dropped from scope (§4.1).
- **Texas Mutual** (TX WC state fund): included as mainstream TX WC because the data provider doesn't flag it as residual-market. Methodologically it IS the TX residual-market option — treat the model's WC/TX ranking with that context.
- **Fronting carriers** (Southlake Specialty, Highlander Specialty, MS Transverse Specialty, Trisura): shells for MGUs/MGAs. The real risk-taker is behind them. Can't reliably un-front without an external map.

### 8.7 No ground-truth validation available

All validation uses the same underlying data (observed placements) in different cuts. True validation would require operational outcomes from Rosella's broker team — "we called the top-100, how many converted?"

---

## 9. Next Steps

If this were a production system:

1. **Learning-to-rank (LambdaRank)** instead of binary classification — directly optimises NDCG@K, aligned with the ranking metric we actually care about.
2. **Market-segmented models** — separate models for standard-risk vs non-standard-risk DOTs with routing on DOT features. Would specifically improve the Texas county-mutual / specialty-market segment.
3. **Retention outcomes as labels** — if Rosella's broker outcomes become available (renewal / churn / cancellation / profitability), supervised signal for "placement quality" rather than relying on the appetite proxy.
4. **Fronting-carrier unwinding** — map Southlake / Highlander / MS Transverse / Trisura shells to actual risk-takers via a fronting-relationship table (external data).
5. **Dynamic per-DOT scoring context** — serve different top-K depending on broker intent (new placement vs renewal vs re-shop).
6. **Live operational feedback loop** — Rosella's actual call outcomes feed back as positive/negative labels for continuous retraining. Reduces reliance on the appetite proxy over time.
7. **Hyperparameter search** — current LightGBM config is a reasonable starting point, not an optimum. GridSearch / Optuna run with GroupKFold would likely squeeze another 1-3 pts.
8. **Sub-entity-level direct modelling** — train on sub-entity grain rather than group+distance. Needs careful handling of thin-sub-entity cells (would push `MIN_SUB_ENTITY_PLACEMENTS` lower).

---

## Appendix A — Deliverable interpretation

### A.1 `rosella_appetite_outliers.csv` (wide — the call list)

Top 10,000 DOTs ranked by `priority`.

| Column | Meaning |
|---|---|
| `usdot`, `company_name`, `state`, `power_units` | DOT identifier + minimal profile |
| `worst_line` | Line (bipd/wc/cargo/surety_bond) with worst-fit current placement |
| `observed_carrier` | Current carrier for `worst_line` |
| `observed_rank_in_line` | Model's rank of the observed carrier within state's pool |
| `pool_size` | How many carriers in the state's pool for this line |
| `appetite_outlier_score` | `observed_rank / pool_size`. 1.0 = at bottom of pool, 0.0 = at top. |
| `top20_recommended` | Pipe-separated list of top-20 recs, format `"GROUP (best_subentity)"` |
| `priority` | Composite score (formula below) |

**Priority score formula:**

```
priority = appetite_outlier_score × log1p(power_units) × (1 + 0.1 × n_bundling_lines)
```

Where:
- `appetite_outlier_score ∈ [0, 1]` — 1.0 means actual carrier is at the bottom of the pool
- `log1p(power_units)` — scales up larger fleets (more revenue potential per account) without letting mega-fleets 100× dominate the call list
- `n_bundling_lines` — number of lines where the top-recommended carrier also appears in top-5 of ≥1 other line for this DOT

Rationale:
- **Multiplicative** across the first two terms — a DOT with both high outlier score AND large fleet should rank above either alone.
- `log1p` not raw PU — a 500-PU fleet is more valuable than a 5-PU fleet, but not 100× more.
- **10% bundling bonus is additive-per-line** — 2 bundling lines → +20%, not 2× — because the bundling discount is real but marginal.

Sort by `priority` DESC — broker calls from top.

### A.2 `rosella_rankings_long.csv` (long — programmatic)

| Column | Meaning |
|---|---|
| `usdot`, `line`, `state`, `rank`, `carrier_group`, `fit_score` | ranking |
| `is_observed` | 1 if this is the DOT's actual carrier on this line |
| `pool_size` | state × line pool size |
| `best_subentity` | closest sub-entity within the carrier group |
| `subentity_distance` | variance-weighted Euclidean distance DOT → sub-entity centroid |

### A.3 How a broker uses this

1. Open `rosella_appetite_outliers.csv` (or `_hygiene.csv` for the filtered version).
2. From the top:
   - Pull DOT profile: fleet, state, line of concern.
   - Note the actual poorly-ranked carrier.
   - Pitch top-3 recommendations from `top20_recommended`.
3. For deeper research, cross-reference with the long CSV to see fit scores and sub-entity recommendations for all lines this DOT operates in.

---

## Appendix B — Reproducibility

Pipeline is cached end-to-end. Fresh kernel + full warmup takes ~2 min from cached state, ~45 min from scratch.

### B.1 Cache files (in `cache/`)

| File | Produced by | Load time |
|---|---|---|
| `analysis.duckdb` | cell 5 opens; cells 1E, 2A-2D populate | ~1 s |
| `vintage_and_provider_map.pkl` | cell 1C | ~1 s |
| `dot_df.parquet` | cell 2E | ~15-20 s |
| `models.pkl` (joblib) | Part 3C | ~5 s |
| `scored_obs_rank.pkl` | Part 4A | ~20 s |
| `line_tenure.pkl` | Part 4E | ~1 s |
| `models_weighted.pkl` | Part 4E | ~5 s |
| `scored_obs_rank_weighted.pkl` | Part 4F | ~20 s |
| `models_temporal.pkl` + `results_temporal.pkl` | Part 4K | ~5 s |

### B.2 Invalidating caches

- Flip `FORCE_REBUILD = True` in cell 5 to nuke the cache dir on next run.
- To invalidate only consolidation-dependent tables (after `CARRIER_RULES` changes):
  ```python
  for tbl in ['carrier_lookup','recent_events','service_composite',
              'carrier_state_agg','dot_cell_contrib',
              'sub_entity_centroids','dot_subent_contrib']:
      con.execute(f"DROP TABLE IF EXISTS {tbl}")
  ```
  Then re-run cells 1C, 2A-2D. `dot_features` (cell 1E) and `dot_df` (cell 2E) stay cached.

### B.3 Switching models for the deliverable

Set `MODEL_FOR_DELIVERABLE` in cell 1A to `'base'` or `'weighted'`. Re-run cell 1A, then Parts 5A and 5B. **Current recommendation**: `'base'` (wins empirically; see §6.3).

### B.4 Memory architecture

- **DuckDB with `memory_limit = '6GB'`, `threads = 2`, `preserve_insertion_order = false`** — tuned for a laptop with limited RAM.
- **Cell 1E (`dot_features`) chunked into two passes** (core + feature expansion) because a single 163-aggregate GROUP BY over 12M events exceeded 6 GB.
- **Part 4A streaming scoring uses intra-state DOT batching** (`SCORING_BATCH_DOTS = 3000`) — big states (TX, CA, 50K+ DOTs × 70 carriers = 3.5M-row cross-join) blew past 6 GB otherwise.
- **File-backed DuckDB** (`cache/analysis.duckdb`) — tables persist across kernel restarts.

---

## Appendix C — Phase / cell map

| Cell | Phase | Purpose |
|---|---|---|
| 1-5 | — | Imports, DuckDB connect, cache helpers, raw CSV view |
| 6-19 | — | Earlier EDA / discovery (not required for the final model) |
| 20 (md), 21-24 | **Phase 1 — Setup** | Carrier consolidation rules, data vintage, dot_features table (1A-1C, 1E) |
| 25 (md), 26-30 | **Phase 2 — Aggregates & features** | carrier_lookup, recent_events, service composite, carrier_state_agg, sub-entity centroids, scorable-DOTs, derived features (2A-2E) |
| 31 (md), 32-33 | **Phase 3 — EDA & Data Quality** | Sanity check (1D), placement count diagnostic (2F) |
| 34 (md), 35-37 | **Phase 4 — Base model training** | Feature defs + helpers, placement extraction, LightGBM training (3A-3C) |
| 38 (md), 39-46 | **Phase 5 — Base model validation** | Scoring/baselines (4A), SHAP (4B), errors (4C), known-good/known-bad (4D), calibration (4G), pool-size (4H), synthetic DOT (4I), approximate temporal (4J) |
| 47 (md), 48-49 | **Phase 6 — Weighted-tier alternative** | Retrain with tenure-weighted labels (4E), side-by-side comparison (4F) |
| 50 (md), 51 | **Phase 7 — Temporal train/test split** | True out-of-sample validation (4K) |
| 52 (md), 53-54 | **Phase 8 — Deliverables** | Combine, sub-entity distance, deliverable CSVs (5A-5B) |
| 55 (md), 56-58 | **Phase 9 — Market analysis** | CA/TX structure, carrier momentum, call sheets (6A-6C) |
| 59 (md), 60-61 | **Phase 10 — Recommendation hygiene** | Multi-line bundling + hygiene post-processing (7A-7B) |
