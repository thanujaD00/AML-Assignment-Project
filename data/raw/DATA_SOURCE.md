# Data source and provenance

| Item | Value |
|---|---|
| File | `disease-surveillance-2026-10-10.xlsx` (sheets `Data` and `About`) |
| Primary source | Epidemiology Unit, Ministry of Health, Sri Lanka — *Weekly Epidemiological Report*, Table 1 "Distribution of Notified Diseases reported by Medical Officers of Health, by RDHS division" — <https://www.epid.gov.lk/weekly-epidemiological-report/weekly-epidemiological-report> |
| Compiled by | research.dulajmendis.com, dataset *disease-surveillance*, version 2026-10-10 (first release) — <https://research.dulajmendis.com/data/disease-surveillance/> |
| Integrity check | SHA-256 `9f075df094ad55691bb07d0d1a594f047d719ee1dc637f8acc244e739a3f70bb`, identical to the hash published on the compiler's page (checked automatically by the pipeline and by `tests/test_data.py`) |
| Licence | The Epidemiology Unit states no licence; its reports ask that the Epidemiology Unit be acknowledged as the primary data source in all resulting publications. |
| Personal data | None. All values are aggregated case counts per health division. |

## Contents (1,242 rows)

| Measure | Rows | Dates |
|---|---|---|
| Dengue, cases notified in the week | 270 (26 divisions + Sri Lanka × 10 weeks) | weeks 24–33 of 2026 (week ending 2026-06-14 … 2026-08-16) |
| Dengue, cases notified this year to date | 270 | same 10 weeks |
| Dengue, cases notified in weeks 1 to 52 | 81 (27 × 3 years) | 2023, 2024, 2025 |
| Leptospirosis (same three measures) | 621 | same dates — **not used** in this project |

The 26 places are the Regional Director of Health Services (RDHS) divisions. They follow district
boundaries, except that Ampara district is split into the Ampara and Kalmunai divisions.

## Notes from the compiler (sheet `About`)

* Weekly figures come from 10 consecutive issues, Vol. 53 Nos. 25–34.
* Vol. 53 No. 27 printed the same dengue counts for week 26 as No. 26 did for week 25, in all 26
  divisions; its year-to-date column adds them again. The compiler kept the figures as printed and
  advises reading that week with caution.
* Up to 2025 the report weeks ran Saturday–Friday; in 2026 they run Monday–Sunday.
* Week-52 totals leave out the last days of December that fall in the following year's first week.

## Additional issue found by this project

The division-level counts for **week 24 (week ending 2026-06-14)** do not follow the pattern of any
other week: their Spearman rank correlation with week 25 is 0.08 (all other neighbouring week pairs
are ≥ 0.95), and with the 2023–25 annual totals it is 0.03 (other weeks 0.76–0.83). For example
Colombo shows 20 cases (1,138 the next week) and Kilinochchi 419 (6 the next week; its whole 2025
total was 127). The values still sum exactly to the national total, which suggests the numbers were
attached to the wrong divisions somewhere between the printed report and the dataset. We could not
check the original PDF from our environment, so the week is **flagged and excluded, not corrected**.
See `reports/tables/data_quality_rank_consistency.csv`.

## How to cite

> Epidemiology Unit, Ministry of Health, Sri Lanka, Weekly Epidemiological Report, Table 1.
> Compiled by research.dulajmendis.com, dataset disease-surveillance, version 2026-10-10.
