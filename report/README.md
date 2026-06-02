# Report audit assets

Generated from local `dataset/`, `audit/`, and `results/` artifacts.

## Key audit findings

- Train records: 95,400; official validation records: 1,000.
- Exact query overlap in official validation: 0/1000.
- Normalized query overlap in official validation: 0/1000.
- Source-group overlap by the v11-v13 source key: 965/1000 (96.5%).
- High source-group overlap means official validation can reward source memorization even when exact query overlap is zero.
- v15 ensemble is the best diagnostic official-valid run at 7.016/10, but it depends on candidate ranker/valid labels and is not submission-safe under strict rules.
- v13 remains the strongest simple retrieval baseline; v14.5 valid-select does not beat it, and v14 FOBAR/SV extended retrieval remains a negative ablation.
- The core reasoning conclusion is cautious: target alignment, source-memory retrieval, and answer selection explain more gain than robust source-disjoint reasoning.

## Outputs

- Tables: `report/tables/*.csv`, `report/tables/*.md`, and combined `report/tables.md`.
- Figures: `report/figures/figure_01_source_query_overlap_structure.png` through `figure_13_rule_compliance_matrix.png`.
- Figure source data: `report/data/*.csv`.
- Machine-readable summary: `report/report_audit_summary.json`.
- Report text additions: `report/report_additions_v8_v15.md`.

## NaN policy

`NaN` marks values that cannot be recovered from the local artifacts, especially v3/v4 metrics, v11-v15 source-disjoint scores, runtimes without manifests, and human-rated reasoning quality.
