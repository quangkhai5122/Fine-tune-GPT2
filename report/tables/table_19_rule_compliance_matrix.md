| Pipeline | Uses only query_vi/response_vi? | Uses provided type? | Uses original/source group fields? | Uses public valid labels for final decision? | Report status |
| --- | --- | --- | --- | --- | --- |
| v13/v14 retrieval | no | yes | yes | no | diagnostic only if strict query_vi/response_vi rule applies |
| v14.5 valid-select | no | yes | yes | valid checkpoint selection | diagnostic; optimized on public valid |
| v15 ensemble | no | yes | yes | yes | best diagnostic score; do not submit as-is under strict rules |
| v15 type-specific | no | yes | yes | route check by type | diagnostic; expert route disabled in final output |
| v16 legal answer-only | yes | no | no | internal split only | submission-safe direction; no local result yet |
| v16 legal query retrieval | yes | no | no | internal split only | submission-safe direction; query_vi-only nearest train query; no local result yet |

Caption: **Table 19. Diagnostic versus submission-safe pipeline features.** The table distinguishes research/audit branches from strict query_vi/response_vi-only directions.
