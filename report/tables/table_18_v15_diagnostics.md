| Diagnostic | Value | Score | Details |
| --- | --- | --- | --- |
| v15 ensemble ranker | sklearn_extra_trees | 7.016 (7016/10000) | 32 candidates; selected baseline 6.857 (6857/10000) |
| Top ranker choices | model::epoch_08=417; hybrid_v13_gate::epoch_08=163; hybrid_strict_gate::epoch_08=38; hybrid_v13_gate::epoch_04=32; hybrid_v13_gate::epoch_06=30 | 7.016 (7016/10000) | ranker often selects model-only, but uses hybrid candidates enough to improve final score |
| v15 type-specific expert | route_types=[] | 6.927 (6927/10000) | no target type routed to expert because expert did not beat baseline |
| Expert-vs-baseline target types | GSM_FOBAR: best_expert=398@epoch_07 baseline=427 route=False; GSM_SV: best_expert=391@epoch_08 baseline=436 route=False; MATH_FOBAR: best_expert=197@epoch_07 baseline=208 route=False; MATH_SV: best_expert=186@epoch_05 baseline=196 route=False | 6.927 (6927/10000) | expert improves over epochs but remains below general baseline on target types |

Caption: **Table 18. V15 diagnostic details.** V15 ensemble is the best local validation score, while the type-specific expert does not route any target type because it remains below the general baseline.
