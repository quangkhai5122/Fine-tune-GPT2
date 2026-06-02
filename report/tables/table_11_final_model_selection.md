| Candidate | Official valid score | Source-disjoint score | Overlap-valid score | Runtime | Reasoning quality | Final choice |
| --- | --- | --- | --- | --- | --- | --- |
| Best answer-only | 4.329 (4329/10000) | NaN | 4.696 (4696/10000) | NaN | low/medium | no |
| Best compact reasoning | 2.131 (2131/10000) | 1.396 (1396/10000) | NaN | 200.4 min train | medium | no |
| Best hybrid retrieval (v13) | 6.896 (6896/10000) | NaN | 6.337 (6337/10000) | 177.7 min final retrain | retrieval-dependent | strong baseline; no longer best score |
| Best full-train valid-select (v14.5) | 6.823 (6823/10000) | NaN | NaN | NaN | retrieval-dependent | no; below v13/v15 despite valid selection |
| Best diagnostic ensemble (v15) | 7.016 (7016/10000) | NaN | NaN | NaN | answer-selection/retrieval-dependent | best diagnostic score; not submission-safe as-is |
| v14 gate sweep | 6.806 (6806/10000) | NaN | 6.145 (6145/10000) | 169.0 min final retrain | retrieval-dependent | no; below v13 on official and overlap-valid |
| v14 FOBAR/SV retrieval | 6.379 (6379/10000) | NaN | 5.846 (5846/10000) | 131.0 min final retrain | retrieval-dependent; weak FOBAR/SV | no; extended retrieval hurts target variants |

Caption: **Table 11. Final model selection.** The submitted configuration is selected by balancing official score, runtime, robustness, and risk of overfitting to source overlap.
