| Error type | Description | Example id | Diagnostic |
| --- | --- | --- | --- |
| Parse failure | No final number or wrong final-answer anchor | 355 | output format |
| Arithmetic error | Correct relation pattern but wrong numeric computation | 6 | equation check |
| Relation error | Misread relation in the word problem | 18 | relational plan |
| Spurious shortcut | Predicted a frequent answer pattern | 40 | counterfactual |
| Correct answer, wrong reasoning | Answer is correct but rationale is unsupported | NaN | manual rationalization audit |
| Source-memory failure | Same source group has conflicting train answer | 30 | retrieval/source audit |
| Type confusion | GSM/MATH or FOBAR/SV variant confused | 30 | type-wise analysis |

Caption: **Table 12. Qualitative error taxonomy.** The taxonomy separates answer-format failures, computational failures, relational reasoning failures, and rationalization/source-memory risks. Example ids are heuristic seeds for manual inspection.
