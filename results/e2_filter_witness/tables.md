# E2 filter witness: exact privacy loss vs ledger

Order alpha = 7, per-record RDP budget B = 0.6, per-token cap 0.25; claim eps = 2.424 at delta = 1e-6. 72 toy LMs (V^T up to 4^7 = 16,384 outputs).

| policy | runs | violations (D_alpha > B) | max D_alpha / B | median D_alpha / B | max ledger | max exact delta at claimed eps |
|---|---|---|---|---|---|---|
| filter | 72 | 0 | 1.000 | 0.985 | 0.600 | 0.00e+00 |
| fixed | 72 | 0 | 0.833 | 0.809 | 0.500 | 0.00e+00 |
| overspend_filter | 72 | 57 | 1.250 | 1.230 | 0.600 | 0.00e+00 |
| wrong_order_filter | 72 | 72 | 5.433 | 3.395 | 0.600 | 4.87e-03 |
| one_sided_filter | 72 | 0 | 1.000 | 0.985 | 0.600 | 0.00e+00 |
