# Lineage-Equivalence Scoring Map

Cell-type accuracy is scored at the **coarse lineage** level: a predicted label counts
as correct if it maps to the same family as the author's annotation. This credits
biologically valid finer or equivalent calls (e.g. "Plasma cell" vs author "B cells";
"Goblet cell" vs author "Epithelial cells"; "Alveolar macrophage" vs author "Myeloid cells").

Families: **lymphoid, myeloid, epithelial, stromal**.

The mapping is implemented as an ordered substring-priority list (`_FAMILY_PRIORITY` in
`code/nb_10a_validation_benchmark.py`). Ordering matters because some labels contain
shadowing substrings (e.g. "gobleT CELL" contains "t cell"; "Epithelial Mesenchymal
Transition" contains "epithelial"; "Alveolar macrophage" contains "alveolar"). Specific
/ compound terms are therefore tested before bare tokens.

## Examples of intended equivalences
| Predicted | Author | Family | Correct |
|-----------|--------|--------|---------|
| Plasma cell | B cells / B lymphocytes | lymphoid | yes |
| CD8+ T cell, NKT, Treg, NK cell | T/NK cells | lymphoid | yes |
| Alveolar macrophage, Monocyte, pDC | Myeloid cells | myeloid | yes |
| Goblet cell, Ductal cell, Ciliated, Cancer cell | Epithelial cells | epithelial | yes |
| Fibroblast, Endothelial cell | Stromal cells | stromal | yes |

## Caveat (for reviewers)
This mapping is load-bearing for the headline accuracy number. It should be treated as
a frozen, versioned artifact and independently reviewed. A future revision may replace
substring-priority matching with an exact dictionary plus explicit synonym lists to
remove ordering fragility.
