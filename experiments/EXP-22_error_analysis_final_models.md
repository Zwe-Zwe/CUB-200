# EXP-22: Error analysis of the two final models (test set, no retraining)

**Status:** done 2026-10-20 · **Command:**
`python -m birdcls.evaluate --analyse convnext_tiny__tl_L6_hires__full resnet50.tv2_in1k__tl_L6_hires_strongaug__full`
· **Outputs:** `outputs/error_analysis/<run>/` (summary.json, per_class_accuracy.csv, top_confusions.csv,
test_predictions.csv, per_class_hist.png, confused_pairs.png, **misclassified_examples.png**)

## Why

The final models (EXP-20) were chosen and evaluated already; this step only **explains** their test errors for the report
(Results & Discussion). Nothing here changes a model or a decision, so looking at the test set is legitimate: it is
analysis after the single final evaluation, not selection. `--analyse` does not touch `results.csv`.

`misclassified_examples.png` shows, for the 4 most frequent confusions, the **actual misclassified test photo** (most
confident wrong one) above a training photo of the species it was mistaken for.

## Headline numbers (no TTA, 1,204 test images)

| | ConvNeXt-Tiny final | ResNet-50 final |
|---|---|---|
| Top-1 / avg per class | **90.20% / 89.94%** | 86.46% / 86.12% |
| Top-5 | **98.01%** | 96.76% |
| Errors | 118 | 163 |
| Errors within the same bird group (same last word, e.g. Gull→Gull) | 57 (48%) | 80 (49%) |
| Classes at 100% / below 50% / at 0% | 125 / 3 / 0 | 98 / 6 / 0 |
| Median class accuracy | 100% | 86.6% |
| Mean confidence correct / wrong | 0.72 / 0.29 | 0.74 / 0.35 |
| Accuracy when confidence ≥ 0.5 (coverage) | 97.8% (74.5%) | 96.2% (74.6%) |

Confidence values are low in absolute terms because of label smoothing 0.1 (EXP-16), but they still separate correct
from wrong predictions well: wrong predictions have far lower confidence.

## Where the errors are

**ConvNeXt-Tiny, worst classes:** Chuck-will's-widow 25% (n=4), Henslow's Sparrow 29%, Common Tern 43%, Belted
Kingfisher / Whip-poor-will / Red-faced Cormorant / Tennessee Warbler 50%.
**Most common confusions:** Boat-tailed Grackle→Brewer's Blackbird (3), Loggerhead Shrike→Great Grey Shrike (3),
Pelagic→Red-faced Cormorant, Fish Crow→American Crow, California→Ring-billed Gull, Henslow's→Savannah Sparrow,
Common Tern→Elegant / Arctic Tern (2 each).

**ResNet-50, worst classes:** Boat-tailed Grackle 14%, Ring-billed Gull 20%, Tennessee Warbler 33%, American Crow /
California Gull 40%. **Most common:** Boat-tailed Grackle→Brewer's Blackbird (5), Ring-billed→Herring Gull (3),
Bohemian→Cedar Waxwing (3).

Interpretation: almost every frequent confusion is between **visually near-identical sibling species** (all-black
icterids and crows, gulls, terns, cormorants, shrikes, sparrows), where the field marks are tiny (eye colour, bill
shape, tail length) or are seasonal / vocal (Fish vs American Crow are mainly told apart by call). About half of all
errors stay inside the same coarse bird group even by the crude "last word of the name" grouping; the true share of
"sibling" errors is higher (e.g. Brewer's Blackbird vs Boat-tailed Grackle are different words but the same family).
Two error sources visible in `misclassified_examples.png`: the Pelagic Cormorant is in flight and small (pose /
scale), and the Fish Crow photo genuinely has no diagnostic feature.

## Model comparison on the same images

| | count |
|---|---|
| Both correct | 1011 |
| ConvNeXt correct, ResNet wrong | **75** |
| ResNet correct, ConvNeXt wrong | 30 |
| Both wrong | 88 (47 with the same wrong species) |

These discordant counts are the McNemar test of EXP-20 (75 vs 30, p = 1.3e-5). ConvNeXt-Tiny is better on 54 classes,
worse on 20, equal on 126; per-class accuracies correlate (r = 0.67), so both models find the **same species hard**:
the difficulty is mostly in the data, and ConvNeXt fixes many of ResNet's errors (Ring-billed Gull +60 pp, Bohemian
Waxwing / Boat-tailed Grackle +43 pp) rather than making different ones. Of the 88 shared errors, 47 are the same
wrong species, which suggests ambiguous images or near-identical classes rather than model-specific failures.

Note for future work (not done, would need validation-based selection): an oracle that is right whenever either model
is right would reach 1116/1204 = 92.7%, so an ensemble has some headroom; with only 30 ResNet-only wins it would at best
add a little and doubles inference cost.

## Report use

- Results & Discussion, "misclassifications": `outputs/error_analysis/convnext_tiny__tl_L6_hires__full/misclassified_examples.png`
  + the table above (Top-5, errors within group, confidence).
- Per-class histogram (`per_class_hist.png`) to show average-per-class ≈ Top-1 (no class collapses; minimum 25%).
