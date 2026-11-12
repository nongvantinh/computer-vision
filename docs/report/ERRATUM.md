# Erratum for the Phase 1 report (`report.tex`, `report.pdf`)

The report and its figures are left as published. Two corrections apply.

1. **Undefended counts for 39 attacks.** In the original `full_200` run, 39 of the 600 attack rows
   (13 images at three budgets) were scored on pixels that differ from the stored adversarial arrays,
   most likely because those attacks were crafted again after the first evaluation. Rescoring the
   stored arrays (`controls_v1`) gives 186 undefended successes instead of 185 (4/255: 21 instead of
   22; 8/255: 53 instead of 52; 16/255: 112 instead of 111). JPEG 90, 75 and 50 are identical in both
   runs. Raw-answer agreement between the two runs is 96.8% (581 of 600), and 100% (561 of 561) on rows
   with identical inputs. No conclusion of the report changes. Details: `docs/phase2_screen_plan.md`,
   section 2.
2. **The Dangerzone condition uses the legacy adapter.** The adapter embedded the image as JPEG 75
   through Pillow's PDF writer, so the Dangerzone result in the report cannot be attributed to
   Dangerzone. The lossless adapter (`dangerzone_ll`) was added in Phase 2. See `docs/paper/paper.tex`
   and `docs/phase2_preregistration.md`.
