# Code review notes

This folder is a duplicated working copy of `Vis_per_comp-main`. The original
folder was not modified.

## Applied improvements

- Added `pengwin_pipeline.py` with reusable preprocessing, split, transform,
  model, inference, and postprocessing helpers.
- Fixed the data bootstrap gap: the notebook already ensured labels, but image
  download was not actually called before normalization.
- Added strict image/label pairing by filename stem before splitting data.
- Centralized the two training variants:
  - `context`: `RandSpatialCropd`, the best-performing setup in the notebook.
  - `foreground`: `RandCropByPosNegLabeld` plus flip, kept for comparison.
- Kept the existing checkpoints unchanged. The best reported model remains the
  non-augmentation/context model.
- Kept secure zip extraction and added URL allow-list validation for the
  expected Zenodo record.

## Review findings

- The README says the notebook downloads both zips automatically, but the image
  preprocessing cell did not call `ensure_dataset()` for images.
- The notebook relies on execution state. Several cells redefine `model`,
  `val_loader`, and transforms, so a fresh run is sensitive to cell order.
- The image-label pairing used `zip(sorted(images), sorted(labels))`; this is
  fragile when one side has missing or extra files.
- Final evaluation uses the validation split as the reporting split. This is
  acceptable for the course TP if stated clearly, but it should not be described
  as a held-out challenge/test score.
- The best-performing result is coherent: preserving anatomical context with
  `RandSpatialCropd` beats aggressive foreground sampling for this task.

## Recommended presentation stance

- Lead with the clinical/computer-vision problem: binary 3D bone segmentation
  from pelvic CT.
- Emphasize that the strongest result is the context-preserving model:
  `Mean Dice RAW = 0.8405`, `Mean Dice POST = 0.8914`.
- Present the augmentation experiment as an ablation that taught something:
  foreground-balanced crops increased local bias and false positives; cleanup
  helped but did not catch the context model.
- Be explicit about limitations: no external test set, binary segmentation
  collapses fragment labels, and threshold/postprocessing were tuned on the
  validation workflow.
