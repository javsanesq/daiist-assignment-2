# Selected dataset: technical evidence companion

Dataset-selection checkpoint: 26 October 2026. I accepted the published VinBigData-derived chest X-ray version described below and the project-defined target of **any of its 14 annotated findings**. This document records data and audit facts. Training results and my conclusions are presented in REPORT.md.

## Exact public version and unchanged local export

- Workspace: `vinbigdataxrayproject`; project: `chest-xray-symptom-detection`; version: **3**; task: object detection; export: YOLOv8.
- Public version: [Chest xray symptom Detection, version 3](https://universe.roboflow.com/vinbigdataxrayproject/chest-xray-symptom-detection/dataset/3).
- Local data: `datasets/vinbigdata-v3/`; configuration: `datasets/vinbigdata-v3/data.yaml`.
- The supplied README names the version `2025-06-11 11:24pm` and records export on 13 June 2025 at 08:09 GMT. The local download was recorded on 26 October 2026.
- Published preprocessing: fit each image into **1024 x 1024 pixels with black padding**. The export states **no image augmentation**. These are properties of the public export, not a commitment about subsequent training-time augmentation.
- Downloaded archive: 1,137,139,887 bytes; SHA-256 `686af6bba0f78a9f69d88d017a4da7fc0f75e7ab60a059319dbb8d6b7493394e`.

Images, labels and all three published splits were retained as supplied. No duplicates were removed, labels changed, datasets merged, or new split created. The Roboflow partition is its own partition of 14,998 source training images; it is **not** the original competition's 15,000/3,000 training/test partition. The original download metadata records its historical candidate-audit status; the later acceptance decision is recorded here and in `PROJECT_STATUS.md`.

## Image-level target and split sizes

The unit is one exported **image**, not a verified patient, visit or complete examination. A target-positive image has at least one annotation from the 14 classes below; a target-negative image has none. All 10,604 empty annotation files in this version match explicit `No finding` source records. This interpretation is established by the source join, not by treating an arbitrary empty label file as healthy.

| Published split | Images | Any finding: positive | No listed finding: negative | Positive share | Retained boxes |
|---|---:|---:|---:|---:|---:|
| Training | 10,499 | 3,051 | 7,448 | 29.06% | 10,648 |
| Validation | 3,000 | 919 | 2,081 | 30.63% | 3,172 |
| Test | 1,499 | 424 | 1,075 | 28.29% | 1,545 |
| Total | 14,998 | 4,394 | 10,604 | 29.30% | 15,365 |

The 3,000-image validation set offers substantially more observations for threshold analysis than the 40-image DENTEX v2 or 141-image DentXpert v4 alternatives. I preferred more validation observations; this does not establish clinical representativeness. The observed split prevalence is not the approved hypothetical deployment prevalence in `scenario.json`.

The accepted application is an educational simulation of image-review priority. `No finding` denotes absence of these listed source findings; it does not establish overall health. Finding presence is a proxy chosen for this project, not a measured urgency, severity, emergency or intervention-time outcome. All images still receive clinical assessment. For image-level scoring, the planned score is the largest confidence among predictions in the positive class set (all 14 classes), with zero when no such box is predicted. Negative images must remain in both evaluation and cost denominators.

## Classes and retained annotation counts

IDs below are the exact zero-based export order. `Nodule-Mass` corresponds to source spelling `Nodule/Mass`; the other class names match the source join. `ILD` is the source abbreviation for interstitial lung disease. Source class 14, `No finding`, is represented by an empty annotation file rather than a fifteenth detection class.

There is exactly one retained box per present class per image. Consequently, each entry below is both the number of boxes and the number of images with that class. Class counts overlap across images with several findings and must not be summed as distinct positive images.

| Export ID | Class | Train | Validation | Test | Total |
|---:|---|---:|---:|---:|---:|
| 0 | Aortic enlargement | 2,134 | 632 | 301 | 3,067 |
| 1 | Atelectasis | 131 | 35 | 20 | 186 |
| 2 | Calcification | 304 | 106 | 42 | 452 |
| 3 | Cardiomegaly | 1,590 | 492 | 218 | 2,300 |
| 4 | Consolidation | 241 | 64 | 48 | 353 |
| 5 | ILD | 246 | 100 | 40 | 386 |
| 6 | Infiltration | 434 | 121 | 58 | 613 |
| 7 | Lung Opacity | 909 | 276 | 137 | 1,322 |
| 8 | Nodule-Mass | 577 | 180 | 69 | 826 |
| 9 | Other lesion | 796 | 220 | 118 | 1,134 |
| 10 | Pleural effusion | 706 | 209 | 117 | 1,032 |
| 11 | Pleural thickening | 1,378 | 397 | 206 | 1,981 |
| 12 | Pneumothorax | 70 | 21 | 5 | 96 |
| 13 | Pulmonary fibrosis | 1,132 | 319 | 166 | 1,617 |

All classes occur in every split, but some denominators are small: pneumothorax has only 21 validation and 5 test images; atelectasis has 35 and 20. The large overall validation split does not remove uncertainty for rare classes.

## Box-size summaries

Area is exported box width multiplied by height, divided by the full 1024 x 1024 image area **including black padding**. Values below are **10th percentile / median / 90th percentile, expressed as percentages of image area**. They describe supplied boxes, not true lesion extent or clinical severity. VinBigData v3 contains 15,365 valid box rows and no polygon rows. Full precision remains in `evidence/vinbigdata-v3/audit.json`.

| Class | Train: q10 / median / q90 (%) | Validation: q10 / median / q90 (%) | Test: q10 / median / q90 (%) |
|---|---:|---:|---:|
| Aortic enlargement | 0.686 / 1.119 / 2.001 | 0.687 / 1.141 / 2.218 | 0.717 / 1.088 / 1.986 |
| Atelectasis | 0.915 / 2.837 / 10.329 | 0.983 / 2.905 / 7.344 | 1.763 / 2.933 / 10.821 |
| Calcification | 0.036 / 0.247 / 2.575 | 0.042 / 0.284 / 1.935 | 0.041 / 0.148 / 2.189 |
| Cardiomegaly | 3.228 / 4.904 / 7.822 | 3.407 / 4.954 / 7.791 | 3.205 / 4.965 / 7.740 |
| Consolidation | 0.680 / 2.694 / 7.574 | 0.807 / 2.527 / 7.348 | 0.989 / 2.915 / 6.073 |
| ILD | 1.844 / 5.412 / 14.154 | 1.361 / 5.417 / 11.227 | 1.337 / 6.749 / 13.940 |
| Infiltration | 0.633 / 2.558 / 7.354 | 0.787 / 2.335 / 6.829 | 0.363 / 2.068 / 9.278 |
| Lung Opacity | 0.283 / 1.532 / 6.179 | 0.168 / 1.435 / 5.949 | 0.289 / 1.981 / 5.638 |
| Nodule-Mass | 0.019 / 0.136 / 2.170 | 0.023 / 0.136 / 2.772 | 0.026 / 0.152 / 2.893 |
| Other lesion | 0.059 / 1.091 / 6.731 | 0.092 / 1.358 / 7.341 | 0.064 / 1.186 / 5.008 |
| Pleural effusion | 0.138 / 0.474 / 8.510 | 0.129 / 0.536 / 8.594 | 0.139 / 0.538 / 7.452 |
| Pleural thickening | 0.119 / 0.336 / 1.282 | 0.126 / 0.340 / 1.378 | 0.134 / 0.337 / 1.322 |
| Pneumothorax | 0.439 / 5.197 / 21.583 | 0.848 / 4.039 / 17.093 | 0.543 / 0.944 / 2.303 |
| Pulmonary fibrosis | 0.128 / 0.757 / 4.188 | 0.097 / 0.842 / 3.313 | 0.128 / 0.750 / 4.818 |

Small retained regions, including the roughly 0.136% median training area for `Nodule-Mass`, make localization more sensitive to image resizing. These summaries do not determine model input size or guarantee performance.

## Source provenance and annotation correspondence

The [VinBigData competition documentation](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection/data) defines 14 localized findings plus `No finding`, with image IDs, radiologist IDs and pixel box coordinates. The source [VinDr-CXR publication](https://www.nature.com/articles/s41597-022-01498-w) and [PhysioNet release documentation](https://physionet.org/content/vindr-cxr/1.0.0/) describe collection from two Vietnamese hospitals, with multiple radiologists. The richer PhysioNet taxonomy has 22 local findings and 6 global labels; those counts must not be presented as the taxonomy of this 14-class export. Original training images had three independent reader annotations. The original test consensus process does not describe this Roboflow test partition.

The public CSV used for the local correspondence check was a **third-party GitHub mirror**, not a direct official competition download: [train_full_dataset.csv](https://raw.githubusercontent.com/AlexisBalayre/chest-x-ray-abnormalities-detection/main/data/train_full_dataset.csv). It contains 67,914 rows and 15,000 image IDs; SHA-256 `d3c52e5b96a73b9263e535ee803dde7fd053e4c8c6a50faf6b00f7dd8f5e2a80`. Its schema and aggregate counts agree with the primary descriptions. This is traceable correspondence evidence; it does not independently verify the mirror's entire custody history or source image pixels.

Every one of the **14,998** exported image IDs and every image's set of class names matches that CSV. The 4,394 positive images are all retained. The source has 10,606 `No finding` images, while the export has 10,604; two source IDs are absent for an undocumented reason: `ddbe41935f4c7c4c6624654585e12652` and `f652652f33d50987fec8a4b136433edd`. Both are `No finding` in the source. No exported source ID repeats across splits. Source records and original image-pixel equivalence are different checks; the original DICOM archive was not downloaded for pixel comparison.

Evidence: `evidence/source_vinbigdata_provenance.json` and `evidence/vinbigdata-v3/source_id_comparison.json`.

## Representative-box limitation

The export has **15,365 image/class groups and exactly 15,365 boxes**, compared with **36,096 raw abnormal annotation rows** in the matching source. It retains class presence but not all radiologists' boxes or every annotated same-class region.

Geometry comparisons on 4,036 multiclass images with repeated source annotations strongly support retention of the **first source CSV row for each image/class**. After fitting scale and offset separately on each axis, this rule agrees within two exported pixels for all 4,036 images and within one pixel for 4,003. Mean, median, final-row and union-box rules fit much less often. Because original dimensions are absent from the CSV, this is a fitted geometric comparison, not independent verification of the uploader's transformation code.

There are 2,782 image/class groups across 1,697 images with at least one additional same-reader source box having overlap below 0.1 IoU with the first box (3,856 additional boxes). This demonstrates omitted source annotation regions; it does not adjudicate separate clinical lesions. Additional annotations from other readers can also represent alternative views of the same finding.

Therefore, native detection metrics describe agreement with the **published representative boxes**, not exhaustive lesion localization or radiologist consensus. A prediction on a source-annotated region omitted by this export can be counted as an unmatched detection. The verified image-level class sets support the selected finding-presence proxy, but do not validate clinical urgency. Do not call these boxes merged, averaged, consensus, or weighted-box-fused.

Evidence: `evidence/vinbigdata-v3/geometry_aggregation.json` summarizes the correspondence checks; source-image illustrations and intermediate per-image records remain local.

## Integrity, repetition screening and patient independence

The local audit found no unreadable-image or invalid-label issues, no missing or orphan annotation files, no identical decoded-pixel groups, and no shared source image IDs. A 64-bit dHash screen produced **323,220 cross-split candidate pairs**; similar chest outlines make these candidates especially nonspecific.

Every candidate received cached 64 px grayscale and high-pass detail correlations. A union of the strongest combined/component rankings and a joint high-correlation rule selected **822 pairs** for 256 px SSIM and ORB/homography checks. No pair met both cheap correlation thresholds (intensity at least 0.985 and detail at least 0.90). The **20 highest-ranked** paired previews were inspected. No copied image was established in those previews; this is a bounded review, not proof that none exist.

The leading train/validation pair remains unusually similar and **unresolved as possibly related acquisitions**: source IDs `8242cf16d8311b8af8c0280899d8ed4b` and `b829a8ee3f5ba1d16bc28c47b3d70a34`. It has 249 geometrically consistent ORB matches out of 304, but visible diaphragm, gas-pattern and positioning differences. Registered detail correlation was 0.614 and masked SSIM 0.817. Neither a copied image nor shared patient identity was established.

Source image IDs are derived from DICOM SOP instance identifiers, not patient IDs; the [source de-identification code](https://github.com/vinbigdata-medical/vindr-cxr/blob/main/de-identification/anonymize.py) supports that distinction. Patient identifiers are unavailable in the export, and no verified patient mapping or patient-disjoint assertion was found. Patient independence is **unverified**, not demonstrated or disproved. Hash screening can miss cropping, large transformations and related acquisitions. No claim of leakage-free, patient-independent or prospective clinical validation is supported.

Evidence: `evidence/vinbigdata-v3/audit.json`, `image_manifest.csv`, and `strong_similarity_review/` (screening methodology, top-pair findings and visual-review conclusions; bulk intermediate scores and image previews remain local). Primary-source wording is recorded in `evidence/source_vinbigdata_provenance.json`.

## Licence and use notices

The Roboflow version, `data.yaml` and supplied README report **CC BY 4.0** and attribute the upload to a Roboflow user. The [Kaggle competition](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection) lists use as subject to Competition Rules. The original [PhysioNet release](https://physionet.org/content/vindr-cxr/1.0.0/) uses the Credentialed Health Data License 1.5.0 and describes restrictions including no redistribution.

These notices belong to different releases; the uploader's authority to apply CC BY 4.0 has not been independently verified. I retain this discrepancy rather than treating the Roboflow label as established permission for every upstream source. This is a provenance disclosure, not a legal conclusion. The submission package excludes copied source images. No restricted original archive was accessed in these source checks.

## Decision and evaluation boundaries

I accepted this version with the representative-box, source/patient and licensing limitations explicit, and selected presence of **any of all 14 findings** as priority-positive. Existing images and labels remain unchanged. This is an image-level educational evaluation of the published derivative.

`scenario.json` is the authority for the approved hypothetical operating assumptions. At the original dataset-selection checkpoint, policy approval was still pending. The completed project now records the adopted assumptions and thresholds in `scenario.json` and `results/frozen_policy.json`; those later records supersede the initial selection status. Dataset audit inspection of test labels/images is data-quality review, not model-performance selection. The completed model evaluation and sensitivity analysis are documented in REPORT.md and the included results.
