# Assignment 2 Report
<!-- student-revised-report -->

- **Name**: Javier Sánchez Esquivel
- **Student ID**: 19731
- **Email**: jsanchez.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

This local educational study combines genuine model results with explicitly hypothetical operating costs. Numerical tables come from the saved experiment evidence. AI assistance is disclosed below.

## Business problem, and the cost of being wrong

A hypothetical hospital radiology team uses one chest X-ray image to assign review priority: routine clinical review, an extra human priority check, or accelerated clinical review. Every image still receives clinical assessment. One image is not a verified patient or complete examination. A positive means at least one annotated target finding, so this is a finding-presence proxy for priority, not a validated measure of clinical urgency.

I chose a higher cost for a missed priority because overlooking a potentially dangerous finding could delay attention and make a problem harder or more expensive to address later. An unnecessary priority mainly creates additional staff work. This is why I use EUR 500 for a missed priority and EUR 25 for an unnecessary priority, a 20:1 ratio.

These are assumed hospital-related costs for comparing workflows, not measured hospital costs or guaranteed consequences. An informal discussion with a relative working in hospital management supported their plausibility, but did not provide an audited costing study. The EUR 25 represents downstream work caused by a wrong escalation, separate from the EUR 2 initial priority check. A reviewed image that is wrongly escalated therefore incurs EUR 27 in these two activities; an automatically escalated false positive incurs EUR 25. Shared clinical assessment is excluded equally from all alternatives.

## Dataset

The exact Roboflow Universe source is workspace **vinbigdataxrayproject**, project **chest-xray-symptom-detection**, version **3** ([public version](https://universe.roboflow.com/vinbigdataxrayproject/chest-xray-symptom-detection/dataset/3)); the task is object detection. The original published splits, images and labels are retained. I chose a larger validation set for threshold selection and accepted this chest X-ray version with its annotation and independence limitations.

The positive set is every one of the 14 export classes: 0: Aortic enlargement; 1: Atelectasis; 2: Calcification; 3: Cardiomegaly; 4: Consolidation; 5: ILD; 6: Infiltration; 7: Lung Opacity; 8: Nodule-Mass; 9: Other lesion; 10: Pleural effusion; 11: Pleural thickening; 12: Pneumothorax; 13: Pulmonary fibrosis. An explicit empty annotation means none of these categories is present. All 10,604 empty labels match source `No finding` records; missing labels are separately rejected. This does not establish general health.

All 14,998 exported image IDs and image-level class sets match a third-party public mirror of the source annotation CSV; 2 of its 15,000 source image IDs are absent from this export. The public mirror was not an authenticated official download. The Roboflow test split is a partition of source training images, not the original competition's test set.

The export preserves representative source-reader boxes rather than exhaustive same-class regions or radiologist consensus. Image-level class presence is corroborated, but localization scores can penalize predictions on omitted source regions. Exact pixel/source-ID checks found no confirmed cross-split repetition; one highly similar train/validation pair remains unresolved as possibly related acquisitions. Image IDs are not patient IDs, so patient independence is unverified. The export's licence notice and upstream release notices also differ; no redistribution entitlement is inferred. Full provenance and bounded audit findings are in the [dataset companion](docs/DATASET_DESCRIPTION.md).

## What the dataset's composition implies

Training images teach the model what to look for and identify. Validation images help us check how it performs on images it was not trained on, and choose the review boundaries. Having 3,000 validation images gives us more evidence for comparing those boundaries, although some individual findings are still rare. If images from the same patient appear in both sets, the model might recognise patient-specific features instead of demonstrating that it generalises to new patients. We do not have verified patient identifiers, so we cannot rule this out. More images alone would not resolve that limitation.

| Split | Images | Positive | Explicit negative | Positive share | Boxes |
| --- | --- | --- | --- | --- | --- |
| Train | 10,499 | 3,051 | 7,448 | 29.1% | 10,648 |
| Validation | 3,000 | 919 | 2,081 | 30.6% | 3,172 |
| Test | 1,499 | 424 | 1,075 | 28.3% | 1,545 |

| Finding | Train boxes | Val boxes | Test boxes | Median train box area |
| --- | --- | --- | --- | --- |
| Aortic enlargement | 2134 | 632 | 301 | 1.119% |
| Atelectasis | 131 | 35 | 20 | 2.837% |
| Calcification | 304 | 106 | 42 | 0.247% |
| Cardiomegaly | 1590 | 492 | 218 | 4.904% |
| Consolidation | 241 | 64 | 48 | 2.694% |
| ILD | 246 | 100 | 40 | 5.412% |
| Infiltration | 434 | 121 | 58 | 2.558% |
| Lung Opacity | 909 | 276 | 137 | 1.532% |
| Nodule-Mass | 577 | 180 | 69 | 0.136% |
| Other lesion | 796 | 220 | 118 | 1.091% |
| Pleural effusion | 706 | 209 | 117 | 0.474% |
| Pleural thickening | 1378 | 397 | 206 | 0.336% |
| Pneumothorax | 70 | 21 | 5 | 5.197% |
| Pulmonary fibrosis | 1132 | 319 | 166 | 0.757% |

Across all training annotations, box-area q10/median/q90 are 0.149% / 1.212% / 6.205% of the full image, including padding. These are supplied bounding-box areas, not lesion severity or exhaustive lesion extent. Small regions are particularly sensitive to resizing. Class counts overlap across images with several findings and cannot be added to obtain positive-image totals.

The sparsest test denominators are Pneumothorax (5 boxes) and Atelectasis (20). Their class estimates remain fragile despite the larger overall validation set. Observed test prevalence is 28.29%, whereas the operating scenario assumes 10.0%. Prevalence weighting adjusts class mix under stable conditional performance; it does not make this convenience sample representative of a real hospital or correct patient dependence or distribution shift.

## Finetuning and baselines

The genuine pretrained **yolo11n** detector was finetuned locally on **NVIDIA GeForce RTX 5080**, Windows-11-10.0.26200-SP0, using PyTorch 2.11.0+cu128 / CUDA 12.8 and Ultralytics 8.4.173. Detection supplies both finding boxes and an image score while retaining the published task; it does not establish that detector training is the best possible approach to image priority.

The run completed **30 epochs** out of a maximum 100, with patience 20; selected checkpoint epoch: **10**. Logged training-loop duration was 32.39 minutes, including per-epoch validation but excluding subsequent final-best validation and result generation. Epoch identity comes from actual on model save checkpoint selection event, not an arbitrary argmax tie.

Settings were image size 1024, batch 16, workers 4, seed 42, optimizer AdamW, initial learning rate 0.0005, deterministic mode True and AMP True. Explicit warmup bias learning rate was 0; nominal accumulation batch was 64. Geometry, flips, mosaic, mixing, copy-paste and hue/saturation transforms were disabled; brightness augmentation was hsv_v=0.1. Validation **mAP50-95 alone** selected the checkpoint in the installed implementation (fitness weights [0,0,0,1]); the checkpoint criterion did not use test metrics.

**Validation-only improvement history.** The bounded improvement study finished with **4/4 candidates** reaching complete validation artifacts. Failed, deferred or unstarted attempts retain their recorded status and have no substituted performance values. Its recommendation is **n1024_resolution**; the checkpoint actually bound to the current root results is **n1024_resolution**, identified by the saved-weight SHA-256. A recommendation is not an automatic model or policy adoption.

| Candidate | Checkpoint family | Input | Status | Completed epochs | Val mAP50-95 | Val hybrid EUR/1,000 | Val checks/day |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original_baseline | yolo11n.pt | 768 | original completed baseline | 19 | 0.082 | 3,184.45 | 40.6 |
| n768_adamw_control | yolo11n.pt | 768 | validation_ready | 32 | 0.090 | 2,842.21 | 29.7 |
| n1024_resolution | yolo11n.pt | 1024 | validation_ready | 30 | 0.091 | 2,711.86 | 27.0 |
| s1024_capacity | yolo11s.pt | 1024 | validation_ready | 36 | 0.085 | 3,364.30 | 17.2 |
| s1280_feature_scale | yolo11s.pt | 1280 | validation_ready | 35 | 0.087 | 3,105.53 | 17.4 |

Predeclared changes: n768_adamw_control: Corrected AdamW warmup_bias_lr=0; lr0=0.0005; Longer predeclared stopping budget | n1024_resolution: Same corrected protocol as n768; Use the published 1024-pixel image resolution | s1024_capacity: Same corrected protocol and 1024 resolution; Larger YOLO11s capacity | s1280_feature_scale: Conditional on remaining time and memory; Upscales 1024 source images; adds no source detail; Feature-scale experiment. Configured epoch caps are not completed epochs; time-budget stops and native early stopping are distinguished in each attempt's worker/training records. The 1280 setting, when attempted, upsamples the published 1024 images and adds no source detail.

Candidates are compared by standalone validation macro mAP50-95, with per-class performance and hypothetical cost/capacity tradeoffs retained. These are validation estimates, not additional test results. The fixed candidate set and a single seed do not establish a global optimum. Study cost columns use the recorded study reference: reviewer sensitivity/specificity 90.0%/90.0%, prevalence 10.0%, FP EUR 25 and FN EUR 500; these remain hypothetical.

**Test-exposure limitation:** the original baseline test results had already been examined before this improvement study was designed. Study workers used validation only and did not run new test inference. Any later evaluation on that same test partition is therefore reused held-out evidence, not a new blind final test. An independent external or newly reserved cohort is required for a stronger confirmation.

**Descriptive follow-up after adoption.** I chose the recommended model and its validation-selected boundaries before this checkpoint was evaluated. Both versions below use the same previously inspected test partition; this table did not select or retune the policy.

| Version | Test AP50 | Test mAP50-95 | Model-only EUR/1,000 | Hybrid EUR/1,000 | Hybrid checks/day |
| --- | --- | --- | --- | --- | --- |
| Original 768 | 0.1726 | 0.0800 | 5,080.23 | 2,692.22 | 40.5 |
| Adopted 1024 | 0.2005 | 0.0888 | 3,797.87 | 2,669.16 | 24.3 |

The validation hybrid-cost reduction did not transfer at the same magnitude: reused-test projected hybrid cost changed from EUR 2,692.22 to EUR 2,669.16, while review demand fell from 40.5 to 24.3 checks/day. Expected hybrid missed priorities are nearly unchanged (2.38 versus 2.36 per 1,000), while expected unnecessary priorities rise (43.45 versus 49.48). Reduced review labour offsets this under the hypothetical assumptions. 

| Same test images: model | AP50 | mAP50-95 | Image sensitivity | Image specificity |
| --- | --- | --- | --- | --- |
| Finetuned yolo11n | 0.200 | 0.089 | 97.4% | 88.9% |
| Pretrained COCO yolo11n; empty target mapping | 0.000 | 0.000 | 0.0% | 100.0% |
| Predict nothing / routine queue for all | 0.000 | 0.000 | 0.0% | 100.0% |

All three rows use the same 1,499 test images and target annotations. The finetuned image decision uses the validation-frozen binary threshold 0.04. The actual COCO model produced 76,495 raw detections at the cache floor; none of its 80 class names is an accepted clinical equivalent, so the explicit semantic mapping is empty. Numeric class IDs were never equated. Target AP/recall are analytically zero with no target predictions, and target precision is undefined. Predict-nothing has identical target outputs. Beating this limited baseline does not by itself demonstrate useful clinical zero-shot transfer or clinical utility.

The new model could be useful because fewer extra priority checks would leave staff more time for other clinical work. Every image would still receive clinical assessment. I would not claim that the model will necessarily reduce a hospital's overall costs: our projected saving is small and the comparison only includes missed priorities, unnecessary escalations, extra priority checks and inference. It does not include every cost of introducing and maintaining the system.

## Per-class evaluation and failure modes

| Finding | Test boxes | AP50 | mAP50-95 | P @ 0.04 | R @ 0.04 |
| --- | --- | --- | --- | --- | --- |
| Aortic enlargement | 301 | 0.819 | 0.408 | 0.408 | 0.927 |
| Atelectasis | 20 | 0.142 | 0.047 | undefined | 0.000 |
| Calcification | 42 | 0.113 | 0.035 | 0.167 | 0.095 |
| Cardiomegaly | 218 | 0.905 | 0.489 | 0.486 | 0.950 |
| Consolidation | 48 | 0.097 | 0.035 | undefined | 0.000 |
| ILD | 40 | 0.075 | 0.020 | 0.308 | 0.100 |
| Infiltration | 58 | 0.123 | 0.042 | 0.149 | 0.241 |
| Lung Opacity | 137 | 0.133 | 0.040 | 0.175 | 0.248 |
| Nodule-Mass | 69 | 0.179 | 0.056 | 0.157 | 0.261 |
| Other lesion | 118 | 0.021 | 0.006 | 0.000 | 0.000 |
| Pleural effusion | 117 | 0.125 | 0.044 | 0.091 | 0.470 |
| Pleural thickening | 206 | 0.033 | 0.009 | 0.056 | 0.049 |
| Pneumothorax | 5 | 0.000 | 0.000 | undefined | 0.000 |
| Pulmonary fibrosis | 166 | 0.042 | 0.011 | 0.044 | 0.253 |

The highest class mAP50-95 is Cardiomegaly (0.489); the lowest is Pneumothorax (0.000). Mean AP averages represented classes equally, so abundant-class counts do not directly weight the headline average, although they affect estimation support. Fixed P/R use confidence ≥ 0.04, IoU ≥ 0.50 and score-ordered one-to-one matching against published boxes; zero prediction denominators are undefined.

Native mean P/R are 0.408 / 0.222 at the test split's descriptive maximum smoothed mean-F1 point (confidence 0.0781). This is distinct from cost-selected image routing and cannot change the frozen policy. Native AP/matching and fixed cached-prediction metrics are labeled separately.

| Observed image truth / prediction | Routine | Accelerated |
| --- | --- | --- |
| No listed finding | 956 | 119 |
| At least one listed finding | 11 | 413 |

The object matrix has 2,506 unmatched predictions and 757 unmatched annotations under its separate class-agnostic geometric matching. Largest nonzero class-confusion cells (annotated → predicted) were Pleural thickening → Pleural effusion: 19; Lung Opacity → Pulmonary fibrosis: 11; Consolidation → Lung Opacity: 9; Other lesion → Pulmonary fibrosis: 8. Its background row/column encode unmatched objects; there is no enumerable true-negative box count. [Full object confusion](results/evidence_figures/observed_object_confusion.png) and [image confusion](results/evidence_figures/observed_image_confusion.png) retain orientation and denominators.

The declared deterministic example rule found 119 image false-positive candidates and 11 image false-negative candidates. It selects up to two highest-score cases per category, using image ID for ties; misses are the highest scores below the frozen binary threshold. These are annotation-defined examples, not a random sample or adjudicated clinical errors.

| Saved example | Image score | Published annotations |
| --- | --- | --- |
| false positive 1 | 0.6735 | None of the listed findings |
| false positive 2 | 0.6105 | None of the listed findings |
| false negative 1 | 0.0352 | Pulmonary fibrosis, Consolidation, ILD, Atelectasis, Other lesion |
| false negative 2 | 0.0351 | ILD, Other lesion, Pulmonary fibrosis, Infiltration, Pleural thickening |

A positive image can be prioritized through one finding while another box is missed. Conversely, a plausible region omitted from representative annotations can be counted as an unmatched prediction. These figures cannot independently establish clinical truth or lesion completeness.

## Cost matrix and operating point

| Annotated truth / priority decision | Routine | Accelerated |
| --- | --- | --- |
| No listed finding | EUR 0 | EUR 25 |
| Listed finding | EUR 500 | EUR 0 |

These penalties are the hypothetical values justified above. Extra human review costs 2/60 × EUR 60 = **EUR 2.00** per checked image. Human sensitivity/specificity are assumed 90.0% / 90.0%, including difficult model-selected cases. Model inference adds EUR 0.01 per image; the scenario uses prevalence 10.0%. None of these service quantities was measured clinically.

Reviewer performance is the assumption I would most want to verify. The adopted reference is 90% sensitivity and 90% specificity on the images sent for an extra check; these rates were assumed, not measured.

Expected cost per 1,000 = expected FP × FP penalty + expected FN × FN penalty + expected reviews × review cost + model inference allowance. Each class-conditional action rate is multiplied by its assumed population share before costs are added. This retains negative-image denominators and avoids treating the test prevalence as the operating base rate.

The image score is maximum target-box confidence, or zero for no target prediction. Caches use floor 0.001, NMS IoU 0.7, maximum 300 detections and image size 1024; standard prediction uses rect=False, half=False, augment=False. Threshold changes reuse this low-floor cache. Confidence is not a calibrated probability.

The validation search grid is 0–1 in steps of 0.01, plus 1.001 for a genuine above-score endpoint. Binary policies minimize expected cost; hybrid policies must also meet expected review capacity. Ties use cost, fewer false negatives, less review, then lower thresholds. I adopted these thresholds after comparing the validation results. The policy was frozen at **2026-10-26T08:58:49.105350+00:00**, before test inference, with bindings to weights, validation predictions, the full dataset manifest, configuration and scenario.

| Frozen policy | Thresholds | Validation EUR/1,000 | Test-projected EUR/1,000 |
| --- | --- | --- | --- |
| Model only | t = 0.04 | 4,029.61 | 3,797.87 |
| Model + human | L = 0.02; U = 0.15 | 2,711.86 | 2,669.16 |

These are hypothetical costs inferred from observed routing, not observed expenditure or clinical benefit. The [validation cost sweep](results/validation_binary_cost_sweep.csv) records the search; the test set assesses the frozen choice rather than choosing another threshold. Validation was also used for early stopping, so threshold-selection overfitting remains possible.

| Assumption varied | Tested values | Frozen hybrid EUR/1,000 range | Checks/day range |
| --- | --- | --- | --- |
| Assumed target prevalence | 0.05, 0.1, 0.2 | 2,148.65–3,710.17 | 24.2–24.3 |
| FN/FP penalty ratio (FP EUR25) | 5, 20, 50 | 1,784.72–4,438.02 | 24.3–24.3 |
| Assumed reviewer sensitivity and specificity | 0.8, 0.9, 0.95, 1 | 1,805.35–3,532.96 | 24.3–24.3 |
| Minutes per priority review | 1, 2, 4 | 2,547.69–2,912.09 | 24.3–24.3 |
| Review hours per available person per day | 1, 2, 4 | 2,669.16–2,669.16 | 24.3–24.3 |
| Inference allowance (EUR per image) | 0, 0.01, 0.05 | 2,659.16–2,709.16 | 24.3–24.3 |

Within these chosen one-factor ranges, **FN/FP penalty ratio (FP EUR25)** produces the largest frozen-hybrid cost span (EUR 2,653.30 per 1,000). This depends on the tested ranges and is not a universal ranking of clinical importance. Reviewer accuracy on uncertain cases remains unmeasured. [Full sensitivity evidence](results/sensitivity_analysis.json) separately reports validation-only reselection without held-out retuning or adoption of those exploratory alternatives.

## What a person still has to check

With **L = 0.02, U = 0.15**, score < L goes to routine clinical review, L ≤ score < U receives an extra human priority check, and score ≥ U receives accelerated clinical review. The human checks the image and clinical context for priority; this prototype supplies neither history nor a demonstrated urgency outcome. All images remain in clinical care. L = U removes the review band; 1.001 is an above-score endpoint, and a binary threshold of zero flags even images with no detections.

| Observed held-out annotation class | Routine | Extra human check | Accelerated |
| --- | --- | --- | --- |
| Negative | 898 | 131 | 46 |
| Positive | 5 | 50 | 369 |

Observed test routing sends **181/1,499 images (12.1%)** to the review band. At the assumed 10.0% operating prevalence, the projected share is **12.1%**. These percentages answer different questions and are not interchangeable.

At 200 images/day and 2 minutes/check, the frozen hybrid policy projects **24.29 checks/day**, **0.81 staff hours/day** and **1 person(s)** with 2 assigned hours each. Available staffing is 1 person(s), or 60 checks/day. This test-based expectation is within assumed capacity; projected backlog without extra staffing is 0.00 checks/day. Validation feasibility does not guarantee test or daily queue feasibility.

| Operating alternative | EUR/1,000 | Checks/day | Hours/day | People needed | Within capacity |
| --- | --- | --- | --- | --- | --- |
| Routine queue for all (no added triage) | 50,000.00 | 0.0 | 0.00 | 0 | Yes |
| Human priority review for all | 9,250.00 | 200.0 | 6.67 | 4 | No |
| Model only | 3,797.87 | 0.0 | 0.00 | 0 | Yes |
| Model + human review | 2,669.16 | 24.3 | 0.81 | 1 | Yes |

After assumed review, the hybrid policy projects **49.48 false priorities** and **2.36 missed priorities** per 1,000 operating images. Its cost comprises EUR 2,416.22 error penalties, EUR 242.93 additional review and EUR 10.00 inference. Human decisions and final outcomes were not observed. The [worked arithmetic](results/evidence_figures/result_evidence.json) preserves observed cells, prevalence rescaling, residual-error formulas and staffing calculation.

Review-all remains visible even when it exceeds capacity. The required COCO/predict-nothing model baselines are distinct from these four operating alternatives. No automatic withholding of clinical assessment is proposed.

## Limitations & next steps

| Observed methodological limitation | Concrete next step |
| --- | --- |
| Priority proxy rather than observed urgency | Define and label the actual clinical/workflow decision with qualified reviewers before evaluating that use. |
| Representative boxes omit source regions | Recover appropriate complete/adjudicated annotations; keep localization and image-presence claims separate. |
| Patient independence and one related-image pair unresolved | Obtain verified patient/visit grouping, resolve related acquisitions and perform grouped and external evaluation. |
| Sparse classes and validation reuse | Collect more independent rare-class cases; use an additional development or external cohort for policy confirmation. |
| Assumed reviewer accuracy, time and penalties | Measure the actual uncertain-case review process, queue capacity and operating consequences before making staffing claims. |
| Uncalibrated confidence and possible prevalence/domain shift | Assess calibration and conditional performance in the intended population; monitor drift without retuning on the held-out study test. |

For the frozen binary rule, image sensitivity is 97.4% (marginal Wilson 95% interval 95.4%–98.5%; n=424) and specificity is 88.9% (86.9%–90.7%; n=1075). These are image-level binomial intervals conditional on the chosen rule. They omit selection and label uncertainty and may be too narrow when images are related. No interval treats assumed human decisions as observed trials.

I do not consider this model ready for real hospital use. It would need much more work, resources and validation before that could be considered. For now, I would finish and demonstrate it as a class assignment prototype. Reviewer performance is an assumption I would want to verify rather than take for granted.

Implementation and assumptions are documented in the [method companion](docs/METHOD_AND_ASSUMPTIONS.md); [local run instructions](RUN_LOCAL.md) explain the saved-artifact dashboard and reproducible pipeline. A separate local illustrated study PDF provides detailed arithmetic and explanations. It is not bundled in the public package because it contains source X-ray images. The dashboard can retrieve three verified examples directly from Roboflow after normal key setup; see the local run instructions.

## Generative AI use disclosure

I used OpenAI Codex to assist with dataset-access and audit code, Windows setup, the training and evaluation pipeline. It also helped explain unfamiliar concepts and produced earlier report drafts. I selected the problem, dataset and operating assumptions, adopted the model and review boundaries after explanation, and supplied the reasoning used in the decision paragraphs.
