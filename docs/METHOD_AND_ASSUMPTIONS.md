# Adopted method and hypothetical operating assumptions

I adopted the dataset, target and operating assumptions below. The final model is YOLO11n at 1024 pixels: 30 completed epochs, selected checkpoint epoch 10. I adopted the recommended overnight candidate and its validation-selected boundaries (binary 0.04; hybrid 0.02/0.15). Evaluation is complete. The test partition had already been examined for the original model, so this is a disclosed follow-up evaluation, not a new blind test. `project_config.json`, `scenario.json` and `results/frozen_policy.json` bind the final choice.

## Intended use and target

The hypothetical operator is a hospital radiology image-review team. The unit is one chest X-ray image, not a verified patient or complete examination. The model helps set review priority; every case still receives clinical assessment. The dataset has finding labels, not observed urgency labels, so the target is an educational proxy for priority rather than a validated urgency diagnosis.

The chosen public export is [VinBigData chest-xray-symptom-detection, version 3](https://universe.roboflow.com/vinbigdataxrayproject/chest-xray-symptom-detection/dataset/3), workspace `vinbigdataxrayproject`. Its published train/validation/test splits are preserved.

| Split | Images | Annotated positive images | Explicit negative images | Boxes |
|---|---:|---:|---:|---:|
| Train | 10,499 | 3,051 | 7,448 | 10,648 |
| Validation | 3,000 | 919 | 2,081 | 3,172 |
| Test | 1,499 | 424 | 1,075 | 1,545 |

An image is positive if it has at least one annotation belonging to any of these 14 classes, preserving the export's class order:

| ID | Finding | ID | Finding |
|---:|---|---:|---|
| 0 | Aortic enlargement | 7 | Lung Opacity |
| 1 | Atelectasis | 8 | Nodule-Mass |
| 2 | Calcification | 9 | Other lesion |
| 3 | Cardiomegaly | 10 | Pleural effusion |
| 4 | Consolidation | 11 | Pleural thickening |
| 5 | ILD | 12 | Pneumothorax |
| 6 | Infiltration | 13 | Pulmonary fibrosis |

A negative image has none of these 14 annotations. Source labels corroborate these as `No finding`; this does not mean general health or clinical clearance. Missing label files are errors, not negatives.

The pre-training audit found no malformed labels, exact decoded-pixel overlap, or repeated source filename groups across splits. Pixel and label reconciliation precedes training, and byte hashes bind later evaluation to the same files. Accepted limitations remain: the export retains a representative source-reader box per present class and may omit additional same-class regions; source IDs identify images, not verified patients; one highly similar train/validation pair remains unresolved as possibly related acquisitions after the reviewed top 20 similarity pairs; and source CSV corroboration uses a public mirror rather than an authenticated official download. Neither hashes nor the review establish patient independence. Audit evidence is in `evidence/vinbigdata-v3/`; the final manifest preserves these caveats.

## Training and checkpoint selection

The run uses the genuine pretrained COCO YOLO11n checkpoint on this Windows RTX 5080 PC. CUDA matrix multiplication, NMS and convolution passed before training. The approved settings are up to 100 epochs, early-stopping patience 20, image size 1024, batch 16, four workers, seed 42, deterministic mode, AdamW and initial learning rate 0.0005, explicit bias warm-up learning rate 0.0 and nominal accumulated batch 64. AMP is enabled. Mosaic, mixup, CutMix, copy-paste, flips, rotation, translation, scaling, shear, perspective, hue/saturation changes and RGB/BGR swapping are disabled; only small brightness augmentation (`hsv_v=0.1`) is retained. These are training transforms, not changes to the published images or splits.

**Installed Ultralytics 8.4.173 selects detection checkpoints by validation mAP50-95 only.** Its `Metric.fitness()` uses weights `[0, 0, 0, 1]` on `[precision, recall, mAP50, mAP50-95]`. It does not use the older 0.1 mAP50 + 0.9 mAP50-95 mixture. `BaseTrainer.validate()` updates the best fitness on improvement; `save_model()` writes `best.pt` when current fitness equals the best, so an equal-fitness later checkpoint can replace an earlier one. Early stopping uses the same validation fitness. This criterion selects weights; it does not select a cost threshold and does not use test results.

This was checked directly in the installed `ultralytics/utils/metrics.py` (`Metric.fitness`) and `ultralytics/engine/trainer.py` (`validate`, `save_model`, `final_eval`) on 26 October 2026. Finalization strips checkpoint epoch metadata and copies the complete training history into best.pt; a best-epoch summary must therefore preserve the actual checkpoint identity rather than assume the last history row or the first maximum is the selected epoch. Logs and provenance record the actual installed version and completed run.

## Prediction and action rules

Saved validation and test predictions use confidence floor **0.001**, NMS IoU **0.7**, at most **300** detections per image and input size **1024**. Standard prediction uses `rect=False`, `half=False`, `augment=False`; uploaded-image inference follows the same settings. The image score is the maximum confidence among predictions for the 14 positive classes, or zero if there are none. Confidence is an uncalibrated score, not a disease probability.

All published images remain in the cache, including those with no labels or no predictions. Operating thresholds are applied to this same low-floor cache; results below a higher prediction floor are never invented.

- Score < L: routine clinical queue.
- L <= score < U: an extra human priority check.
- Score >= U: accelerated clinical review.

L = U is a model-only binary policy. The endpoint 1.001 is deliberately above every possible score: it permits flag-nothing or review-all policies even if a score equals 1. It is not a probability. At L = U = 0, every image is accelerated, including images with no detections.

## Adopted hypothetical scenario

| Quantity | Adopted value | Interpretation |
|---|---:|---|
| Operating positive prevalence | 10% | Hypothetical incoming image mix, separate from observed dataset prevalence |
| Unnecessary priority FP penalty | EUR 25 | Incremental priority-error penalty |
| Missed priority FN penalty | EUR 500 | Incremental priority-error penalty; 20 times FP |
| Extra priority-check time | 2 minutes | Assumed time for the additional review |
| Reviewer hourly cost | EUR 60 | Hypothetical staff cost |
| Reviewer sensitivity / specificity | 90% / 90% | Unmeasured assumptions, also on the model's uncertain cases |
| Workload | 200 images/day | Hypothetical service workload |
| Available extra-review staffing | One person, two hours/day | 60 extra checks/day; 30% of daily images |
| Inference allowance | EUR 0.01/image | Hypothetical local compute allowance, not an API fee |
| TP/TN incremental error cost | EUR 0 | Shared routine clinical assessment is excluded equally, not claimed to be free |

I chose the 20:1 penalty ratio because overlooking a potentially dangerous finding could delay attention and make a problem harder or more expensive to address later. An unnecessary priority mainly creates additional staff work. These are assumed hospital-related costs, informed by an informal discussion with a relative in hospital management, not measured hospital costs or a universal rule about clinical harm. The EUR 25 downstream escalation cost is separate from the EUR 2 initial priority check. All images still receive clinical assessment.

## Cost and staffing calculations

Estimate action rates separately among annotated negative and positive images. Weight negative rates by 1-pi and positive rates by pi, where pi is the assumed operating prevalence. This adjusts class mix under stable conditional performance; it does not correct general distribution shift.

After applying the assumed reviewer sensitivity and specificity to the review band:

`Cost per 1,000 = 1,000 × [(1-pi) × FPR × C_FP + pi × FNR × C_FN + review_fraction × review_cost + inference_cost]`.

Review cost is `2/60 × 60 = EUR 2`. Inference is included only for policies using the model. Reviewed negatives can become false priorities according to assumed reviewer specificity; reviewed positives can remain missed priorities according to assumed reviewer sensitivity. These final confusion counts are expected scenario quantities and can be decimals. Observed image routing is an integer 2-by-3 matrix and is reported separately.

Daily checks equal daily images times the prevalence-weighted review fraction. Daily staff hours equal checks times minutes per check divided by 60. People needed are the ceiling of staff hours divided by hours assigned per person. Do not round expected case counts before computing costs. Capacity is an expected-workload constraint, not a guarantee that each day's arrivals fit.

For scale, reviewing every image requires 200 checks and 6.67 staff hours/day under these assumptions, exceeding the assigned 60 checks/two hours. This is scenario arithmetic, not a model result. The comparison reports that infeasibility rather than dropping unreviewed cases.

## Validation freeze and held-out evaluation

Training, early stopping and checkpoint selection use train/validation only. Proposed operating policies are selected on validation with a grid from 0 through 1 in steps of 0.01 plus the 1.001 endpoint. The binary minimum minimizes expected cost; the two-threshold minimum additionally respects expected daily review capacity. Ties use cost, then fewer expected false negatives, then less human review, then lower L and U. This is a grid minimum, not a continuous or guaranteed clinical optimum.

The current `scenario.policy_rule_approved` is true. I retained assumed 90% reviewer sensitivity/specificity and explicitly adopted the recommended candidate's validation-grid boundaries: binary 0.04 and hybrid lower 0.02 / upper 0.15. These are recorded in `adopted_thresholds`. The final freeze was saved at `2026-10-26T08:58:49.105350+00:00`, before this checkpoint's test inference, with SHA-256 bindings to the cache, weights, configuration, scenario and manifest. The freeze and evaluation both disclose prior test exposure from the original baseline. The candidate configuration is preserved byte-for-byte from the validation-only study; its historical `test_inference_authorised=false` is superseded for this adopted checkpoint by the later my recorded adoption in the scenario. Test results never selected the replacement or its boundaries. Repeated use of validation can still overfit it.

For the adopted 1024-pixel model and 90% reviewer scenario, validation-based projected costs are EUR 4,029.61 per 1,000 for the model-only policy and EUR 2,711.86 for the combined workflow, with about 27.0 extra checks/day. These are the final validation figures in `results/frozen_policy.json`. Earlier baseline and reviewer scenarios are historical; the final report compares the relevant model versions explicitly. Reviewer performance remains unmeasured.

The finetuned model, genuine pretrained COCO model and predict-nothing baseline use identical held-out image IDs and target annotations. COCO has no semantically equivalent class among these 14 chest findings. Its original 80-class predictions are actually computed and retained, then an explicit empty mapping yields no target detections. Numeric IDs are never equated. For represented target classes, no target predictions imply AP and recall zero and precision undefined; this is a semantic limitation of the baseline, not evidence of useful medical zero-shot ability. Predict-nothing has the same target outputs and avoids the model inference allowance.

Report native object AP50 and mAP50-95 separately from image priority performance. Native detector precision/recall use the split's maximum smoothed mean-F1 point at IoU 0.50; the descriptive test maximum-F1 point must not change the frozen triage thresholds. Fixed-threshold object metrics are computed separately from standard cached predictions at the validation-frozen binary threshold. Native and fixed matching implementations can differ. Object confusion matrices include unmatched-background rows/columns and have no meaningful true-negative box count; image confusion matrices count images. Localization metrics assess the published representative boxes, not exhaustive lesion detection.

Operational alternatives are routine queue for all, human priority review for all, model only, and model plus human review. They answer a staffing/cost question distinct from the three required model baselines.

## Sensitivity and uncertainty

The completed one-factor scenario stress tests use prevalence 5/10/20%; FN:FP ratio 5/20/50 with FP EUR 25; paired human sensitivity/specificity 80/90/95/100%; review duration 1/2/4 minutes; and assigned staff time 1/2/4 hours/day. These are hypothetical variations of the adopted assumptions. They show the frozen policy's projected test costs and a separately labeled validation-only reselection. They do not optimize on test or adopt a new policy from test results.

Report small per-class test counts, including only five pneumothorax boxes, alongside the relevant metrics. Image-level Wilson intervals describe binomial uncertainty conditional on the observed images; unproved patient independence prevents interpreting them as patient-level certainty. No modeled cost reduction, assumption about reviewers or dataset result establishes readiness for clinical deployment.
