# Assignment 2 — local submission copy

Javier Sánchez Esquivel · 19731 · BBADBA 5B

This is an educational chest X-ray review-priority prototype. It is not a diagnostic system or a hospital deployment. Every image still receives clinical assessment.

## Review and run

1. Extract this ZIP and open a terminal in `daiist-assignment-2`.
2. Follow `SETUP.md` for uv installation if needed, then run `uv sync --frozen`.
3. Run `uv run python main.py app` and open `http://127.0.0.1:7860/`.
4. Upload an image, or set your own Roboflow key through the normal assignment setup and click **Load three examples from Roboflow**. Then click **Run model on this image** for real inference.
5. Read `REPORT.md` for the findings, assumptions, limitations and AI disclosure. `RUN_LOCAL.md` explains setup, examples, verification and retraining.

The app opens from the included weights and results without training, a dataset download or a key. Only the optional example download needs a key and internet. `uv run python check.py` is the unchanged original checker and separately requires a valid Roboflow key.

`uv run python main.py train` performs a complete new training/evaluation run on Windows CUDA. It takes real GPU time and is unnecessary to inspect the submitted model. A separate unattended run was verified; it did not replace the adopted submission checkpoint. The saved report describes that adopted checkpoint, not an arbitrary future rerun.

## Contents and boundaries

Included: source code, locked dependencies, the genuine adopted `weights/best.pt`, prediction caches, numerical results and charts, source-audit records, tests and verification evidence. `PACKAGE_MANIFEST.json` lists packaged files and their SHA-256 hashes.

Excluded: API keys, datasets, environments, experiment archives, raw chat/learning responses, copied X-ray photographs and image-containing previews. The illustrated study PDF remains a separate local companion. The example button retrieves three audited validation images from their source and checks their hashes; it does not establish redistribution permission or demonstrate representative accuracy. Source availability and access conditions still apply.

The test partition was inspected for the earlier baseline before the improvement study. The report therefore describes the adopted model's test results as reused held-out evidence, not a new blind evaluation. Reviewer performance and operational costs are assumptions, not measured hospital outcomes.

This ZIP was prepared locally. No fork, push, pull request, Blackboard submission or publication has been performed. The original assignment describes those separate submission steps; the ZIP alone does not complete them.
