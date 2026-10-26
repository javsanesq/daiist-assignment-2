# Run this project on Windows

Extract the submission ZIP, then open PowerShell in its `daiist-assignment-2` folder. The original `README.md` and `SETUP.md` remain assignment source material. This file describes the implemented local workflow.

Install uv as described in SETUP.md. If it was installed in the usual Windows location but is not on PATH, open a new terminal or set a session-only alias:

```powershell
Set-Alias uv "$env:USERPROFILE\.local\bin\uv.exe"
```

Equivalently, use the full executable path, for example:

```powershell
& "$env:USERPROFILE\.local\bin\uv.exe" run python main.py app
```

## Environment and local dashboard

If dependencies are not already present, install the exact lockfile environment:

```powershell
uv sync --frozen
```

The adopted 1024-pixel model, frozen policy and follow-up evaluation are complete. To open the current project:

```powershell
uv run python main.py app
```

The dashboard reads saved weights, configuration, manifest and completed result artifacts. It does not train, download the dataset or require a Roboflow API key. Raw dataset images may be absent. Uploaded-image prediction runs locally; the held-out evaluation and interactive validation analysis use the saved caches. The app checks artifact hashes and will explain missing or inconsistent outputs rather than display fabricated results. It is a local educational app; it does not publish a public service.

## Ready-made examples for marking

If examples are not already cached, click **Load three examples from Roboflow** in the dashboard. This optional action uses your own Roboflow key from the normal assignment setup and an internet connection. It retrieves only the three selected validation images from the published export (about 5.3 MB including ZIP metadata), verifies their recorded SHA-256 hashes, and stores them locally under `assets/demo/`. It does not retrain or alter the results. Select an example to inspect its saved prediction, then click **Run model on this image** to perform real inference. The examples illustrate the three routes; they are not a representative test sample.

The public package can omit copied X-ray photographs while retaining this route to obtaining examples from their source. Source access and licence conditions still apply; this mechanism does not establish redistribution rights. The dashboard continues to work with uploaded images if the source or a key is unavailable.

## Optional staged run for a new substantive decision

I used a staged validation run to review the original policy decision. This optional mode remains available when investigating a different model or policy; it is not the default submission command. A CUDA check can be run before training:

```powershell
uv run python tools/windows_preflight.py
$env:ASSIGNMENT_PHASE = 'train_validation'
uv run python main.py train
```

This checks the approved data, trains the real model and saves validation predictions, metrics and an unfrozen policy proposal. It does not infer on test images. Review the validation findings, record the selected policy in `scenario.json`, and set `policy_rule_approved` only after reviewing that decision. Explicit thresholds, when adopted, use `adopted_thresholds = {"binary_threshold": t, "lower": L, "upper": U}`; otherwise leave it null to use the approved validation-grid rule.

Continue using the genuine saved weights, without retraining:

```powershell
$env:ASSIGNMENT_PHASE = 'evaluate_frozen'
uv run python main.py train
Remove-Item Env:ASSIGNMENT_PHASE
```

Continuation verifies the saved bindings and unchanged source data, writes the frozen decision before test inference, evaluates the finetuned and pretrained models, and creates the final baseline, cost, sensitivity and plotting artifacts. A changed configuration, scenario, validation cache, manifest or checkpoint is an error, not an invitation to retune on test.

## A new complete training run

The normal assignment entry point reruns the already approved experiment from training through evaluation without a new prompt. It records configuration and scenario hashes in `results/full_run_authorization.json` before training. Training provenance and the subsequent policy freeze are bound to that receipt. The existing fixed thresholds and cost assumptions remain unchanged; the old study-checkpoint adoption is retained as historical evidence rather than misapplied to a newly trained file.

```powershell
Remove-Item Env:ASSIGNMENT_PHASE -ErrorAction SilentlyContinue
uv run python main.py train
```

This starts a new training run. Existing generated results and `weights/best.pt` are preserved under a unique `history/run_.../` folder before replacement. Source images, annotations and published splits are not changed. Existing run logs remain under `runs/`. Training requires Windows CUDA device 0 and does not silently fall back to CPU. After evaluation, local demonstration examples are refreshed for the new checkpoint so the app can open immediately. No test result selects thresholds or triggers model replacement in the original project.

The optional staged mode still requires an explicit checkpoint decision for a new study candidate. Changed files, invalid data, unavailable CUDA or violated capacity constraints produce an error; unattended execution does not mean silently ignoring those failures. A report describing an earlier experiment remains historical after deliberately retraining and must not be presented as the new run's results.

When the exact approved dataset version is absent, training downloads it through `api_key.load_api_key()`. The private key may be supplied by the environment or local `.env`; never paste it into documentation or commit it. To set it privately when needed:

```powershell
uv run python tools/setup_key.py
```

The approved audit evidence is still required after a fresh download. A partial dataset directory is rejected rather than overwritten. The initial dataset selection, split evidence and accepted caveats are in `project_config.json` and `docs/METHOD_AND_ASSUMPTIONS.md`.

## Checks and boundaries

The pipeline records factual epoch/duration metadata in `results/training_summary.json`. For a completed run started before that helper was added, generate it from the genuine saved checkpoint and training CSV without inference or retraining:

```powershell
uv run python tools/extract_training_summary.py
```

The helper loads only the provenance-bound local checkpoint on CPU. Ambiguous best-epoch evidence remains unknown rather than choosing an arbitrary tied maximum.

```powershell
uv run python -m pytest -q
uv run python check.py
```

Tests use small synthetic cases and do not train a model. `check.py` remains the original assignment checker. `main.py`, the original assignment files and the dependency lockfile have not been weakened for a passing check. The commands above run locally and do not submit or publish the project.
