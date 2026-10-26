"""Exercise the real saved-results dashboard on a dedicated local port.

Requires completed genuine evaluation artifacts. Launches exactly main.py app,
with no Roboflow key in its environment, and stops only its own process tree.
Uses real validation predictions and one manifest-bound validation image.
This is functional/API verification; visual inspection remains a separate check.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from pipeline import local_path, read_json, sha256_file, write_json
from triage import evaluate_policy


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def plot_data(value):
    if isinstance(value, str):
        value = json.loads(value)
    require(isinstance(value, dict), "Plot callback did not return a serialized plot.")
    if "plot" in value:
        value = value["plot"]
        value = json.loads(value) if isinstance(value, str) else value
    require("data" in value, "Plot callback lacks figure data.")
    return value


def numeric_array(value):
    if isinstance(value, dict) and "bdata" in value:
        array = np.frombuffer(base64.b64decode(value["bdata"]), dtype=value["dtype"])
        shape = value.get("shape")
        if shape:
            array = array.reshape(tuple(int(n.strip()) for n in shape.split(",")))
        return array
    return np.asarray(value)


def verify_policy_response(response, supplied_lower, supplied_upper, artifacts, api_name):
    require(isinstance(response, (list, tuple)) and len(response) == 6, "Slider callback returned the wrong output count.")
    returned, cards, note, observed, expected, curve = response
    require(api_name in {"/change_lower", "/change_upper"}, "Unknown slider callback.")
    lower, upper = float(supplied_lower), float(supplied_upper)
    crossed = lower > upper
    if crossed:
        adjusted_value = lower if api_name == "/change_lower" else upper
        require(isinstance(returned, (int, float)) and abs(float(returned) - adjusted_value) < 1e-12,
                "Crossed slider was not adjusted consistently.")
        lower = upper = adjusted_value
    else:
        require(returned == {"__type__": "update"},
                "A noncrossing callback must skip the opposite slider to avoid restoring a stale value.")
    _, labels, scores = artifacts["val_data"]
    result = evaluate_policy(labels, scores, lower, upper, artifacts["scenario"])
    observed_array = numeric_array(plot_data(observed)["data"][0]["z"])
    expected_array = numeric_array(plot_data(expected)["data"][0]["z"])
    require(np.array_equal(observed_array, result["observed_actions"]), "Displayed observed routing differs from validation counts.")
    require(np.allclose(expected_array, result["expected_resolved_confusion_per_1000"], rtol=0, atol=1e-8),
            "Displayed expected confusion differs from prevalence/reviewer calculation.")
    card_text = html.unescape(str(cards))
    require(f"{result['cost_per_1000']:,.2f}" in card_text, "Displayed cost did not update to the expected value.")
    require(f"{result['review_fraction']:.1%}" in card_text, "Displayed review share did not update.")
    require(f"{result['daily_reviews']:.1f} / {artifacts['scenario'].daily_capacity:.0f}" in card_text,
            "Displayed daily demand/capacity differs.")
    capacity_text = "Within assumed capacity" if result["within_capacity"] else "Exceeds assumed capacity"
    require(capacity_text in note, "Displayed capacity status differs.")
    figure = plot_data(curve)
    require(len(figure["data"]) >= 2, "Validation curve or frozen marker is missing.")
    sweep = artifacts["selected"]["binary_sweep"]
    require(np.allclose(numeric_array(figure["data"][0]["x"]), [r["lower"] for r in sweep]), "Curve thresholds differ from frozen validation sweep.")
    require(np.allclose(numeric_array(figure["data"][0]["y"]), [r["cost_per_1000"] for r in sweep]), "Curve costs differ from saved validation sweep.")
    require(float(figure["data"][1]["x"][0]) == artifacts["selected"]["binary"]["lower"], "Frozen binary marker moved.")
    shapes = figure.get("layout", {}).get("shapes", [])
    require(len(shapes) == 2 and [float(s["x0"]) for s in shapes] == [lower, upper], "Current curve threshold markers did not update.")
    return {"lower": lower, "upper": upper, "observed_actions": result["observed_actions"],
            "cost_per_1000": result["cost_per_1000"], "review_fraction": result["review_fraction"],
            "within_capacity": result["within_capacity"], "passed": True}


def stop_process_tree(process):
    if process is None or process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)],
                       capture_output=True, check=False, creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15)


def verify(root=ROOT, port=7861, output=None):
    root = Path(root).resolve()
    # Validate completed files before opening a port, log or output report.
    from app import load_dashboard_artifacts
    artifacts = load_dashboard_artifacts(root)
    evaluation_path = root / "results/evaluation_provenance.json"
    evaluation = read_json(evaluation_path)
    protected = {"results/evaluation_provenance.json": sha256_file(evaluation_path),
                 "results/frozen_policy.json": sha256_file(root / "results/frozen_policy.json"),
                 "scenario.json": sha256_file(local_path(artifacts["config"]["scenario_path"], root)),
                 "project_config.json": sha256_file(root / "project_config.json"),
                 "weights/best.pt": sha256_file(root / "weights/best.pt"),
                 "results/validation_predictions.json": sha256_file(root / "results/validation_predictions.json"),
                 **{"results/" + name: value for name, value in evaluation["artifact_sha256"].items()}}
    with socket.socket() as probe:
        require(probe.connect_ex(("127.0.0.1", port)) != 0, f"Port {port} is occupied; refusing to test another process.")
    manifest = read_json(root / "results/dataset_manifest.json")
    rows = {r["image_id"]: r for r in manifest["splits"]["val"]}
    ids, labels, scores = artifacts["val_data"]
    # A deterministic real positive validation image; selection is not a performance sample.
    index = sorted((i for i, y in enumerate(labels) if y), key=lambda i: (-float(scores[i]), ids[i]))[0]
    row = rows[ids[index]]
    dataset_root = local_path(artifacts["config"]["dataset"]["root"], root).resolve()
    image_path = (dataset_root / row["image"]).resolve()
    require(image_path.is_relative_to(dataset_root), "Validation image path escapes the approved data root.")
    require(sha256_file(image_path) == row["image_sha256"], "Validation upload image changed since the saved manifest.")
    output = Path(output) if output else root / "evidence/dashboard_integration.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.with_suffix(".log")
    downloads = output.parent / "dashboard_integration_files"
    environment = {k: v for k, v in os.environ.items() if k.upper() != "ROBOFLOW_API_KEY"}
    environment.update(APP_PORT=str(port), GRADIO_ANALYTICS_ENABLED="False", HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
    command = [sys.executable, str(root / "main.py"), "app"]
    report = {"schema_version": 1, "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "command": command, "port": port, "api_key_environment_unset": True,
              "launches_exact_main_app_entrypoint": True, "visual_review_performed": False,
              "frozen_policy_sha256": protected["results/frozen_policy.json"],
              "evaluation_provenance_sha256": protected["results/evaluation_provenance.json"],
              "checks": [], "log": str(log_path)}
    process = None
    try:
        flags = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, **flags)
            started = time.monotonic()
            url = f"http://127.0.0.1:{port}"
            while True:
                require(process.poll() is None, "App exited before its HTTP endpoint became available; inspect the local log.")
                try:
                    with urllib.request.urlopen(url, timeout=3) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, OSError, TimeoutError):
                    pass
                require(time.monotonic() - started < 120, "App did not respond within the original checker's 120-second startup window.")
                time.sleep(.5)
            report["startup_seconds"] = round(time.monotonic() - started, 3)
            from gradio_client import Client, handle_file, __version__
            # token=False tells the underlying HF header builder not to read/send a stored token.
            client = Client(url, token=False, verbose=False, analytics_enabled=False,
                            download_files=downloads, httpx_kwargs={"timeout": 30.0})
            report["gradio_client_version"] = __version__
            selected = artifacts["selected"]["hybrid"]
            requests = [
                ("all_accelerated", "/change_lower", 0., 0.),
                ("all_routine", "/change_lower", 1.001, 1.001),
                ("all_review", "/change_upper", 0., 1.001),
                ("lower_crosses_upper", "/change_lower", .9, .2),
                ("upper_crosses_lower", "/change_upper", .9, .2),
                ("frozen_hybrid", "/change_lower", selected["lower"], selected["upper"]),
            ]
            for name, endpoint, supplied_l, supplied_u in requests:
                response = client.submit(supplied_l, supplied_u, api_name=endpoint).result(timeout=60)
                report["checks"].append({"name": name, "api_name": endpoint,
                    **verify_policy_response(response, supplied_l, supplied_u, artifacts, endpoint)})
            require(report["checks"][0]["cost_per_1000"] != report["checks"][1]["cost_per_1000"], "Extreme policies did not change cost.")
            require(report["checks"][0]["review_fraction"] == 0 and report["checks"][2]["review_fraction"] == 1,
                    "Extreme policies did not change review share from zero to one.")
            uploaded = client.submit(handle_file(str(image_path)), selected["lower"], selected["upper"], api_name="/predict_image").result(timeout=60)
            require(len(uploaded) == 3, "Image callback returned the wrong output count.")
            annotated, description, boxes = uploaded
            match = re.search(r'data-score="([0-9.eE+-]+)"', description)
            require(match is not None, "Live image score is absent from the result.")
            live_score = float(match.group(1))
            require(abs(live_score - float(scores[index])) <= .002,
                    f"Live and cached validation scores differ beyond 0.002: {live_score} vs {float(scores[index])}.")
            def action_for(score):
                return "ACCELERATED CLINICAL REVIEW" if score >= selected["upper"] else "ROUTINE CLINICAL QUEUE" if score < selected["lower"] else "EXTRA HUMAN PRIORITY CHECK"
            route_match = re.search(r'data-route="([^"]+)"', description)
            require(route_match is not None, "The plain-language route card lacks its declared route.")
            actual_action = html.unescape(route_match.group(1))
            # The visible score has four decimals; machine metadata retains 12 significant digits.
            possible_actions = {action_for(max(0., live_score - 1e-11)), action_for(min(1., live_score + 1e-11))}
            require(actual_action in possible_actions, "Live score and stated route differ beyond metadata rounding.")
            annotated_path = Path(annotated["path"] if isinstance(annotated, dict) else annotated)
            with Image.open(annotated_path) as image:
                require(image.size == (row["width"], row["height"]), "Annotated output dimensions changed.")
                image.verify()
            require(boxes.get("headers") == ["Finding", "Confidence", "Box score band"], "Prediction table schema differs.")
            for finding, confidence, band in boxes["data"]:
                require(finding in artifacts["validation"]["class_names"], "Upload returned an unknown medical class.")
                require(float(confidence) >= selected["lower"] - .00005, "Displayed box falls below the lower threshold.")
                require(band in ("Accelerated", "Review band"), "Displayed box band is invalid.")
            report["upload"] = {"image_id": ids[index], "source_image_sha256": row["image_sha256"],
                "cached_score": float(scores[index]), "live_score": live_score,
                "score_tolerance": .002, "route": actual_action, "displayed_boxes": len(boxes["data"]),
                "annotated_output": str(annotated_path), "passed": True}
            for relative, value in protected.items():
                path = local_path(artifacts["config"]["scenario_path"], root) if relative == "scenario.json" else root / relative
                require(sha256_file(path) == value, f"Dashboard interaction changed frozen evidence: {relative}")
            require(sha256_file(image_path) == row["image_sha256"], "Upload changed the source image.")
            report.update(status="passed", frozen_artifacts_unchanged=True, source_image_unchanged=True)
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        stop_process_tree(process)
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["own_process_tree_stopped"] = process is None or process.poll() is not None
        write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.root, args.port, args.output)
    print(json.dumps({"status": report["status"], "port": report["port"], "slider_checks": len(report["checks"]),
                      "upload_passed": report["upload"]["passed"]}))


if __name__ == "__main__":
    main()
