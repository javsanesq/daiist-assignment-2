"""Summarize genuine local training outputs on CPU without inference or retraining."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import extract_training_summary


if __name__ == "__main__":
    result = extract_training_summary(ROOT)
    print(f"Saved training_summary.json: {result['completed_epochs']} completed epochs; "
          f"checkpoint epoch {result['best_epoch']} ({result['best_epoch_source']}); "
          f"{result['training_loop_seconds_to_final_logged_epoch']:.2f} seconds through the final logged epoch.")
