"""Run full, weather, and calendar ablations with 24 forecast covariate hours."""

from __future__ import annotations

import csv
import json

import numpy as np

import run_dual_gru_covariate_ablation as experiment
import train_gru_numpy as base


DATA_DIR = base.BASE_DIR / "timeseries_dataset_dual_gru_covariates24_weather3"
OUTPUT_ROOT = base.BASE_DIR / "dual_gru_numpy_ablation_results_covariates24_weather3"


def main() -> None:
    experiment.DATA_DIR = DATA_DIR
    experiment.OUTPUT_ROOT = OUTPUT_ROOT
    metadata = json.loads((DATA_DIR / "metadata.json").read_text(encoding="utf-8"))
    splits = {}
    for split in ("train", "val", "test"):
        with np.load(DATA_DIR / f"{split}.npz") as data:
            splits[split] = {key: data[key] for key in data.files}

    results = [
        experiment.train_condition(condition, metadata, splits)
        for condition in ("full", "no_weather", "no_calendar")
    ]
    identities = [result["test_identity"] for result in results]
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("Ablation conditions used different test arrays.")

    labels = {
        "full": ("Full model", "None"),
        "no_weather": ("Without weather", "Three weather variables"),
        "no_calendar": ("Without calendar", "Hour, weekday, and holiday variables"),
    }
    rows = []
    for result in results:
        result["architecture"]["covariate_gru"] = "24 x 13 input, 32 hidden units"
        result["covariate_alignment"] = "Following 24 forecast hours only"
        condition_dir = OUTPUT_ROOT / result["condition"]
        (condition_dir / "metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        label, removed = labels[result["condition"]]
        test = result["test_metrics"]
        rows.append(
            {
                "condition": label,
                "removed_features": removed,
                "mae": float(test["mae"]),
                "rmse": float(test["rmse"]),
                "r2": float(test["r2"]),
                "smape_percent": float(test["smape_percent"]),
                "best_epoch": int(result["best_epoch"]),
                "epochs_trained": int(result["epochs_trained"]),
            }
        )

    full = rows[0]
    for row in rows:
        for metric in ("mae", "rmse", "r2", "smape_percent"):
            row[f"{metric}_change_vs_full"] = row[metric] - full[metric]

    summary = {
        "comparison_basis": {
            "same_test_arrays_verified": True,
            "same_samples": {"train": 38499, "val": 4812, "test": 4813},
            "same_architecture_and_training_settings": True,
            "entrance_input": "Previous 24 hours, 1 channel",
            "covariate_input": "Forecast 24 hours, 13 channels",
            "weather_representation": "No precipitation, precipitation, and snow",
        },
        "rows": rows,
    }
    OUTPUT_ROOT.mkdir(exist_ok=True)
    (OUTPUT_ROOT / "ablation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUTPUT_ROOT / "ablation_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("FINAL_SUMMARY")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
