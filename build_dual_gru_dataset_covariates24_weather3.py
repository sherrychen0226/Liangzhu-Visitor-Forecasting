"""Build dual-GRU inputs using only forecast-period covariates.

The entrance branch receives entrance counts from the previous 24 hours.  The
covariate branch receives calendar and three-category weather features aligned
with the following 24 forecast hours.  Targets are the entrance counts for
those same following 24 hours.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
HISTORICAL_DIR = BASE_DIR / "timeseries_dataset_len24_pred24_overnight_zero_fill_weather3"
FUTURE_DIR = BASE_DIR / "timeseries_dataset_len24_pred24_overnight_zero_fill_weather3_future_covariates"
OUTPUT_DIR = BASE_DIR / "timeseries_dataset_dual_gru_covariates24_weather3"


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    source_meta = json.loads((FUTURE_DIR / "metadata.json").read_text(encoding="utf-8"))

    for split in ("train", "val", "test"):
        with np.load(HISTORICAL_DIR / f"{split}.npz") as historical, np.load(
            FUTURE_DIR / f"{split}.npz"
        ) as future:
            for key in ("y", "X_time", "y_time", "sample_id"):
                if not np.array_equal(historical[key], future[key]):
                    raise ValueError(f"Mismatched {split} arrays: {key}")

            x_entrance = historical["X"][:, :, 0:1].astype(np.float32)
            x_covariates = future["X"][:, :, 1:14].astype(np.float32)
            covariate_time = future["X_covariate_time"]

            if not np.array_equal(covariate_time, future["y_time"]):
                raise ValueError(f"Forecast covariates are not target-aligned: {split}")

            np.savez_compressed(
                OUTPUT_DIR / f"{split}.npz",
                X_entrance=x_entrance,
                X_covariates=x_covariates,
                y=future["y"].astype(np.float32),
                entrance_time=historical["X_time"],
                covariate_time=covariate_time,
                y_time=future["y_time"],
                sample_id=future["sample_id"],
            )

    metadata = {
        "source_datasets": [str(HISTORICAL_DIR), str(FUTURE_DIR)],
        "output_dir": str(OUTPUT_DIR),
        "valid_samples": source_meta["valid_samples"],
        "split_ratio": source_meta["split_ratio"],
        "splits": {"train": 38499, "val": 4812, "test": 4813},
        "entrance_input_shape_per_sample": [24, 1],
        "covariate_input_shape_per_sample": [24, 13],
        "target_shape_per_sample": [24],
        "covariate_feature_names": source_meta["feature_names"][1:],
        "weather_grouping": source_meta["weather_grouping"],
        "alignment": {
            "entrance": "Previous 24 hours",
            "covariates": "Following 24 forecast hours, aligned one-to-one with targets",
            "target": "Following 24 forecast hours",
        },
        "reference_targets_and_times_verified": True,
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
