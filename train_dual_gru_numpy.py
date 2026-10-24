"""Train two GRU encoders and a joint 24-hour entrance-count predictor."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import train_gru_numpy as base


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "timeseries_dataset_dual_gru_covariates48_weather3"
OUTPUT_DIR = BASE_DIR / "dual_gru_numpy_results_covariates48_weather3"


def xavier(rng: np.random.Generator, in_dim: int, out_dim: int) -> np.ndarray:
    limit = math.sqrt(6.0 / (in_dim + out_dim))
    return rng.uniform(-limit, limit, (in_dim, out_dim)).astype(np.float32)


def init_encoder(rng: np.random.Generator, prefix: str, input_size: int) -> dict:
    h = base.HIDDEN_SIZE
    return {
        f"{prefix}_W_zx": xavier(rng, input_size, h),
        f"{prefix}_W_zh": xavier(rng, h, h),
        f"{prefix}_b_z": np.zeros(h, np.float32),
        f"{prefix}_W_rx": xavier(rng, input_size, h),
        f"{prefix}_W_rh": xavier(rng, h, h),
        f"{prefix}_b_r": np.zeros(h, np.float32),
        f"{prefix}_W_nx": xavier(rng, input_size, h),
        f"{prefix}_W_nh": xavier(rng, h, h),
        f"{prefix}_b_n": np.zeros(h, np.float32),
    }


def init_params(rng: np.random.Generator) -> dict:
    params = {}
    params.update(init_encoder(rng, "entrance", 1))
    params.update(init_encoder(rng, "covariate", 13))
    params["W_y"] = xavier(rng, 2 * base.HIDDEN_SIZE, base.OUTPUT_LEN)
    params["b_y"] = np.zeros(base.OUTPUT_LEN, np.float32)
    return params


def encode(params: dict, prefix: str, x: np.ndarray, keep_cache: bool):
    h = np.zeros((len(x), base.HIDDEN_SIZE), np.float32)
    caches = []
    for t in range(x.shape[1]):
        xt, h_prev = x[:, t, :], h
        z = base.sigmoid(xt @ params[f"{prefix}_W_zx"] + h_prev @ params[f"{prefix}_W_zh"] + params[f"{prefix}_b_z"])
        r = base.sigmoid(xt @ params[f"{prefix}_W_rx"] + h_prev @ params[f"{prefix}_W_rh"] + params[f"{prefix}_b_r"])
        rh = r * h_prev
        n = np.tanh(xt @ params[f"{prefix}_W_nx"] + rh @ params[f"{prefix}_W_nh"] + params[f"{prefix}_b_n"])
        h = (1.0 - z) * n + z * h_prev
        if keep_cache:
            caches.append((xt, h_prev, z, r, rh, n))
    return h, caches


def forward(params: dict, x_e: np.ndarray, x_c: np.ndarray, keep_cache=False):
    h_e, cache_e = encode(params, "entrance", x_e, keep_cache)
    h_c, cache_c = encode(params, "covariate", x_c, keep_cache)
    joined = np.concatenate([h_e, h_c], axis=1)
    prediction = joined @ params["W_y"] + params["b_y"]
    if keep_cache:
        return prediction, joined, cache_e, cache_c
    return prediction


def backward_encoder(params, prefix, caches, dh, grads):
    for xt, h_prev, z, r, rh, n in reversed(caches):
        dz = dh * (h_prev - n)
        dn = dh * (1.0 - z)
        dh_prev = dh * z
        dn_pre = dn * (1.0 - n * n)
        grads[f"{prefix}_W_nx"] += xt.T @ dn_pre
        grads[f"{prefix}_W_nh"] += rh.T @ dn_pre
        grads[f"{prefix}_b_n"] += dn_pre.sum(0)
        drh = dn_pre @ params[f"{prefix}_W_nh"].T
        dr = drh * h_prev
        dh_prev += drh * r
        dr_pre = dr * r * (1.0 - r)
        grads[f"{prefix}_W_rx"] += xt.T @ dr_pre
        grads[f"{prefix}_W_rh"] += h_prev.T @ dr_pre
        grads[f"{prefix}_b_r"] += dr_pre.sum(0)
        dh_prev += dr_pre @ params[f"{prefix}_W_rh"].T
        dz_pre = dz * z * (1.0 - z)
        grads[f"{prefix}_W_zx"] += xt.T @ dz_pre
        grads[f"{prefix}_W_zh"] += h_prev.T @ dz_pre
        grads[f"{prefix}_b_z"] += dz_pre.sum(0)
        dh_prev += dz_pre @ params[f"{prefix}_W_zh"].T
        dh = dh_prev


def backward(params, x_e, x_c, y):
    pred, joined, cache_e, cache_c = forward(params, x_e, x_c, True)
    grads = {k: np.zeros_like(v) for k, v in params.items()}
    dy = (2.0 / y.size) * (pred - y)
    grads["W_y"] = joined.T @ dy
    grads["b_y"] = dy.sum(0)
    d_joined = dy @ params["W_y"].T
    backward_encoder(params, "entrance", cache_e, d_joined[:, :base.HIDDEN_SIZE], grads)
    backward_encoder(params, "covariate", cache_c, d_joined[:, base.HIDDEN_SIZE:], grads)
    for grad in grads.values():
        np.clip(grad, -1.0, 1.0, out=grad)
    return base.mse_loss(pred, y), grads


def predict_batches(params, x_e, x_c, batch_size=1024):
    return np.concatenate([
        forward(params, x_e[i:i + batch_size], x_c[i:i + batch_size])
        for i in range(0, len(x_e), batch_size)
    ])


def metrics(pred_scaled, y_scaled, y_actual, y_mean, y_std):
    pred = np.maximum(pred_scaled * y_std + y_mean, 0.0)
    error = pred - y_actual
    denom = float(np.sum((y_actual - y_actual.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "r2": float(1.0 - np.sum(error ** 2) / denom),
        "smape_percent": float(np.mean(2 * np.abs(error) / (np.abs(pred) + np.abs(y_actual) + 1e-6)) * 100),
        "mse_scaled": base.mse_loss(pred_scaled, y_scaled),
    }, pred


def load(split):
    data = np.load(DATA_DIR / f"{split}.npz")
    return {key: data[key] for key in data.files}


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = {split: load(split) for split in ("train", "val", "test")}
    train = data["train"]
    e_mean = train["X_entrance"].mean((0, 1), keepdims=True)
    e_std = train["X_entrance"].std((0, 1), keepdims=True) + 1e-6
    c_mean = train["X_covariates"].mean((0, 1), keepdims=True)
    c_std = train["X_covariates"].std((0, 1), keepdims=True) + 1e-6
    y_mean, y_std = float(train["y"].mean()), float(train["y"].std() + 1e-6)
    for values in data.values():
        values["e_s"] = (values["X_entrance"] - e_mean) / e_std
        values["c_s"] = (values["X_covariates"] - c_mean) / c_std
        values["y_s"] = (values["y"] - y_mean) / y_std

    rng = np.random.default_rng(base.SEED)
    params = init_params(rng)
    state = {"m": {k: np.zeros_like(v) for k, v in params.items()}, "v": {k: np.zeros_like(v) for k, v in params.items()}}
    best_loss, best_epoch, wait, step = float("inf"), 0, 0, 0
    best_params = {k: v.copy() for k, v in params.items()}
    history = []
    for epoch in range(1, base.EPOCHS + 1):
        order = rng.permutation(len(train["y"]))
        losses = []
        for start in range(0, len(order), base.BATCH_SIZE):
            idx = order[start:start + base.BATCH_SIZE]
            loss, grads = backward(params, train["e_s"][idx], train["c_s"][idx], train["y_s"][idx])
            step += 1
            base.adam_update(params, grads, state, step)
            losses.append(loss)
        val_pred = predict_batches(params, data["val"]["e_s"], data["val"]["c_s"])
        val_loss = base.mse_loss(val_pred, data["val"]["y_s"])
        if val_loss < best_loss:
            best_loss, best_epoch, wait = val_loss, epoch, 0
            best_params = {k: v.copy() for k, v in params.items()}
        else:
            wait += 1
        row = {"epoch": epoch, "train_mse_scaled": float(np.mean(losses)), "val_mse_scaled": float(val_loss), "best_val_mse_scaled": float(best_loss), "epochs_without_improvement": wait}
        history.append(row)
        print(json.dumps(row), flush=True)
        if wait >= base.EARLY_STOPPING_PATIENCE:
            break

    val_scaled = predict_batches(best_params, data["val"]["e_s"], data["val"]["c_s"])
    test_scaled = predict_batches(best_params, data["test"]["e_s"], data["test"]["c_s"])
    val_metrics, _ = metrics(val_scaled, data["val"]["y_s"], data["val"]["y"], y_mean, y_std)
    test_metrics, test_pred = metrics(test_scaled, data["test"]["y_s"], data["test"]["y"], y_mean, y_std)
    result = {
        "model": "Dual NumPy GRU encoders with joint linear predictor",
        "dataset": str(DATA_DIR),
        "architecture": {"entrance_gru": {"sequence_length": 24, "input_size": 1, "hidden_size": 32}, "covariate_gru": {"sequence_length": 48, "input_size": 13, "hidden_size": 32}, "fusion": "Concatenate final hidden states", "predictor": "Linear 64-to-24 output layer"},
        "samples": {k: int(len(v["y"])) for k, v in data.items()},
        "maximum_epochs": base.EPOCHS, "epochs_trained": len(history), "early_stopping_patience": base.EARLY_STOPPING_PATIENCE,
        "best_epoch": best_epoch, "best_val_mse_scaled": float(best_loss), "batch_size": base.BATCH_SIZE, "learning_rate": base.LEARNING_RATE, "seed": base.SEED,
        "val_metrics": val_metrics, "test_metrics": test_metrics,
    }
    np.savez_compressed(OUTPUT_DIR / "dual_gru_numpy_model.npz", **best_params, entrance_mean=e_mean, entrance_std=e_std, covariate_mean=c_mean, covariate_std=c_std, y_mean=np.array(y_mean), y_std=np.array(y_std))
    np.savez_compressed(OUTPUT_DIR / "test_predictions.npz", pred=test_pred, actual=data["test"]["y"], entrance_time=data["test"]["entrance_time"], covariate_time=data["test"]["covariate_time"], y_time=data["test"]["y_time"])
    (OUTPUT_DIR / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("FINAL_RESULTS")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
