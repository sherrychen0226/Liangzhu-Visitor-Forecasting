# Forecasting Hourly Visitor Arrivals at Liangzhu Archaeological Ruins Park

A deep-learning time series project that predicts **hourly visitor arrivals for the next 24 hours** at Liangzhu Archaeological Ruins Park (Hangzhou, China), using a **dual-GRU neural network implemented from scratch in NumPy** — no PyTorch or TensorFlow.

## Project Overview

Parks need short-term visitor forecasts for staffing, ticketing, crowd control, and safety planning. This project builds a complete forecasting pipeline on historical entrance records spanning **June 2019 to June 2026** (~58,800 hourly observations):

> Raw entrance logs → hourly aggregation → calendar & weather enrichment → sliding-window samples → dual-GRU training → evaluation & feature ablation.

The model consumes the **previous 24 hours of entrance counts** together with the **calendar and weather conditions of the 24 forecast hours**, and outputs 24 hourly arrival predictions in one shot.

## Repository Structure

| File | Purpose |
|---|---|
| `process_chengzhi_hourly.py` | Aggregates raw entrance records into hourly counts; merges Chinese holiday calendar (2019–2026) and Hangzhou hourly weather; exports the enriched CSV/XLSX |
| `build_dual_gru_dataset_covariates24_weather3.py` | Builds sliding-window samples: past-24h entrance branch + future-24h covariate branch, split into train/val/test |
| `train_dual_gru_numpy.py` | Dual-GRU model, training loop, and evaluation — implemented in pure NumPy (manual forward/backward passes) |
| `run_dual_gru_covariates24_weather3_ablation.py` | Ablation study: full model vs. no-weather vs. no-calendar |
| `余杭区良渚古城遗址公园综合管控系统-游客出入园信息20260618.csv` | Raw entrance records |
| `china_holiday_2019_2026.json` | Chinese holiday calendar cache |
| `hangzhou_hourly_weather_2019-06-26_2026-06-02.json` | Hangzhou hourly weather cache |
| `Project Report.pdf` | Full project report |

## Data

- **Source:** Entrance records from the park's comprehensive management system (城址区, the main heritage site area).
- **Coverage:** Hourly from 2019-06-26 to 2026-06, with missing timestamps gap-filled (overnight hours zero-filled).
- **Features per hour:** entrance count, weekday, holiday flag, and weather condition.

## Methodology

- **Data Processing:** Raw entrance records aggregated into hourly counts; missing timestamps handled; ~48,124 valid sliding-window samples generated (train / val / test ≈ 38,499 / 4,812 / 4,813).
- **Feature Engineering:** Two input branches —
  - *Entrance branch:* entrance counts from the previous 24 hours;
  - *Covariate branch:* 13 channels covering hour / weekday / holiday encodings and 3 weather variables, aligned with the **following 24 forecast hours** (so the model only uses covariates that are known ahead of time, such as calendar facts and weather forecasts).
- **Model Architecture:** A dual-GRU encoder — one GRU per branch — whose final hidden states are concatenated and mapped by a linear head to 24 hourly predictions. Xavier initialization, early stopping on validation loss.

## Results

Test-set performance (24-hour-ahead forecast):

| Metric | Result |
|---|---|
| MAE | 65.40 |
| RMSE | 157.79 |
| R² | 0.854 |
| SMAPE | 97.39% |

> Note: SMAPE is inflated by overnight hours with near-zero arrivals (the denominator collapses toward zero), so MAE / RMSE / R² are the more informative metrics here.

## Ablation Study

A covariate ablation (full vs. without-weather vs. without-calendar) showed that **calendar information contributes significantly to forecasting accuracy**, while weather variables provide more limited overall improvements.

## Reproducing

1. Install dependencies: `pandas`, `openpyxl`, `numpy`.
2. Run the pipeline in order:
   ```bash
   python process_chengzhi_hourly.py
   python build_dual_gru_dataset_covariates24_weather3.py
   python train_dual_gru_numpy.py
   python run_dual_gru_covariates24_weather3_ablation.py
   ```
   

