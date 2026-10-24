# Liangzhu-Visitor-Forecasting
Forecasting hourly visitor arrivals using a dual-GRU neural network with calendar and weather features.
# Forecasting Hourly Visitor Arrivals at Liangzhu Archaeological Ruins Park

A time series forecasting project using deep learning to predict hourly visitor arrivals at Liangzhu Archaeological Ruins Park in Hangzhou, China.

## Project Overview

This project aims to support visitor management and operational planning through short-term visitor forecasting. Using historical entrance records from June 2019 to June 2026, I developed a **dual-GRU neural network** to predict visitor arrivals for the next 24 hours.

The project covers data preprocessing, exploratory data analysis, feature engineering, model development, and evaluation.

## Methodology

- **Data Processing:** Aggregated raw entrance records into hourly counts, handled missing timestamps, and generated 48,124 time series samples.
- **Feature Engineering:** Incorporated calendar information and weather conditions alongside historical entrance counts.
- **Model Architecture:** Developed a dual-GRU model that separately processes the previous 24-hour entrance counts and the next 24-hour calendar and weather features to generate 24 hourly predictions.

## Results

The model achieved the following performance on the test set:

| Metric | Result |
|---|---|
| MAE | 65.40 |
| RMSE | 157.79 |
| R² | 0.854 |
| SMAPE | 97.39% |

A covariate ablation study showed that **calendar information contributed significantly to forecasting accuracy**, while weather variables provided more limited overall improvements.


