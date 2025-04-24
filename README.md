# Time Series Forecasting with SARIMA on AirPassengers Dataset

This project implements a complete pipeline for **time series forecasting** using the **SARIMA** (Seasonal ARIMA) model on the classic `AirPassengers` dataset. It includes preprocessing, exploratory analysis, model training, forecasting, and visualization of prediction confidence.

---

## Project Overview

This project follows a structured process to forecast monthly air passenger counts using SARIMA:

1. **Data Parsing & Preprocessing**  
   The raw dataset has a time column in fractional year format. This is converted into a proper datetime index with monthly granularity.

2. **Train-Test Split**  
   The data is split chronologically into training (until 1959) and testing (from 1960 onwards) for a realistic forecasting scenario.

3. **SARIMA Modeling**  
   A Seasonal ARIMA model is fitted on the training set and used to forecast passenger numbers for the test period. The model considers both trend and seasonal components.

4. **Visualization**  
   Forecast results are plotted alongside actual data, including confidence intervals to express prediction uncertainty.

---

## Key Features

1. **Date Conversion from Fractional Format**  
   Custom logic to convert `time` (e.g., 1950.5) into real `datetime` objects using year and fractional months.

2. **SARIMA Forecasting**  
   Implements SARIMA with seasonal terms to capture yearly patterns in monthly data.

3. **Confidence Interval Plotting**  
   Forecasts include a visual representation of upper and lower prediction bounds using `conf_int`.

4. **Evaluation Split**  
   Realistic split ensures that the model does not see future data during training.

---

## Technical Steps

- **Preprocessing**
  - Convert fractional years into `datetime` monthly index
  - Set up the dataset with proper naming and indexing

- **Model Setup**
  - Define SARIMA parameters: `order=(1,1,1)` and `seasonal_order=(1,1,1,12)`
  - Fit the model using `SARIMAX` from `statsmodels`

- **Forecasting**
  - Predict values for the test set period
  - Extract both predicted means and confidence intervals

- **Visualization**
  - Plot training, testing, and forecast curves
  - Use `matplotlib` to display the confidence bands over time

---

## File Outputs

- `Forecast Plot`: Line chart showing training data, test data, model forecast, and confidence interval bands.
- (Optional) CSV exports can be added to save the predictions and model results.

---

## Libraries Used

- `pandas`, `numpy`
- `matplotlib`
- `statsmodels`
- `sklearn.metrics`

---

## Conclusion

This project shows how to implement a **classical SARIMA model** for univariate time series forecasting.  
It demonstrates the value of **seasonal modeling**, **confidence intervals**, and proper **time-based train/test splitting** for realistic evaluation.

Ideal for those exploring traditional statistical models before jumping into machine learning or deep learning for time series.

---
