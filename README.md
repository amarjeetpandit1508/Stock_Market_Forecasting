
# Stock Market Forecasting with Deep Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.7.0](https://img.shields.io/badge/PyTorch-2.7.0-orange.svg)](https://pytorch.org/)
[![Matplotlib 3.6.2](https://img.shields.io/badge/Matplotlib-3.6.2-green.svg)](https://matplotlib.org/)
[![NumPy 1.24.0](https://img.shields.io/badge/NumPy-1.24.0-blue.svg)](https://numpy.org/)
[![Pandas 1.5.2](https://img.shields.io/badge/Pandas-1.5.2-yellowgreen.svg)](https://pandas.pydata.org/)
[![Scikit-learn 1.5.0](https://img.shields.io/badge/Scikit--learn-1.5.0-lightblue.svg)](https://scikit-learn.org/)
[![SciPy 1.10.0](https://img.shields.io/badge/SciPy-1.10.0-lightgrey.svg)](https://scipy.org/)

This repository contains a **stock price prediction and forecasting model** implemented in Python using **deep learning (Stacked LSTM)**. The goal of this project is to explore time-series forecasting of stock prices and demonstrate how neural networks such as LSTM (Long Short-Term Memory) can be used to predict future stock values based on historical data.

---

## Overview

Stock market prediction is a classic problem in machine learning and deep learning due to the **complex, non-linear, and time-dependent nature** of financial data. In this project:

- Historical stock price data is collected (e.g., TESLA)  
- Data is preprocessed and scaled for training  
- A **Stacked LSTM network** is trained to learn temporal dependencies  
- Predictions are made for future closing prices  
- Resulting forecasts are visualized and evaluated  

> ⚠️ This model is **for educational purposes only** and should *not* be used for real financial or investment decisions.

---

## Key Features

✔️ Downloads and preprocesses historical stock data  
✔️ Uses **LSTM neural networks** to model time-series behavior  
✔️ Trains on a portion of the dataset and tests on unseen data  
✔️ Predicts future stock closing prices  
✔️ Scales and visualizes results using common libraries

---

## How It Works

1. **Data Collection**   
   - Historical stock data is fetched directly using the **yfinance** library  
   - Relevant fields such as *open*, *high*, *low*, *close*, and *volume* are used  
   - Data is accessed in memory and processed without being saved locally  


2. **Preprocessing**  
   - Stock price values (especially *close* prices) are scaled (e.g., MinMaxScaler)  
   - Time windows are created for the sequential data  
   - Train and test splits are created for model evaluation  

3. **Model Creation (Stacked LSTM)**  
   - A deep LSTM network is built using TensorFlow/Keras  
   - Layers are stacked to capture complex patterns  
   - Model is compiled and trained on the dataset

4. **Prediction & Forecast**  
   - Predictions on test data demonstrate model accuracy  
   - Future stock prices can be forecasted using prior windows  
   - Results are plotted for visualization

5. **Evaluation**  
   - Plots compare real vs. predicted stock prices  
   - Loss trends can be observed to evaluate model learning

---

## Tools & Libraries

This project uses the following tools and libraries:

- **[PyTorch](https://pytorch.org/)** – LSTM deep learning models
- **[NumPy](https://numpy.org/)** – Numerical computations
- **[Pandas](https://pandas.pydata.org/)** – Data manipulation and preprocessing
- **[Matplotlib](https://matplotlib.org/)** – Data visualization
- **[Scikit-learn](https://scikit-learn.org/)** – Data preprocessing and utilities
- **[SciPy](https://scipy.org/)** – Scientific computing and MATLAB `.mat` file handling


---
