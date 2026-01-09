# Transformer-Based Quantitative Stock Selection Strategy

## 📈 Project Overview
This project implements a deep learning-based quantitative trading strategy focusing on the **CSI 300 Index (沪深300)**. 

Unlike traditional linear factor models or LSTM baselines, this model leverages the **Transformer Encoder** architecture with **Multi-head Attention** mechanisms to capture long-term dependencies in financial time series data and mine non-linear alpha factors.

## Key Performance (Out-of-Sample)
* **Information Coefficient (IC):** `0.0582` (Strong predictive power)
* **Strategy Return:** `20.04%`
* **Benchmark Return:** `11.96%`
* **Excess Return:** `+8.08%`

![Backtest Result](backtest_result.png)
*(Figure: Out-of-sample backtest comparison against CSI 300 Benchmark)*

## Technical Architecture
* **Data Pipeline**: Tushare Pro API (Daily frequency).
* **Feature Engineering**: Constructed 6 technical factors including Volatility (5D), MACD, RSI, and Momentum, processed with Z-Score normalization.
* **Model Core**: 
    * **Transformer Encoder**: 2 Layers, 4 Heads, d_model=64.
    * **Positional Encoding**: Preserving sequence order information.
    * **Loss Function**: MSE (Mean Squared Error).
* **Backtesting**: Vectorized backtesting framework simulating daily top-k holding strategy.

## How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Transformer-Quant-Strategy.git](https://github.com/YOUR_USERNAME/Transformer-Quant-Strategy.git)
   Install dependencies:

Bash
pip install -r requirements.txt
Configure Token: Open main.py and replace tushare_token with your own Tushare Pro API token.

Run the strategy:

Bash
python main.py
