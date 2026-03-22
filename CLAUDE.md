# Regime-Aware ML Trading System

## Project Overview
A modular algorithmic trading system that combines regime detection, 
EMA crossover strategy, and machine learning (Logistic Regression) 
to generate trading signals. Supports backtesting and paper trading 
via Alpaca API.

## Project Structure
- data/          → Alpaca data fetching
- strategy/      → EMA strategy, ML strategy signal generation
- features/      → Feature engineering for ML
- models/        → Model training and prediction
- backtesting/   → Backtest engine and performance metrics
- paper_trading/ → Live paper trade execution (in progress)
- config.py      → All parameters live here — edit this first

## Current State
Phase 1 (ML Layer) — COMPLETE
Phase 2 (Paper Trading) — IN PROGRESS
Phase 3 (Robustness & Polish) — TODO
Phase 4 (README) — TODO

## Key Design Decisions
- Strict no-leakage discipline: features use only past data, 
  shift(-1) only used for target creation
- EMA and ML use decoupled regime thresholds (see config.py)
- Train/test split is always time-based, never shuffled
- Per-symbol model training (cross-symbol was tested, performed worse)
- Transaction cost of 0.1% applied on every position change

## Parameters (config.py)
- SYMBOLS: AAPL, MSFT, NVDA, SPY, QQQ
- LOOKBACK_DAYS: 1500
- SHORT_EMA / LONG_EMA: 10 / 20
- EMA regime: slope=0.0007, r2=0.6
- ML regime: slope=0.0003, r2=0.3
- ML_PROB_THRESHOLD: 0.55
- TRANSACTION_COST: 0.001

## What NOT to do
- Do not shuffle time series data
- Do not fit scaler on test data
- Do not add complex models (no XGBoost, no deep learning)
- Do not break existing backtest engine
- Do not hardcode parameters — everything goes in config.py

## Dependencies
alpaca-trade-api, scikit-learn, pandas, numpy, python-dotenv