# Corporate Credit Risk Screening & Financial Distress Prediction
> **A Hybrid Credit Risk Architecture Combining Structural Merton Models with Machine Learning and Explainable AI (SHAP)**

----

### Research Motivation

**Bridging Structural Credit Diagnostics and Dynamic Risk Architecture**

While the foundational multi-asset stress-testing framework established an independent top-down validation layer for macroeconomic and systematic liquidity shocks, institutional capital deployment requires a granular, bottom-up exposure model at the corporate entity level. In cross-sectional alpha generation—such as the balance-sheet dynamics explored during the WorldQuant BRAIN International Quant Championship (IQC) 2026—statistical fundamental anomalies frequently mask underlying structural solvency risks and tail-event vulnerabilities.

Standard machine learning classifiers applied to corporate distress prediction often operate as opaque "black boxes," lacking theoretical grounding in capital structure dynamics and failing regulatory requirements for model interpretability (e.g., Basel III/IV frameworks). Conversely, traditional structural approaches like the Merton (1974) model offer explicit market-implied solvency metrics—such as Distance to Default ($DD$)—yet suffer from rigid assumptions when processing high-dimensional distress indicators.

To resolve this dichotomy, this project constructs a **Hybrid Corporate Credit Risk & Financial Distress Screening Engine**. By integrating Merton's structural option-pricing framework with gradient-boosted decision trees and Explainable AI (SHAP), this framework transforms raw corporate fundamental and market dynamics into calibrated Default Probabilities (PD) and Credit Value at Risk (Credit VaR).

---

### Objective

* **Engineered Multi-Sourced Feature Pipeline:** Construct an ingestion pipeline integrating SEC EDGAR fundamental financials with equity market volatility to extract non-linear distress triggers.
* **Structural Distance to Default Solver:** Implement a numerical optimization solver for the Merton (1974) model to reverse-engineer unobservable Asset Value ($V_A$) and Asset Volatility ($\sigma_A$).
* **Hybrid Predictive Modeling:** Develop ensemble classifiers (LightGBM/XGBoost) and survival models to predict corporate distress, comparing performance against Altman Z-Score baselines.
* **Interpretability & Stress Testing:** Apply SHAP attribution to decompose risk drivers and evaluate portfolio Expected Loss ($EL$) under macro credit regime shifts.
* 


