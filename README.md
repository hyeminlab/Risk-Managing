# Corporate Credit Risk Screening & Financial Distress Prediction
> **A Hybrid Credit Risk Architecture Combining Structural Merton Models with Machine Learning and Explainable AI (SHAP)**

---

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
* **Interactive Risk Engine & Scenario Simulator:** Build a Streamlit-based diagnostic interface incorporating macro credit stress tests (interest rate/equity shocks) and individual entity-level XAI diagnostics.

---

### Architecture & Pipeline Overview

* **Data Preprocessing & Feature Engineering:** Ingest fundamental ratios and SEC EDGAR financial data, generating 95 core liquidity/solvency features.
* **Merton Structural Solver:** Reverse-engineer unobservable Asset Value ($V_A$) and Asset Volatility ($\sigma_A$) using numerical optimization (Nelder-Mead).
* **Hybrid Feature Matrix:** Combine structural metrics ($DD$, $PD$) with high-dimensional accounting ratios.
* **Ensemble Credit Engine:** Train cost-sensitive LightGBM and XGBoost classifiers optimized for minority class recall.
* **Institutional XAI Layer:** Generate SHAP Summary Plots and Waterfall Plots for local and global interpretability.
* **Macro Stress-Testing Engine:** Simulate macroeconomic shocks (e.g., interest rate hikes, equity market drawdowns) on entity default risk.

---

### Structural Merton Model Formulation

The Merton (1974) structural model views corporate equity as a European call option on total firm asset value $V_A$ with a strike price equal to the face value of debt $D$ maturing at time $T$. Since $V_A$ and asset volatility $\sigma_A$ are unobservable, they are reverse-engineered by solving the non-linear system of equations using numerical optimization:

$$E = V_A N(d_1) - D e^{-rT} N(d_2)$$

$$\sigma_E = \left( \frac{V_A}{E} \right) N(d_1) \sigma_A$$

Where:
$$d_1 = \frac{\ln(V_A / D) + (r + 0.5 \sigma_A^2) T}{\sigma_A \sqrt{T}}, \quad d_2 = d_1 - \sigma_A \sqrt{T}$$

The **Distance to Default ($DD$)** and **Structural Probability of Default ($PD$)** are extracted as:

$$\text{Distance to Default } (DD) = d_2$$

$$\text{Merton } PD = N(-d_2)$$

---

### Key Performance & Empirical Results

Addressing extreme class imbalance via cost-sensitive learning (`scale_pos_weight`) yielded significant performance gains in corporate distress detection, outperforming traditional baselines.

| Model Architecture | Default Recall (Target=1) | Precision | ROC-AUC | Strategic & Practical Implications |
| :--- | :---: | :---: | :---: | :--- |
| **Altman Z-Score (Baseline)** | $12.5\%$ | $0.21$ | $0.6210$ | High false negative rate; rigid linear cutoff fails under non-linear stress |
| **Random Forest** | $18.2\%$ | $0.68$ | $0.9541$ | Severe minority-class underfitting due to unweighted bagging |
| **LightGBM Classifier** | **$50.0\%$** | $0.50$ | **$0.9514$** | **Optimal balance** between Precision and Recall for general credit screening |
| **XGBoost Classifier** | **$70.0\%$** | $0.44$ | **$0.9468$** | **Maximum Distress Sensitivity**; ideal for risk-averse institutional screening |

> **Key Finding:** Incorporating the structural Merton $DD$ feature into boosted tree architectures increased distress detection (Recall) from $18\%$ to **$70\%$**, effectively mitigating False Negative risk (undetected insolvency) in institutional portfolios.

---

### Explainable AI (XAI) & Granular Entity Diagnostics

To satisfy Basel regulatory frameworks, model predictions are decomposed using SHAP (SHapley Additive exPlanations) values to resolve global feature importance and local sample-level insolvency drivers.

#### 1. Global Risk Drivers (SHAP Summary Analysis)
* **Interest Rate Dynamics (`Continuous interest rate`):** Captures effective interest payment capacity. In interaction with debt ratios, higher effective rates signal debt-service sustainability, insulating against immediate distress.
* **Capital Return (`Net Income / Total Assets - ROA`):** Inverse relationship with default probability; deteriorating ROA serves as the primary early-warning trigger for credit downgrades.
* **Leverage Structure (`Total Debt / Net Worth`):** Strong positive SHAP contribution to default probability once leverage exceeds structural thresholds.

#### 2. Local Case Study: Single-Entity Waterfall Diagnostic

* **Base Expected Value $E[f(x)]$:** $-6.29$ (Baseline Average Risk Score)
* **Risk Factor 1 (`Continuous interest rate after tax`):** $+1.69$ (Primary Stress Factor)
* **Risk Factor 2 (`Interest-bearing debt interest rate`):** $+1.54$ (Debt Burden Pressure)
* **Risk Factor 3 (`Total debt / Total net worth`):** $+1.00$ (Capital Deterioration)
* **Risk Factor 4 (`Net Income to Total Assets`):** $+0.79$ (Profitability Decay)
* **Other Features:** $+1.27$
* **Final Predicted Risk Score $f(x)$:** $+2.814$ (High-Risk Insolvency Triggered)

---

### Macro Stress-Testing & Interactive Dashboard

The engine incorporates a real-time scenario simulator allowing risk managers to evaluate portfolio resilience under macroeconomic shock regimes:

* **Shock Scenario A (Monetary Tightening):** Interest rates $+200\text{bps}$ $\rightarrow$ Escalates interest burden metrics, shifting SHAP contributions rightward.
* **Shock Scenario B (Equity Crash):** Asset value $V_A$ $-30\%$ $\rightarrow$ Sharp reduction in Merton $DD$, driving non-linear spikes in ensemble $PD$.

---

### Tech Stack & Dependencies

* **Language & Core:** Python 3.11, NumPy, Pandas
* **Structural Solver:** SciPy (`scipy.optimize.minimize`, `scipy.stats.norm`)
* **Machine Learning:** XGBoost, LightGBM, Scikit-Learn
* **Model Interpretability:** SHAP (SHapley Additive exPlanations)
* **Visualization & Web App:** Matplotlib, Streamlit

---

### Macro Stress-Testing & Dynamic Scenario Analysis

The engine incorporates a real-time macro-shock simulator that evaluates corporate resilience under extreme market regimes:

* **Scenario Parameters:**
  * **Interest Rate Shock:** $+200\text{bps}$ rate hike applied to interest-bearing liabilities.
  * **Equity & Asset Shock:** $-30\%$ drawdown applied to market equity and asset metrics.
* **Granular Impact Diagnosis:**
  * Dynamically re-evaluates Merton Distance to Default ($DD$) and tree-based ensemble probabilities ($PD$).
  * Utilizes SHAP Waterfall plots to identify feature-level stress triggers (e.g., interest burden explosion vs. profitability decay) under macro shifts.
