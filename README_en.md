## Time Series Machine Learning Project: Solar Power Generation and Electricity Demand Forecasting (Tokyo Area)

### 1. Project Overview

This project focuses on **Solar Power Generation** and **Electricity Demand** time series data, recorded at 30-minute intervals for the **TEPCO Power Grid** area. We implemented and benchmarked five diverse forecasting models—**Holt-Winters, SARIMAX, Prophet, LightGBM, and GRU**—to compare their predictive accuracy and model characteristics.

The methodology employed **Time Series Cross-Validation (Sliding Window method)** to prevent data leakage. We performed **Feature Engineering** (incorporating weather and calendar variables) and **Hyperparameter Optimization (Optuna)** to identify the optimal model and approach for the unique characteristics of each time series.

### 2. Project Workflow & Notebooks

The analysis progressed sequentially across the following nine notebooks:

| No. | Notebook | Technical Focus and Key Insights |
| :--- | :--- | :--- |
| **01** | `01_Data_Overview_and_Stationarity_Diagnostics.ipynb` | Performed **Stationarity Diagnostics (ADF, KPSS, OCSB)** to identify the need for differencing ($d=1$) for both series. |
| **02** | `02_STL_Decomposition.ipynb` | Utilized **MSTL (Multiple STL)** for time series decomposition into multiple seasonalities (Daily $P=48$, Weekly $P=336$, Annual $P=17520$), visually confirming complex structures, such as the nighttime zero values in generation data. |
| **03** | `03_Hierarchical_Pattern_Analysis_and_Data_Insight.ipynb` | Quantified the annual, weekly, and daily patterns using **time-based aggregation**. This established the rationale for using these patterns as **external features (calendar variables)** in subsequent modeling. |
| **04** | `04_Baseline_Models.ipynb` | Constructed **Holt-Winters** and **SARIMAX** models. Incorporated **Fourier terms as exogenous variables** in SARIMAX to manage the computational load associated with long seasonal periods ($S=336$). Established a strong baseline for consumption forecasting (MASE $\approx 0.51$). |
| **05** | `05_Prophet_Forecast.ipynb` | Applied the Prophet model, leveraging its built-in **component decomposition** for interpretability. Despite tuning with Optuna, the model's accuracy was limited and did not consistently surpass other baseline models. |
| **06** | `06_Energy_EDA_and_Feature_Selection.ipynb` | Added **weather and calendar variables**. Applied a multi-step feature selection process using **LASSO and SHAP**. Confirmed **solar\_radiation** as the dominant feature for generation and identified **vapor\_pressure** as a critical climate driver for consumption. |
| **07** | `07_LightGBM_Forecast.ipynb` | Built a LightGBM model utilizing the selected features. It showed good initial performance for power generation (MASE $\approx 1.85$), but hyperparameter optimization with Optuna yielded only **limited or negative accuracy improvement**, suggesting potential overfitting or flawed tuning methodology. |
| **08** | `08_GRU_Forecast.ipynb` | Implemented a **GRU** model using PyTorch. **Strict scaling** was applied exclusively to training data within each sliding window to rigorously prevent data leakage. Tuning successfully improved the generation forecast accuracy (MASE $2.77 \rightarrow 1.77$). |
| **09** | `09_Implementation_of_Custom_Evaluation_Metric_and_Final_Summary.ipynb` | Aggregated and compared the results of all five models. Introduced the **Custom Evaluation Metric: My\_Eval\_Index** ($MAE_{test}$ / $MAE_{test\_seasonaly\_naive}$) to provide a normalized, relative comparison of performance across models, mitigating the scale bias inherent in standard MASE for generation data. |

### 3. Key Technical Insights

#### 3.1 Solar Power Generation Forecasting (Challenges and Solutions)

Solar power generation forecasting faced challenges primarily due to the non-linear structure of **nighttime zero values** and strong dependence on external weather factors.

*   **Superior Models**: Models incorporating external features, specifically **LightGBM** and **GRU**, demonstrated performance superior to classical methods for generation forecasting.
*   **Feature Dominance**: **Global Solar Radiation** (`solar_radiation`) was confirmed by correlation analysis, LASSO, and SHAP to be the overwhelming and most dominant explanatory feature.

#### 3.2 Electricity Demand Forecasting (Success and Drivers)

*   **Superior Models**: **Holt-Winters** and **SARIMAX (using Fourier terms)** delivered the most stable and accurate results (MASE $\approx 0.5$ to $0.53$), effectively modeling the strong periodic structure of electricity demand.
*   **Critical Feature**: Through LASSO and SHAP analysis, **Vapor Pressure** (`vapor_pressure`) was identified as the most important climate feature. It serves as a superior indicator of HVAC (heating, ventilation, and air conditioning) demand than simple temperature or humidity, correlating strongly with human sensation (e.g., muggy heat).

#### 3.3 Evaluation and Rigor

*   **MASE and Custom Metrics**: We utilized the scale-independent **MASE** as the primary performance metric for assessing generalization capabilities. To address MASE's tendency to inflate error values for series with high seasonal correlation (like solar power generation), the novel metric **My\_Eval\_Index** was developed and applied for a fairer relative comparison.
*   **Data Leakage Prevention (GRU)**: For the deep learning model (GRU), a rigorous process was designed to prevent information leakage: the **MinMaxScaler** was fitted exclusively on the training data within each sliding window iteration, and applied to the corresponding test window features to ensure fair generalization assessment.

### 4. Technical Stack

| Area | Tool / Library |
| :--- | :--- |
| **Language** | Python 3.9+ |
| **Time Series Models** | `statsmodels` (SARIMAX, Holt-Winters), `Prophet`, `pmdarima` (OCSB) |
| **Machine Learning** | `LightGBM`, `scikit-learn` (LASSO, TimeSeriesSplit), `Optuna` (H/P Tuning) |
| **Deep Learning** | `PyTorch` (GRU) |
| **Interpretability/EDA** | `SHAP` (Feature Importance), `MSTL` Decomposition |
| **Data Processing** | `pandas`, `numpy`, `src/` (Utility Modules) |


### 5. Directory Structure

The main directory structure of this project is as follows:
```
.
├── notebooks_en/
│   ├── 01_Data_Overview_and_Stationarity_Diagnostics.ipynb (Data validation and stationarity diagnostics)
│   ├── ... (02 through 09 Jupyter Notebooks: Model building and evaluation steps)
├── src/
│   ├── data_utils.py (Data loading and preprocessing functions)
│   ├── evaluation_utils.py (Functions for MAE, MASE, RMSE, and Custom Evaluation Index)
│   ├── plot_utils.py (Functions for plotting forecast results)
│   └── forecast_utils.py (Sliding window function definition for time series analysis)
├── data/
│   ├── e_gen_demand.csv (Electricity generation and demand data)
│   ├── weather_data.csv (Weather data)
│   └── df_shifted.csv (Feature engineered data)
├── results/
│   └── preds/ (Forecast result files (.pkl) for each model)
├── requirements.txt (List of dependencies)
└── README_en.md (This file)

```
