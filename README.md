# Predictive VM Resource Provisioning in a Cloud Data Centre

**Stack:** Python 3.12, ARIMA, XGBoost, LSTM (TensorFlow-Keras), SimPy, Streamlit, Plotly
**Dataset:** Bitbrains GWA-T-12 VM workload trace (`1.csv`)

## Problem Statement
Develop a machine learning application to forecast future VM resource requirements in a cloud
data centre. Use a suitable workload dataset and compare at least three ML models for predicting
CPU and/or memory. Evaluate the models with appropriate metrics. Integrate the selected model
with a cloud simulation environment to demonstrate proactive VM provisioning based on predicted
demand. Develop a dashboard to visualise workload patterns, predictions, VM provisioning decisions
and resource utilisation.

## Project Overview
A cloud provider must keep enough VMs running to serve demand, but too many VMs waste money and
too few break the SLA. A reactive system adds VMs only after the load has already increased.
This project forecasts CPU demand five minutes ahead with machine learning, so that VMs can be
added **before** the spike arrives (proactive provisioning). The whole flow is:

`Dataset -> Preprocessing -> Train 3 models -> Compare metrics -> Select best -> Cloud simulation -> Dashboard`

---

## Task 1 - Workload Dataset and Forecasting Application

**Dataset.** The Bitbrains GWA-T-12 trace comes from a real managed-hosting data centre. Each row is
a 5-minute sample of one VM with CPU usage (MHz and %), CPU capacity, memory usage, memory capacity,
disk throughput and network throughput. The loaded file has **8,634 rows**.

**Why this dataset.** It contains real CPU and memory measurements per VM at a regular interval,
which is what time-series forecasting needs. It is small enough to run on a laptop.

**Preprocessing.** The raw file is semicolon-separated, so it is split into proper columns, the
timestamp is converted to datetime, the data is cleaned, scaled and converted into sequences for
the models. Train and test sets are split **chronologically** (no shuffling) so the future never
leaks into training.

**Workload observations.**

| Statistic | Value |
|-----------|-------|
| Average CPU usage | 4.0 % |
| Peak CPU usage | 97.9 % |
| Minimum CPU usage | 0.5 % |

The workload is bursty: long idle periods with sudden CPU and memory spikes. This is the case where
a reactive system reacts too late and where forecasting is useful.

*Dashboard Section A - historical workload patterns (CPU and memory), with average, peak, minimum CPU and total records.*

<img width="959" height="538" alt="Screenshot 2026-10-05 152829" src="https://github.com/user-attachments/assets/6ab3b1b1-2e56-43f9-9c14-d8f69627923a" />

---

## Task 2 - Comparison of Three ML Models

| Model | Type | Role |
|-------|------|------|
| ARIMA | Statistical | Baseline |
| XGBoost Regressor | Gradient-boosted trees | Machine learning model |
| LSTM Neural Net | Recurrent deep learning | Sequence model |

All three models are trained on the same training data and tested on the same unseen test set to
predict the next CPU utilisation value.

- **ARIMA** assumes a linear, stationary pattern. On this spiky data it predicts an almost constant
  value and cannot follow the spikes.
- **XGBoost** captures non-linear relationships and follows the spikes, but it also produces false
  small peaks.
- **LSTM** learns the time dependency across previous steps and follows the real spike most closely.

*Dashboard Section B - model leaderboard and actual vs predicted CPU utilisation on the test window.*

<img width="959" height="537" alt="Screenshot 2026-10-05 152839" src="https://github.com/user-attachments/assets/7590b5e1-fc78-46ca-91cd-0718cf846ef8" />


*Static comparison chart saved by `train_models.py`.*

<img width="959" height="539" alt="Screenshot 2026-10-05 152859" src="https://github.com/user-attachments/assets/ac645c6b-7e78-43c9-8e32-6c97c23a8cf5" />


---

## Task 3 - Evaluation with Prediction Metrics

| Model | RMSE | MAE | MAPE (%) |
|-------|------|-----|----------|
| ARIMA Baseline | 31.5995 | 29.9825 | 3749.72 |
| XGBoost Regressor | 9.7230 | 2.4174 | 58.10 |
| **LSTM Neural Net** | **8.2858** | **1.6094** | **52.91** |

- **RMSE** (root mean squared error) penalises large errors, which matters because missing a spike causes SLA violations.
- **MAE** (mean absolute error) is the average error in CPU percentage points.
- **MAPE** (mean absolute percentage error) is the relative error. It is very high here because the CPU
  is close to 0 % most of the time, so small absolute errors become huge percentages. RMSE and MAE are
  therefore the more reliable metrics for this dataset.

**Result:** LSTM has the lowest RMSE (8.29) and MAE (1.61), about 74 % lower RMSE than ARIMA and 15 %
lower than XGBoost. **LSTM is selected** for the simulation.

*Model evaluation leaderboard with RMSE, MAE and MAPE (%).*

<img width="959" height="539" alt="Screenshot 2026-10-05 152844" src="https://github.com/user-attachments/assets/df58ef1e-f23d-4b30-b12f-07ba55ff8f4e" />


---

## Task 4 - Cloud Simulation and Proactive VM Provisioning

A SimPy discrete-event simulation models a data centre of physical hosts running VMs. Every 5-minute
step the controller decides how many hosts to keep active. Two controllers are compared:

| Controller | Decision is based on |
|------------|----------------------|
| **Predictive (AI)** | LSTM forecast of the next step, so capacity is added before the spike |
| **Reactive (baseline)** | Current observed demand only |

Each step produces one decision: `HOLD`, `SCALE_UP +n` or `SCALE_DOWN -n`. The dashboard shows the
actual demand, predicted demand, provisioned capacity, active hosts and VMs, SLA violations and a
log of every decision. An interactive spike simulator lets the user choose a base load and a spike
size and compare how both controllers respond.

**Result of the run.** The predictive controller made 33 correct early scale-ups and wasted 737 fewer
CPU units than the reactive controller. The LSTM sometimes under-predicts at the very start of a
sharp spike, which caused SLA violations (see Results Summary).

*Dashboard Section C - CPU demand vs provisioned capacity and active hosts/VMs over time.*

<img width="956" height="442" alt="Screenshot 2026-10-05 152923" src="https://github.com/user-attachments/assets/a09f5a83-cfb4-4a12-b77b-c1be817ec7eb" />


*Scaling decision breakdown and live decision log.*

<img width="959" height="406" alt="Screenshot 2026-10-05 152935" src="https://github.com/user-attachments/assets/b22e91f4-b61b-4974-a0d7-de4fa7e1ac29" />


---

## Task 5 - Dashboard

The Streamlit dashboard (`dashboard.py`) visualises everything:

| Section | What it shows |
|---------|---------------|
| Header KPIs | Best model, forecast RMSE, SLA violations, CPU waste saved, early scale-ups |
| A | Historical workload patterns (CPU and memory) |
| B | Model leaderboard and actual vs predicted curves |
| C | VM provisioning decisions, scaling breakdown and decision log |
| D | Cluster resource utilisation, predictive vs reactive |
| E | Interactive workload spike simulator |

The sidebar can re-run the full pipeline, change the display window, select which forecast models to
plot and switch between predictive and reactive mode.

*Dashboard overview with the KPI cards (best model, forecast RMSE, SLA violations, CPU waste saved, early scale-ups).*

<img width="959" height="500" alt="Screenshot 2026-10-05 152810" src="https://github.com/user-attachments/assets/0907f664-1385-49d6-bbc2-1bc0928b931c" />


*Dashboard Section D - cluster resource utilisation, predictive vs reactive.*

<img width="959" height="539" alt="Screenshot 2026-10-05 152943" src="https://github.com/user-attachments/assets/f8740a72-8f10-4e60-971b-d0f2c39af4de" />


### Section E - Interactive Workload Spike Simulator

This section lets the user create a traffic burst manually and see how both controllers react to it
in real time, without waiting for a spike in the dataset.

**Inputs (sliders)**
- **Base Cluster Utilisation (%)** - the normal load on the cluster before the spike (example: 50 %).
- **Simulated Spike (+% CPU)** - the extra CPU demand added by the burst (example: +30 %).
- **Trigger Spike** button - applies the spike and updates all outputs.

**Outputs**

| Output | Meaning | Example run |
|--------|---------|-------------|
| Actual Spike Demand | Base load plus spike | 80.0 % |
| ML Forecast (Prediction) | Demand predicted by the model, with its error | 79.4 % (-0.6 % error) |
| AI Hosts Provisioned | Hosts added by the proactive (look-ahead) controller | 10 |
| Reactive Hosts | Hosts added by the reactive (current-demand) controller | 10 |
| Provisioned Capacity | Capacity available to absorb the demand, per controller | 100 % each |
| SLA status | Whether the demand is absorbed by the capacity | SLA MAINTAINED (80 % demand absorbed by 100 % capacity) |

A bar chart compares Actual Demand, AI Prediction, AI Provisioned Capacity and Reactive Capacity, so the
gap between demand and provisioned capacity is visible at a glance. If the spike is larger than the
provisioned capacity, the section shows an SLA breach instead.

**Purpose.** It demonstrates the idea of the project interactively: the predictive controller sizes the
cluster from the forecast (look-ahead), while the reactive controller sizes it only from the demand
it currently sees.

<img width="956" height="505" alt="Screenshot 2026-10-05 153002" src="https://github.com/user-attachments/assets/58c6fb60-5a77-4c9c-b04b-0cba8730d233" />


---

## Results Summary

| Metric | Predictive (AI) | Reactive (baseline) |
|--------|-----------------|---------------------|
| Average utilisation | 11.3 % | 10.9 % |
| Wasted CPU units | 32,707 | 33,444 |
| SLA violations | 13 | 0 |
| Correct early scale-ups | 33 | - |

The predictive controller uses resources slightly better, but in this run it had more SLA violations
than the reactive baseline.

## Limitations and Future Work
- Only one VM trace is used. Aggregating many Bitbrains VMs would be more representative of a data centre.
- The LSTM under-predicts at the start of sharp spikes. A safety margin on the forecast (for example 1.2x
  the predicted demand) or multi-step forecasting can reduce SLA violations.
- MAPE is unreliable when the true value is near zero.
- Models and simulation target CPU. Memory is analysed in the workload charts only.

## Project Structure
```
cloud_comp/
|-- 1.csv                      Bitbrains trace
|-- data_prep.py               preprocessing
|-- train_models.py            ARIMA, XGBoost, LSTM training and comparison
|-- cloud_sim.py               SimPy simulation
|-- run_pipeline.py            runs the full pipeline
|-- dashboard.py               Streamlit dashboard
|-- requirements.txt
`-- *.joblib, *.pkl, *.keras   saved models and results
```

## How to Run
```powershell
venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py
streamlit run dashboard.py
```
Dashboard opens at http://localhost:8501
