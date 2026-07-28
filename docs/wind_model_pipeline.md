# Wind Forecasting Model — Data Pipeline

**Notebook:** [`train_forecast_models.ipynb`](../train_forecast_models.ipynb)  
**Saved model:** [`models/wind_model.json`](../models/wind_model.json)

---

## What This Model Does

It predicts hourly wind power output (in MW) for the **Gulf of Suez wind corridor** near Ras Ghareb, Egypt, which has a nameplate capacity of **500 MW**. Given real wind speed readings for any hour, the model outputs an estimated generation number between 0 and 500 MW.

---

## Step-by-Step Data Pipeline

### Step 1 — Raw Data Source (NASA POWER API)

The weather data comes from **NASA's public POWER API** for GPS coordinates of the Gulf of Suez wind corridor (28.35°N, 33.10°E).

**Parameters fetched from NASA:**

| NASA Parameter Code | What It Measures | Unit |
|---|---|---|
| `WS10M` | Wind speed at 10 metres above ground | m/s |
| `WS50M` | Wind speed at 50 metres above ground (standard met mast height) | m/s |
| `WD50M` | Wind direction at 50 metres | degrees (0–360) |
| `T2M` | Air temperature at 2 metres above ground | °C |

Raw API response is cached to `data/raw_suez.csv` so it doesn't need to be re-fetched.

---

### Step 2 — Hub-Height Wind Speed Extrapolation

Real wind turbines have hubs at **100 metres** above ground, but NASA only provides wind data up to 50m. We extrapolate to 100m using the **Wind Power Law**:

```
ws100m = ws50m × (100 / 50) ^ α
```

Where **α = 0.14** is the standard atmospheric surface roughness exponent for flat desert terrain (the Hellmann exponent). This gives us a more accurate input for the turbine power curve calculation.

| Height | Source | Column Name |
|---|---|---|
| 10m | NASA direct measurement | `ws10m` |
| 50m | NASA direct measurement | `ws50m` |
| 100m | Power-law extrapolated | `ws100m` |

---

### Step 3 — Feature Engineering

After loading the raw data, we compute additional time-based features (same as solar):

| New Column | How It's Computed | Why It Matters |
|---|---|---|
| `hour_of_day` | UTC hour (0–23) | Wind patterns in the Gulf of Suez have a strong daily cycle driven by land-sea thermal gradients |
| `month` | Month number (1–12) | Gulf of Suez winds peak in winter months when the Mediterranean pressure gradient is strongest |
| `day_of_year` | Calendar day (1–365) | Smooth seasonal transitions |

---

### Step 4 — Physics-Based Label Generation (Piecewise Power Curve)

Instead of a simple linear derate (as with solar), wind turbine output is non-linear. We simulate output using a **standard piecewise turbine power curve**:

```
If wind_speed < cut_in (3 m/s):      output = 0 MW      (turbine is idle)
If cut_in ≤ wind_speed < rated (12): output = (wind_speed³ / rated³) × capacity
If rated ≤ wind_speed < cut_out (25): output = capacity    (full rated power)
If wind_speed ≥ cut_out (25):        output = 0 MW      (turbine shuts down for safety)
```

| Parameter | Value | Explanation |
|---|---|---|
| `SUEZ_CAPACITY_MW` | 500 MW | Total nameplate wind capacity in the Gulf of Suez corridor |
| `cut_in` | 3 m/s | Minimum wind speed for the turbine to start producing power |
| `rated` | 12 m/s | Wind speed at which maximum output is reached |
| `cut_out` | 25 m/s | Wind speed at which turbines shut down to protect equipment |
| Gaussian noise σ | 8 MW | Simulates turbulence, partial curtailment, and fleet-level variance |

The cubic relationship between wind speed and power output (`wind³`) is why the model needs to learn a non-linear mapping — and why XGBoost is a good fit.

---

### Step 5 — Train/Test Split

Identical to the solar model — a strict chronological split with no data leakage:

| Split | Period | Hours |
|---|---|---|
| **Training** | January 1 — October 31, 2025 | ~7,296 hours |
| **Testing** | November 1 — December 31, 2025 | ~1,464 hours |

---

### Step 6 — Model Training

**Algorithm:** XGBoost Regressor

```python
XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05
)
```

**Input features (exact order matters):**

```
ws100m       ← 100m hub-height wind speed (strongest predictor)
ws50m        ← 50m measured wind speed
t2m_suez     ← Air temperature (affects air density, which affects power)
hour_of_day  ← Time of day
month        ← Season
```

Note: `ws100m` is the most important feature because it directly represents the wind speed experienced by the turbine rotor. Temperature matters because colder, denser air produces more power from the same wind speed.

---

### Step 7 — Evaluation

Evaluated on held-out November–December test data:

| Metric | What It Measures |
|---|---|
| **MAE (MW)** | Average absolute prediction error in megawatts |
| **RMSE (MW)** | Root mean squared error — penalizes large misses more |
| **R²** | How much of the output variance the model explains |
| **Capacity-Normalized MAE (%)** | MAE as a percentage of total 500 MW capacity |

---

### Step 8 — Model Saved

```
models/wind_model.json      ← The trained model weights (~655 KB)
models/feature_config.json  ← Feature names + capacity constants
```

---

## Data Flow Diagram

```
NASA POWER API (28.35°N, 33.10°E)
        |
        v
data/raw_suez.csv            ← Cached raw API response
        |
        v
Power Law Extrapolation      ← ws100m = ws50m × (100/50)^0.14
        |
        v
Feature Engineering          ← Add hour_of_day, month, day_of_year
        |
        v
Piecewise Power Curve Labels ← wind_mw = f(ws100m, cut_in=3, rated=12, cut_out=25)
        |
        v
data/nasa_power_training_data.csv   (merged with solar, 8,760 rows)
        |
        v
XGBRegressor.fit(X_train, y_wind)
        |
        v
models/wind_model.json       ← Ready for inference
```

---

## How to Use the Model for Inference

```python
import xgboost as xgb
import pandas as pd
import json

# Load model and feature list
model = xgb.XGBRegressor()
model.load_model("models/wind_model.json")

with open("models/feature_config.json") as f:
    config = json.load(f)

wind_features = config["wind_features"]   # exact column list, exact order

# Load data and predict
df = pd.read_csv("data/nasa_power_training_data.csv", index_col="timestamp_utc", parse_dates=True)
predictions = model.predict(df[wind_features])
predictions_clipped = predictions.clip(0, 500)   # enforce physical bounds
```

---

## Key Difference from Solar Model

| Aspect | Solar Model | Wind Model |
|---|---|---|
| Primary driver | GHI irradiance | ws100m wind speed |
| Power relationship | Roughly linear with derate | Cubic (speed³) — very non-linear |
| Zero-output condition | Night hours | Below cut-in speed OR above cut-out speed |
| Typical variability | Smooth daily arc | High hour-to-hour variance (wind is gusty) |
| Physics label type | Simple derate factor | Piecewise power curve |
