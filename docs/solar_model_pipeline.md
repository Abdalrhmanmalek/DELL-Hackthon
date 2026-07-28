# Solar Forecasting Model — Data Pipeline

**Notebook:** [`train_forecast_models.ipynb`](../train_forecast_models.ipynb)  
**Saved model:** [`models/solar_model.json`](../models/solar_model.json)

---

## What This Model Does

It predicts hourly solar power output (in MW) for the **Benban Solar Park** in southern Egypt, which has a nameplate capacity of **1,800 MW**. Given real weather conditions for any hour, the model outputs an estimated generation number between 0 and 1,800 MW.

---

## Step-by-Step Data Pipeline

### Step 1 — Raw Data Source (NASA POWER API)

The weather data comes from **NASA's public POWER API** — no account or API key required. We request hourly data for the GPS coordinates of Benban Solar Park (24.40°N, 32.95°E) for the full year 2025.

**Parameters fetched from NASA:**

| NASA Parameter Code | What It Measures | Unit |
|---|---|---|
| `ALLSKY_SFC_SW_DWN` | All-sky surface shortwave downward irradiance (Global Horizontal Irradiance, GHI) | W/m² |
| `ALLSKY_SFC_SW_DNI` | Direct Normal Irradiance (DNI) — sunlight hitting a panel straight-on | W/m² |
| `CLRSKY_SFC_SW_DWN` | Clear-sky GHI — what GHI would be with zero cloud cover | W/m² |
| `T2M` | Air temperature at 2 metres above ground | °C |

Raw API response is cached to `data/raw_benban.csv` so it doesn't need to be fetched again.

---

### Step 2 — Feature Engineering

After loading the raw data, we compute three additional time-based features:

| New Column | How It's Computed | Why It Matters |
|---|---|---|
| `hour_of_day` | Extract the UTC hour (0–23) from the timestamp | Solar output follows a strong daily cycle — zero at night, peaks at noon |
| `month` | Extract the month number (1–12) from the timestamp | Winter days are shorter; summer irradiance is higher |
| `day_of_year` | Calendar day number (1–365) | Captures the continuous seasonal arc between months |

These features are merged into the main training dataset `data/nasa_power_training_data.csv`.

---

### Step 3 — Physics-Based Label Generation

The NASA API gives us weather data, not actual power meter readings. So we simulate what the solar farm's power output would be using a physics-based formula:

```
raw_solar_mw = GHI / 1000 * BENBAN_CAPACITY_MW * DERATE_FACTOR
solar_mw = clip(raw_solar_mw + noise, 0, 1800)
```

| Parameter | Value | Explanation |
|---|---|---|
| `BENBAN_CAPACITY_MW` | 1,800 MW | Total nameplate capacity of Benban Solar Park |
| `DERATE_FACTOR` | 0.82 | Real-world efficiency losses (heat, dust, inverter losses, wiring) |
| Gaussian noise σ | 15 MW | Simulates cloud fluctuations and sensor variance |

The result is a realistic hourly `solar_mw` label for each of the 8,760 hours of 2025.

---

### Step 4 — Train/Test Split

The data is split **chronologically**, not randomly. This is important because future data should never influence the training of a time-series model.

| Split | Period | Hours |
|---|---|---|
| **Training** | January 1 — October 31, 2025 | ~7,296 hours |
| **Testing** | November 1 — December 31, 2025 | ~1,464 hours |

---

### Step 5 — Model Training

**Algorithm:** XGBoost Regressor (Gradient Boosted Decision Trees)

```python
XGBRegressor(
    n_estimators=200,   # 200 boosting rounds
    max_depth=5,        # each tree can go 5 levels deep
    learning_rate=0.05  # small steps, lower overfitting risk
)
```

**Input features (exact order matters):**

```
allsky_sfc_sw_dwn   ← GHI (strongest predictor)
allsky_sfc_sw_dni   ← DNI
clrsky_sfc_sw_dwn   ← Clear-sky GHI
t2m_benban          ← Temperature (panels lose efficiency when hot)
hour_of_day         ← Time of day
month               ← Season
```

**Why XGBoost?**
- Handles the non-linear relationship between irradiance and output without needing polynomial transformations.
- Automatically captures interactions (e.g., high temperature + high irradiance behaves differently from either alone).
- Trains fast enough to retrain if new NASA data is pulled.

---

### Step 6 — Evaluation

After training, the model is evaluated on the held-out November–December test data:

| Metric | What It Measures |
|---|---|
| **MAE (MW)** | Average absolute prediction error in megawatts |
| **RMSE (MW)** | Root mean squared error — penalizes large misses more |
| **R²** | How much of the output variance the model explains (1.0 = perfect) |
| **Capacity-Normalized MAE (%)** | MAE as a percentage of total 1,800 MW capacity |

---

### Step 7 — Model Saved

The trained model is saved in native XGBoost JSON format:

```
models/solar_model.json     ← The trained model weights (~616 KB)
models/feature_config.json  ← Feature names + capacity constants
```

The feature config file is critical: when loading the model later for predictions, the input columns must be provided in **exactly the same order** as during training, or predictions will be silently wrong.

---

## Data Flow Diagram

```
NASA POWER API (24.40°N, 32.95°E)
        |
        v
data/raw_benban.csv          ← Cached raw API response
        |
        v
Feature Engineering          ← Add hour_of_day, month, day_of_year
        |
        v
Physics-Based Labels         ← solar_mw = f(GHI, derate=0.82, noise)
        |
        v
data/nasa_power_training_data.csv   (8,760 rows × 13 columns)
        |
        v
XGBRegressor.fit(X_train, y_solar)
        |
        v
models/solar_model.json      ← Ready for inference
```

---

## How to Use the Model for Inference

```python
import xgboost as xgb
import pandas as pd
import json

# Load model and feature list
model = xgb.XGBRegressor()
model.load_model("models/solar_model.json")

with open("models/feature_config.json") as f:
    config = json.load(f)

solar_features = config["solar_features"]   # exact column list, exact order

# Load data and predict
df = pd.read_csv("data/nasa_power_training_data.csv", index_col="timestamp_utc", parse_dates=True)
predictions = model.predict(df[solar_features])
predictions_clipped = predictions.clip(0, 1800)   # enforce physical bounds
```
