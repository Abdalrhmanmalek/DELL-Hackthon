# Shabaka Pulse — Virtual Power Plant & Grid Stability Engine

Shabaka Pulse is an end-to-end Virtual Power Plant (VPP) demand-response and grid stability pipeline. The system optimizes the utilization of renewable energy, mitigates curtailment, and protects grid frequency stability. 

The pipeline fetches real meteorological data, forecasts generation, runs a priority-based dispatch model across clustered industrial facilities, executes a financial settlement engine, runs a digital twin frequency simulation, and exposes an interactive monitoring dashboard.

---

## Project Architecture & Core Logic

Shabaka Pulse addresses Egypt's regional transmission constraints. The combined peak capacity of the Benban Solar Park (1,800 MW) and the Gulf of Suez Wind Corridor (500 MW) is small relative to national demand (20,000–38,000 MW), but local transmission bottlenecks block this power from reaching the national grid. 

To prevent curtailment, Shabaka Pulse dynamically redirects surplus renewable energy to a Virtual Power Plant consisting of 30 heavy industrial facilities (hydraulic pumping, desalination, thermal cold storage, cement/steel batch processing).

```
[ NASA POWER API ] (Benban & Suez weather)
       |
       v
[ Meteorological Data Prep ] (Hub-height wind, time features)
       |
       v
[ XGBoost Forecasting Models ] (1.5x scaled capacity simulation)
       |
       v
[ Surplus Calculation ] (Local headroom constraint)
       |
       +-----------------------+-------------------------+
       |                       |                         |
       v                       v                         v
[ Priority Dispatch ]    [ Settlement Engine ]    [ Digital Twin Sim ]
(Tiers 1 -> 2 -> 3)      (Waterfall payment)      (Swing equation)
       |                       |                         |
       +-----------------------+-------------------------+
                               |
                               v
                     [ Streamlit Dashboard ]
                   (EETC Control Room & Portal)
```

---

## Repository Structure

```
shabaka-pluse/
├── README.md                      # Primary project documentation
├── nasa_power_data_prep.ipynb      # Step 1 — Weather data prep
├── train_forecast_models.ipynb     # Step 2 — XGBoost model training
├── build_facility_dataset.ipynb    # Step 3 — Facility dataset creation
├── facility_clustering.ipynb       # Step 4 — K-Means tiering
├── dispatch_solver.ipynb           # Step 5 — Priority dispatch verification
├── compute_surplus.ipynb           # Step 6 — Scaled surplus computation
├── settlement_engine.ipynb         # Step 7 — Financial settlement ledgers
├── digital_twin_sim.ipynb          # Step 8 — Grid frequency simulation
├── app.py                          # Step 9 — Streamlit dashboard
├── data/
│   ├── raw_benban.csv              # NASA API solar response
│   ├── raw_suez.csv                # NASA API wind response
│   ├── nasa_power_training_data.csv# Merged training dataset
│   ├── facilities.csv              # 30 synthetic VPP facilities
│   ├── facilities_clustered.csv    # Facilities with K-Means tiers
│   ├── surplus_forecast.csv        # Scaled full-year hourly surplus
│   └── settlement_ledgers.json     # Settlement financial outputs
├── docs/
│   ├── solar_model_pipeline.md     # Solar pipeline detailed guide
│   ├── wind_model_pipeline.md      # Wind pipeline detailed guide
│   └── dashboard_guide.md          # Streamlit dashboard guide
└── models/
    ├── solar_model.json            # Trained XGBoost solar weights
    ├── wind_model.json             # Trained XGBoost wind weights
    └── feature_config.json         # Feature schema & capacity metadata
```

---

## Pipeline Execution Details

### Step 1 — `nasa_power_data_prep.ipynb` (Meteorological Prep)
Fetches real hourly meteorological data for the full year 2025 using NASA's public POWER API.
- **Solar location**: Benban Solar Park coordinates (24.40 N, 32.95 E). Fetches Global Horizontal Irradiance (GHI), Direct Normal Irradiance (DNI), clear-sky GHI, and air temperature.
- **Wind location**: Gulf of Suez corridor (28.35 N, 33.10 E). Fetches 10m wind speed, 50m wind speed, 50m wind direction, and air temperature.
- **Calculations**: Extrapolates wind speed to 100m hub-height (`ws100m`) using the Wind Power Law exponent ($\alpha = 0.14$).
- **Output**: `data/nasa_power_training_data.csv` (8,760 hours).

### Step 2 — `train_forecast_models.ipynb` (Forecasting Models)
Trains XGBoost regressors to model generation profiles.
- **Physics Labels**:
  - **Solar**: Linear GHI derate of 0.82 to model system efficiency, capped at 1,800 MW (Benban nameplate), with Gaussian noise ($\sigma=15$ MW).
  - **Wind**: Piecewise turbine power curve (cut-in 3 m/s, rated 12 m/s, cut-out 25 m/s) capped at 500 MW (Suez nameplate), with Gaussian noise ($\sigma=8$ MW).
- **Parameters**: `XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05)`, split chronologically at `2025-11-01`.
- **Output**: `models/solar_model.json`, `models/wind_model.json`, and `models/feature_config.json`.

### Step 3 — `build_facility_dataset.ipynb` (VPP Facility Profile)
Generates 30 synthetic Egyptian industrial facilities.
- **Parameters**: Geographic coordinates jittered around real industrial hubs (Aswan, Suez/Ain Sokhna, Tenth of Ramadan, Borg El Arab).
- **Flexibility**: Assigns maximum flexible capacity, ramp rates, and energy storage types (hydraulic, thermal, batch).
- **Output**: `data/facilities.csv`.

### Step 4 — `facility_clustering.ipynb` (K-Means Tiering)
Clusters the 30 facilities using K-Means on normalized flexibility features (`max_flex_mw` and `ramp_rate_mw_per_min`).
- **Validation**: Elbow method and cross-tabulation validate the clustering against physical energy storage classes.
- **Tiers**:
  - **Tier 1 (Hydraulic)**: Water pumping/desalination. High ramp rate, zero process damage risk. Served first.
  - **Tier 2 (Thermal)**: Cold storage. Moderate ramp rate. Served second.
  - **Tier 3 (Batch)**: Cement/steel. Slow ramp rate, high interruption cost. Served last.
- **Output**: `data/facilities_clustered.csv`.

### Step 5 — `dispatch_solver.ipynb` (Priority Dispatch Solver)
Implements and validates the surplus allocation solver.
- **Algorithm**: Surplus MW is allocated tier by tier: Tier 1 -> Tier 2 -> Tier 3. Within each tier, capacity is saturated in order of `facility_id`.
- **Validation**: Confirms energy conservation (`allocated + remaining = input`) and priority constraints across multiple surplus scenarios.

### Step 6 — `compute_surplus.ipynb` (Surplus & Capacity Expansion)
Runs full-year hourly predictions and calculates grid surplus.
- **Capacity Expansion**: Simulates a 50% capacity expansion. Solar and wind predictions are scaled by **1.5** inside the Pandas DataFrame before calculating total generation and surplus.
- **Local Transmission Proxy**: National demand is modeled as a diurnal and seasonal curve (20,000–38,000 MW). Local headroom limit is defined as `HEADROOM_FRACTION = 0.09` (9% of national demand) with a 400 MW floor.
- **Calculations**: `surplus_mw = max(0, solar_pred_mw + wind_pred_mw - local_headroom_mw)`.
- **Output**: `data/surplus_forecast.csv`.

### Step 7 — `settlement_engine.ipynb` (Financial Settlement)
Computes avoided costs and distributes financial credits for a sample peak winter week (Jan 1 - Jan 7).
- **Avoided Cost Rationale**:
  - **PPA Offset**: Avoided solar/wind contract cost of **1,400 EGP/MWh**.
  - **CCGT Fuel Saving**: preserved gas value calculated using a CCGT heat rate of **7.5 MMBtu/MWh**, LNG spot price of **$10.50/MMBtu**, and an exchange rate of **50 EGP/USD** (equivalent to **3,937.50 EGP/MWh**).
  - **Gross Value**: **5,337.50 EGP/MWh** of absorbed surplus.
- **Waterfall Split**:
  - **50% Factory Discount Credit**: Paid to industrial partners as a billing discount for absorbing power.
  - **40% Treasury Retained Savings**: Retained by the state treasury for avoided subsidy and PPA costs.
  - **10% Platform Operator Fee**: Distributed to the VPP administrator.
- **Output**: `data/settlement_ledgers.json`.

### Step 8 — `digital_twin_sim.ipynb` (Grid Stability Digital Twin)
Models the dynamic grid frequency response using the Swing Equation:
$$\frac{df}{dt} = \frac{f_0}{2H} \cdot \frac{P_{gen} - P_{load}}{S_n}$$
- **Parameters**: Nominal frequency $f_0 = 50.0$ Hz, inertia constant $H = 4.5$ s, system capacity $S_n = 59,700$ MVA.
- **Interlock Trigger**: At $t = 10$ s, a generator outage of 1,800 MW is injected. When the grid frequency falls below the safety threshold of **49.8 Hz**, the VPP fast interlock system triggers, shedding **300 MW** of Tier-1 industrial load in under 100 ms to stabilize frequency decay.

### Step 9 — `app.py` (Streamlit Dashboard)
An interactive dashboard displaying the virtual power plant. Run using:
`streamlit run app.py`

- **EETC Control Room**: Displays real-time grid metrics, full-year solar and wind generation, transmission limits, diurnal/seasonal heatmaps, the digital twin interlock simulation, and an interactive Folium map showing the geographic location and tier status of the 30 active VPP facilities.
- **Dispatch Simulator**: Allows users to select any hour of the year or input a custom surplus MW value to visualize the priority dispatch solver. Displays active facilities, unallocated surplus, a stacked bar chart of allocations by tier, and a tier-wise capacity utilization table.
- **Industrial Partner Portal**: Allows factory managers to select their facility, adjust their flexible capacity commitment slider, and view their dynamic, scaled settlement ledger credit.

---

## Technical Stack

| Domain | Technologies |
|---|---|
| Core Language | Python 3.9+ |
| Data Processing | pandas, numpy |
| API Integration | NASA POWER REST API |
| Machine Learning | XGBoost, scikit-learn |
| Web Application | Streamlit, Streamlit Folium |
| Visualization | Plotly, Folium, matplotlib |

---

## Complete Notebook Execution Order

Run the notebooks in this order on a clean environment to generate all data:

```
1. nasa_power_data_prep.ipynb    → creates data/raw_benban.csv, raw_suez.csv, and nasa_power_training_data.csv
2. train_forecast_models.ipynb   → creates models/solar_model.json, wind_model.json, and feature_config.json
3. build_facility_dataset.ipynb  → creates data/facilities.csv
4. facility_clustering.ipynb     → creates data/facilities_clustered.csv
5. dispatch_solver.ipynb         → validates dispatch logic (runs tests in memory)
6. compute_surplus.ipynb         → creates data/surplus_forecast.csv (runs 1.5x scaled capacity simulation)
7. settlement_engine.ipynb       → creates data/settlement_ledgers.json
8. digital_twin_sim.ipynb        → runs swing equation and interlock simulation
9. app.py                        → Streamlit dashboard reads forecast, clusters, and ledgers
```
