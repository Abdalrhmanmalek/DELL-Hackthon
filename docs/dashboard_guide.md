# Streamlit Dashboard — How It Works

**Script:** [`app.py`](../app.py)  
**Launch command:** `streamlit run app.py`  
**Default URL:** http://localhost:8501

---

## What the Dashboard Is

The Shabaka Pulse dashboard is an interactive web application built with [Streamlit](https://streamlit.io). It has two views — one for the Egyptian grid operator (EETC) and one for industrial facility managers — and both views read from pre-computed CSV and JSON files. The dashboard does not train models or run heavy computations at render time. Everything displayed was computed by the upstream notebooks.

---

## How It Loads Data

When the app starts, it calls a single cached data loader function:

```python
@st.cache_data
def load_data():
    surplus_df  = pd.read_csv("data/surplus_forecast.csv", ...)   # 8,760 rows
    facilities_df = pd.read_csv("data/facilities_clustered.csv")  # 30 rows
    ledgers     = json.load(open("data/settlement_ledgers.json")) # list of dicts
    return surplus_df, facilities_df, ledgers
```

`@st.cache_data` means this function runs **only once** when the app first loads, and the results are held in memory for all subsequent interactions. If a user switches views, resizes the window, or adjusts a slider, the CSV files are NOT re-read from disk.

### Files the Dashboard Reads

| File | Produced By | What It Contains |
|---|---|---|
| `data/surplus_forecast.csv` | `compute_surplus.ipynb` | Hourly solar/wind predictions, headroom proxy, and surplus MW for all 8,760 hours |
| `data/facilities_clustered.csv` | `facility_clustering.ipynb` | The 30 facilities with their assigned operational tier (1, 2, or 3) |
| `data/settlement_ledgers.json` | `settlement_engine.ipynb` | Per-facility billing credits for the sample winter week |

If `settlement_ledgers.json` does not exist yet (i.e., the settlement engine notebook has not been run), the dashboard silently catches the `FileNotFoundError` and sets `ledgers = []`, so the app still loads without crashing.

---

## Navigation

The sidebar contains a radio button toggle:

```
Navigation Mode
  ( ) EETC Control Room
  ( ) Industrial Partner Portal
```

The entire page body changes based on which radio option is selected. This is implemented with a simple `if / elif` block in Python — there is no routing library involved.

---

## View 1: EETC Control Room

This view is designed for a grid operator who needs to understand the current state of renewable surplus and system stability at a glance.

### Section 1 — Metric Cards (Top Row)

Four HTML metric cards displayed side-by-side using `st.columns(4)`.

| Card | Where the Number Comes From |
|---|---|
| **Grid Frequency** | Hardcoded at `50.00 Hz` — represents the baseline nominal state |
| **Peak Surplus MW** | `surplus_df.loc["2025-01-01":"2025-01-07"]["surplus_mw"].max()` — maximum hourly surplus MW during the sample winter week |
| **MWh Absorbed (Week)** | `sum(ledger["mwh_absorbed"] for ledger in ledgers)` — total energy absorbed across all facilities from the settlement ledgers |
| **Gas Value Saved** | `sum(ledger["gross_savings_egp"] for ledger in ledgers)` in millions of EGP |

### Section 2 — Generation vs. Headroom Chart

A Plotly line chart showing three things simultaneously over the sample winter week:

1. **Blue line** → `total_pred_mw`: Combined solar + wind generation forecast
2. **Red dashed line** → `local_headroom_mw`: The transmission corridor's absorption threshold
3. **Green filled area** → Where `total_pred_mw > local_headroom_mw`: This is the curtailable surplus

The green fill is drawn by adding two invisible Scatter traces with `fill='tonexty'` — a standard Plotly technique for shading between two lines.

### Section 3 — Frequency Interlock Chart

A Plotly line chart generated **live inside the app** (not from a file) by running the digital twin swing equation simulation in about 20ms. This is fast enough to do on every page render.

The simulation produces the frequency trajectory showing:
- `t = 0 to 10s`: Grid nominal at 50.00 Hz
- `t = 10s`: 1,800 MW generation drop injected
- `t = interlock trigger`: Frequency crosses 49.8 Hz threshold, 300 MW of Tier-1 load shed
- `t = 10s to 30s`: Frequency stabilizes at a new equilibrium

### Section 4 — Facility Map

A [Folium](https://python-visualization.github.io/folium/) interactive map rendered inside Streamlit using the `streamlit-folium` library.

Each facility is plotted as a `CircleMarker`:
- **Position**: `lat` / `lon` from `facilities_clustered.csv`
- **Color**: Tier 1 = blue, Tier 2 = orange, Tier 3 = red
- **Size**: `max_flex_mw / 5` — larger circles = more flexible capacity
- **Popup**: Facility name, tier, flex MW, and storage type (click any dot)

---

## View 2: Industrial Partner Portal

This view is designed for a factory manager who wants to see their billing credit and adjust their commitment.

### Facility Selector

```python
selected_name = st.selectbox("Select Facility", facility_names)
fac_data = facilities_df[facilities_df["facility_name"] == selected_name].iloc[0]
```

The dropdown lists all 30 facility names from `facilities_clustered.csv`. Selecting one re-renders the entire page body below it.

### Capacity Commitment Slider

```python
offered_mw = st.slider(
    "Flexible Capacity Offered (MW)",
    min_value=0.0,
    max_value=float(fac_data["max_flex_mw"]),   # upper bound is this facility's maximum
    value=float(fac_data["max_flex_mw"]),        # defaults to full capacity
    step=1.0
)
```

The slider adjusts between 0 and the facility's `max_flex_mw`. Currently this value is displayed but not yet wired to recalculate earnings — that would be a next-step feature (real-time partial commitment recalculation).

### Settlement Ledger Table

The app searches the loaded ledgers for the selected facility's ID:

```python
fac_ledger = next(
    (item for item in ledgers if item["facility_id"] == fac_data["facility_id"]),
    None
)
```

If a ledger is found, it displays a 4-row table showing:

| Row | What It Shows |
|---|---|
| Billing Period | `"2025-01-01 to 2025-01-07"` |
| Total Renewable Energy Absorbed | MWh absorbed by this facility during the week |
| Gross System Savings | Combined avoided PPA cost + preserved LNG value (EGP) |
| Factory Billing Credit (50%) | The factory's share of the gross savings after the waterfall split |

If no ledger exists for the selected facility (it absorbed zero MWh during the sample week), a warning message is displayed instead.

---

## Theming

The dark theme is applied using a custom CSS block injected with `st.markdown(..., unsafe_allow_html=True)`:

```css
.stApp              → Sets page background to #0E1117 (GitHub dark)
h1, h2, h3          → Headers forced to #58A6FF (GitHub blue)
.metric-card        → Custom div with dark panel and green top border
.metric-value       → Large bold numbers in #39D353 (GitHub green)
.metric-label       → Subdued grey label text
```

Plotly charts use `template="plotly_dark"` and transparent backgrounds so they blend into the dark page.

---

## Full Data Flow Into the Dashboard

```
NASA POWER API
     |
     v
nasa_power_data_prep.ipynb
     |
     v
data/nasa_power_training_data.csv
     |
     +------> train_forecast_models.ipynb
     |               |
     |               v
     |        models/solar_model.json
     |        models/wind_model.json
     |        models/feature_config.json
     |               |
     +------> compute_surplus.ipynb <-------+
                     |                      |
                     v                      |
           data/surplus_forecast.csv        |
                     |                      |
                     v                      |
           build_facility_dataset.ipynb     |
                     |                      |
                     v                      |
           data/facilities.csv             |
                     |                      |
                     v                      |
           facility_clustering.ipynb        |
                     |                      |
                     v                      |
           data/facilities_clustered.csv    |
                     |                      |
                     +----------------------+
                     |
                     v
           settlement_engine.ipynb
                     |
                     v
           data/settlement_ledgers.json
                     |
                     v
                  app.py
              (reads all three files,
               renders dashboard at
               http://localhost:8501)
```

---

## Startup Checklist

Before running the dashboard, make sure all upstream notebooks have been executed in order:

| Notebook | Required Output |
|---|---|
| `nasa_power_data_prep.ipynb` | `data/nasa_power_training_data.csv` |
| `train_forecast_models.ipynb` | `models/solar_model.json`, `models/wind_model.json`, `models/feature_config.json` |
| `build_facility_dataset.ipynb` | `data/facilities.csv` |
| `facility_clustering.ipynb` | `data/facilities_clustered.csv` |
| `compute_surplus.ipynb` | `data/surplus_forecast.csv` |
| `settlement_engine.ipynb` | `data/settlement_ledgers.json` |

Then launch the dashboard:

```bash
streamlit run app.py
```
