# Contributing to Shabaka Pulse

Thank you for your interest in contributing to **Shabaka Pulse** — Egypt's Virtual Power Plant & Grid Stability Engine.
This guide will help you set up the project, understand the codebase, and submit high-quality contributions.

---

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Environment Setup](#environment-setup)
  - [Running the Dashboard](#running-the-dashboard)
- [Project Structure](#project-structure)
- [Pipeline Execution Order](#pipeline-execution-order)
- [Development Workflow](#development-workflow)
  - [Branching Strategy](#branching-strategy)
  - [Commit Messages](#commit-messages)
  - [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
  - [Python Style](#python-style)
  - [Notebooks](#notebooks)
  - [Dashboard (Streamlit)](#dashboard-streamlit)
- [Where to Contribute](#where-to-contribute)
- [Reporting Issues](#reporting-issues)
- [Code of Conduct](#code-of-conduct)

---

## Getting Started

### Prerequisites

| Requirement  | Version   |
|--------------|-----------|
| Python       | 3.9+      |
| pip          | Latest    |
| Git          | 2.x+      |
| Jupyter      | Any       |

### Environment Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/<org>/shabaka-pluse.git
   cd shabaka-pluse
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install pandas numpy xgboost scikit-learn streamlit streamlit-folium plotly folium matplotlib requests
   ```

4. **Verify the install**

   ```bash
   python -c "import streamlit; print(streamlit.__version__)"
   ```

### Running the Dashboard

The Streamlit dashboard is the main user-facing application. To launch it:

```bash
streamlit run 9-app.py
```

The dashboard has three views:

| View                        | Description                                                  |
|-----------------------------|--------------------------------------------------------------|
| **EETC Control Room**       | Full-year generation overview, surplus heatmaps, digital twin frequency simulation, and facility map |
| **Dispatch Simulator**      | Interactive priority dispatch solver — pick any hour or enter a custom surplus value |
| **Industrial Partner Portal** | Facility-level commitment slider and dynamic settlement ledger |

> **Note:** The dashboard requires pre-computed data files. If you're starting from a clean clone, run the full pipeline first (see below).

---

## Project Structure

```
shabaka-pluse/
├── 1-nasa_power_data_prep.ipynb      # Step 1 — NASA POWER weather data
├── 2-train_forecast_models.ipynb     # Step 2 — XGBoost solar/wind models
├── 3-build_facility_dataset.ipynb    # Step 3 — 30 synthetic VPP facilities
├── 4-facility_clustering.ipynb       # Step 4 — K-Means tier clustering
├── 5-dispatch_solver.ipynb           # Step 5 — Priority dispatch validation
├── 6-compute_surplus.ipynb           # Step 6 — Full-year surplus calculation
├── 7-settlement_engine.ipynb         # Step 7 — Financial settlement engine
├── 8-digital_twin_sim.ipynb          # Step 8 — Grid frequency digital twin
├── 9-app.py                          # Step 9 — Streamlit dashboard
├── data/                             # Generated datasets (CSV, JSON)
├── models/                           # Trained XGBoost model weights
├── docs/                             # Pipeline documentation guides
├── README.md                         # Project overview
└── CONTRIBUTING.md                   # ← You are here
```

---

## Pipeline Execution Order

Before working on the dashboard or downstream notebooks, you must generate the required data by running the notebooks **in order**:

```
Step  Notebook                          Outputs
───── ──────────────────────────────── ─────────────────────────────────────
  1   1-nasa_power_data_prep.ipynb     data/raw_benban.csv, raw_suez.csv,
                                       data/nasa_power_training_data.csv
  2   2-train_forecast_models.ipynb    models/solar_model.json,
                                       models/wind_model.json,
                                       models/feature_config.json
  3   3-build_facility_dataset.ipynb   data/facilities.csv
  4   4-facility_clustering.ipynb      data/facilities_clustered.csv
  5   5-dispatch_solver.ipynb          (in-memory validation — no file output)
  6   6-compute_surplus.ipynb          data/surplus_forecast.csv
  7   7-settlement_engine.ipynb        data/settlement_ledgers.json
  8   8-digital_twin_sim.ipynb         (in-memory simulation — no file output)
  9   streamlit run 9-app.py           (reads data/, models/)
```

> **Tip:** Steps 1–4 are independent of steps 5–8. If you're only working on the dashboard, you only need the output files from steps 1–4 and steps 6–7.

---

## Development Workflow

### Branching Strategy

| Branch   | Purpose                                    |
|----------|--------------------------------------------|
| `main`   | Stable, production-ready code              |
| `dev`    | Active development and integration branch  |
| `feature/<name>` | New features — branch off `dev`   |
| `fix/<name>`     | Bug fixes — branch off `dev`      |

**Workflow:**

```bash
# Create a feature branch from dev
git checkout dev
git pull origin dev
git checkout -b feature/your-feature-name

# Make changes, then push
git add .
git commit -m "feat: add dispatch latency chart to Control Room"
git push origin feature/your-feature-name
```

### Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

| Prefix     | Use Case                              | Example                                           |
|------------|---------------------------------------|----------------------------------------------------|
| `feat:`    | New feature                           | `feat: add monthly surplus breakdown chart`        |
| `fix:`     | Bug fix                               | `fix: correct wind ramp-rate unit conversion`      |
| `docs:`    | Documentation only                    | `docs: update dashboard guide with new screenshots`|
| `refactor:`| Code restructuring (no behavior change)| `refactor: extract dispatch solver to module`      |
| `style:`   | Formatting, whitespace                | `style: apply black formatter to app.py`           |
| `test:`    | Adding or updating tests              | `test: add surplus conservation assertions`        |
| `chore:`   | Build, tooling, config                | `chore: add requirements.txt`                      |

### Pull Request Process

1. **Open a PR** against the `dev` branch.
2. **Title** your PR with a conventional commit prefix (e.g., `feat: ...`).
3. **Describe** what the PR does, why it's needed, and any trade-offs.
4. **Include screenshots** for any dashboard UI changes.
5. **Ensure the pipeline still works** — run the affected notebooks end-to-end.
6. **Request a review** from at least one maintainer.

---

## Coding Standards

### Python Style

- **Formatter:** Use [Black](https://black.readthedocs.io/) with default settings (line length 88).
- **Linter:** Use [Flake8](https://flake8.pycqa.org/) or [Ruff](https://docs.astral.sh/ruff/).
- **Type hints:** Use type annotations for function signatures (see `allocate_surplus` in `9-app.py` for an example).
- **Imports:** Group imports in the order: standard library → third-party → local.

### Notebooks

- Keep notebook cells **focused and well-commented** — each cell should do one thing.
- Use **Markdown cells** to explain the physics, math, and rationale behind each step.
- Clear all cell outputs before committing (`Cell → All Output → Clear`).
- Notebook filenames follow the `<step_number>-<descriptive_name>.ipynb` convention.

### Dashboard (Streamlit)

- All new dashboard components should follow the existing dark theme (background `#0E1117`, cards `#21262D`, accent `#58A6FF`).
- Use Plotly for interactive charts and Folium for maps.
- Wrap data loading in `@st.cache_data` decorators.
- Test the dashboard at different browser widths — use `layout="wide"` responsibly.

---

## Where to Contribute

Here are some areas where contributions are especially welcome:

| Area                        | Description                                                        |
|-----------------------------|--------------------------------------------------------------------|
| 📊 **Dashboard Enhancements** | New charts, improved layouts, additional KPIs, mobile responsiveness |
| 🔬 **Model Improvements**    | Better forecasting accuracy, additional weather features, hyperparameter tuning |
| 🧪 **Testing**               | Unit tests for the dispatch solver, surplus calculations, and settlement logic |
| 📖 **Documentation**         | Expand the `docs/` guides, add architecture diagrams, improve inline comments |
| 🌍 **Localization**          | Arabic language support for the dashboard UI                        |
| ⚡ **Performance**           | Optimize data loading, caching strategies, and large dataset handling |

---

## Reporting Issues

When opening an issue, please include:

1. **Title:** A clear, one-line summary.
2. **Description:** What you expected vs. what actually happened.
3. **Steps to Reproduce:** Numbered steps to reproduce the problem.
4. **Environment:** Python version, OS, browser (for dashboard issues).
5. **Screenshots / Logs:** If applicable.

Use the following labels when available:

| Label        | Meaning                    |
|--------------|----------------------------|
| `bug`        | Something is broken        |
| `enhancement`| Feature request            |
| `docs`       | Documentation improvement  |
| `dashboard`  | Related to `9-app.py`      |
| `pipeline`   | Related to notebooks 1–8   |

---

## Code of Conduct

We are committed to a welcoming and inclusive environment. All contributors are expected to:

- Be respectful and constructive in all interactions.
- Welcome newcomers and help them get started.
- Focus on the technical merits of contributions.
- Report unacceptable behavior to the maintainers.

---

**Thank you for helping build Egypt's smart grid future! ⚡🇪🇬**
