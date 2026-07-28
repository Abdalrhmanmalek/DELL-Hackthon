import json
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import folium
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="Shabaka Pulse VPP", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM THEMING ---
st.markdown(
    """
    <style>
    .stApp { background-color: #0E1117; color: #C9D1D9; }
    .css-1d391kg { background-color: #161B22; }
    h1, h2, h3 { color: #58A6FF !important; }
    .metric-card {
        background-color: #21262D;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        border-top: 4px solid #238636;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #39D353; }
    .metric-label { font-size: 1rem; color: #8B949E; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    surplus_df = pd.read_csv(
        "data/surplus_forecast.csv", index_col="timestamp_utc", parse_dates=True
    )
    facilities_df = pd.read_csv("data/facilities_clustered.csv")
    try:
        with open("data/settlement_ledgers.json", "r") as f:
            ledgers = json.load(f)
    except FileNotFoundError:
        ledgers = []
    return surplus_df, facilities_df, ledgers

surplus_df, facilities_df, ledgers = load_data()

# --- DISPATCH SOLVER (inline for dashboard use) ---
def allocate_surplus(
    surplus_mw: float,
    clustered_facilities: pd.DataFrame,
) -> Tuple[Dict[str, float], float]:
    remaining = float(surplus_mw)
    allocs: Dict[str, float] = {str(f): 0.0 for f in clustered_facilities["facility_id"]}
    for tier in [1, 2, 3]:
        if remaining <= 0.0:
            break
        tier_facs = clustered_facilities[clustered_facilities["tier"] == tier].sort_values("facility_id")
        for _, row in tier_facs.iterrows():
            if remaining <= 0.0:
                break
            fid = str(row["facility_id"])
            allocated = min(float(row["max_flex_mw"]), remaining)
            allocs[fid] = round(allocated, 2)
            remaining -= allocated
    return allocs, round(remaining, 2)

# --- SIDEBAR ---
st.sidebar.title("Shabaka Pulse")
st.sidebar.markdown("Virtual Power Plant Engine")
view_selection = st.sidebar.radio(
    "Navigation Mode",
    ["EETC Control Room", "Dispatch Simulator", "Industrial Partner Portal"]
)

# =====================================================================
# VIEW 1: EETC CONTROL ROOM
# =====================================================================
if view_selection == "EETC Control Room":
    st.title("EETC Control Room")
    st.markdown("Full-year renewable generation overview, surplus pattern, and grid stability.")

    # --- METRICS ---
    total_surplus_mwh = surplus_df["surplus_mw"].sum()
    peak_surplus = surplus_df["surplus_mw"].max()
    nonzero_hours = int((surplus_df["surplus_mw"] > 0).sum())
    total_mwh = sum(l["mwh_absorbed"] for l in ledgers)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Grid Frequency</div><div class="metric-value">50.00 Hz</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Annual Peak Surplus</div><div class="metric-value">{peak_surplus:,.0f} MW</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Annual Surplus Hours</div><div class="metric-value">{nonzero_hours:,} hrs</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Surplus Energy</div><div class="metric-value">{total_surplus_mwh:,.0f} MWh</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # --- FULL YEAR SURPLUS CHART ---
    st.subheader("Full-Year Renewable Generation & Curtailable Surplus")
    fig_year = go.Figure()
    fig_year.add_trace(go.Scatter(
        x=surplus_df.index, y=surplus_df["total_pred_mw"],
        mode="lines", name="Total Generation (Solar + Wind)",
        line=dict(color="#58A6FF", width=1), opacity=0.8
    ))
    fig_year.add_trace(go.Scatter(
        x=surplus_df.index, y=surplus_df["local_headroom_mw"],
        mode="lines", name="Transmission Headroom Limit",
        line=dict(color="#F85149", width=1.5, dash="dash")
    ))
    fig_year.add_trace(go.Scatter(
        x=surplus_df.index, y=surplus_df["surplus_mw"],
        mode="lines", name="Curtailable Surplus (Redirectable MW)",
        line=dict(color="#39D353", width=1.5),
        fill="tozeroy", fillcolor="rgba(57,211,83,0.15)"
    ))
    fig_year.update_layout(
        template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=380, margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="Power (MW)", xaxis_title="Date"
    )
    st.plotly_chart(fig_year, use_container_width=True)

    st.markdown("---")

    # --- MONTHLY HEATMAP ---
    col_heat, col_freq = st.columns([1, 1])

    with col_heat:
        st.subheader("Monthly & Hourly Surplus Distribution")
        pivot = surplus_df.pivot_table(
            values="surplus_mw", index=surplus_df.index.month,
            columns=surplus_df.index.hour, aggfunc="sum"
        ).fillna(0)
        month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[f"{h:02d}:00" for h in pivot.columns],
            y=[month_labels[m-1] for m in pivot.index],
            colorscale="YlOrRd",
            colorbar=dict(title="MWh")
        ))
        fig_heat.update_layout(
            template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=340, margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Hour of Day (UTC)", yaxis_title="Month"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_freq:
        st.subheader("Frequency Interlock — Digital Twin")
        time_steps = np.arange(0, 30, 0.05)
        freq_hist, f_curr, p_gen, p_load, trigger = [], 50.0, 35000.0, 35000.0, False
        for t in time_steps:
            if t >= 10.0 and p_gen == 35000.0:
                p_gen -= 1800.0
            if f_curr < 49.8 and not trigger:
                p_load -= 300.0
                trigger = True
            f_curr += 50.0 * ((p_gen - p_load) / 59700.0) / (2.0 * 4.5) * 0.05
            freq_hist.append(f_curr)

        fig_freq = go.Figure()
        fig_freq.add_trace(go.Scatter(x=time_steps, y=freq_hist, name="Grid Frequency", line=dict(color="#39D353", width=2)))
        fig_freq.add_trace(go.Scatter(x=[0, 30], y=[50.0, 50.0], name="Nominal (50 Hz)", line=dict(color="#8B949E", dash="dash")))
        fig_freq.add_trace(go.Scatter(x=[0, 30], y=[49.8, 49.8], name="Interlock Threshold", line=dict(color="#F85149", dash="dash")))
        fig_freq.update_layout(
            template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=340, margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="Frequency (Hz)", xaxis_title="Time (s)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_freq, use_container_width=True)

    # --- MAP ---
    st.markdown("---")
    st.subheader("Active VPP Industrial Facilities")
    tier_colors = {1: "blue", 2: "orange", 3: "red"}
    m = folium.Map(location=[facilities_df["lat"].mean(), facilities_df["lon"].mean()], zoom_start=6, tiles="CartoDB dark_matter")
    for _, row in facilities_df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=row["max_flex_mw"] / 5,
            color=tier_colors.get(row["tier"], "white"),
            fill=True, fill_opacity=0.8,
            popup=f"<b>{row['facility_name']}</b><br>Tier: {row['tier']}<br>Flex: {row['max_flex_mw']} MW<br>Storage: {row['storage_type']}"
        ).add_to(m)
    st_folium(m, use_container_width=True, height=420)


# =====================================================================
# VIEW 2: DISPATCH SIMULATOR
# =====================================================================
elif view_selection == "Dispatch Simulator":
    st.title("Dispatch Simulator")
    st.markdown(
        "Select any hour from the full year or set a custom surplus value. "
        "The priority dispatch solver will show exactly how that surplus is allocated "
        "across the 30 industrial facilities, tier by tier."
    )

    system_capacity_mw = float(facilities_df["max_flex_mw"].sum())

    st.markdown("---")

    # --- CONTROL PANEL ---
    sim_col, gap, result_col = st.columns([2, 0.15, 3])

    with sim_col:
        st.subheader("Simulation Controls")

        surplus_mode = st.radio(
            "Surplus Source",
            ["Pick from full-year data", "Enter custom surplus MW"],
            horizontal=True
        )

        if surplus_mode == "Pick from full-year data":
            # Date + hour selectors
            min_date = surplus_df.index.date.min()
            max_date = surplus_df.index.date.max()
            selected_date = st.date_input("Select Date", value=pd.Timestamp("2025-06-15").date(), min_value=min_date, max_value=max_date)
            selected_hour = st.slider("Hour of Day (UTC)", 0, 23, 12)

            ts_key = pd.Timestamp(f"{selected_date} {selected_hour:02d}:00:00")
            if ts_key in surplus_df.index:
                row = surplus_df.loc[ts_key]
                sim_surplus_mw = float(row["surplus_mw"])
                solar_at_ts = float(row["solar_pred_mw"])
                wind_at_ts = float(row["wind_pred_mw"])
                headroom_at_ts = float(row["local_headroom_mw"])
            else:
                st.warning("Timestamp not found in dataset.")
                st.stop()

            st.markdown("##### Conditions at selected timestamp")
            ts_summary = pd.DataFrame([
                {"Parameter": "Solar Generation", "Value": f"{solar_at_ts:,.1f} MW"},
                {"Parameter": "Wind Generation", "Value": f"{wind_at_ts:,.1f} MW"},
                {"Parameter": "Combined Total", "Value": f"{solar_at_ts + wind_at_ts:,.1f} MW"},
                {"Parameter": "Transmission Headroom", "Value": f"{headroom_at_ts:,.1f} MW"},
                {"Parameter": "Curtailable Surplus", "Value": f"{sim_surplus_mw:,.1f} MW"},
            ])
            st.dataframe(ts_summary, hide_index=True, use_container_width=True)

        else:
            sim_surplus_mw = st.slider(
                "Custom Surplus MW to Dispatch",
                min_value=0.0,
                max_value=float(system_capacity_mw),
                value=300.0,
                step=10.0,
            )
            st.caption(f"Total system flexible capacity: {system_capacity_mw:,.0f} MW")

        st.markdown("---")
        st.markdown(f"**Surplus to dispatch:** `{sim_surplus_mw:,.1f} MW`")
        st.markdown(f"**System total flex capacity:** `{system_capacity_mw:,.0f} MW`")

    # --- RUN DISPATCH ---
    allocations, remaining_mw = allocate_surplus(sim_surplus_mw, facilities_df)

    alloc_df = pd.DataFrame([
        {
            "facility_id": fid,
            "facility_name": facilities_df.loc[facilities_df["facility_id"] == fid, "facility_name"].values[0],
            "tier": int(facilities_df.loc[facilities_df["facility_id"] == fid, "tier"].values[0]),
            "max_flex_mw": float(facilities_df.loc[facilities_df["facility_id"] == fid, "max_flex_mw"].values[0]),
            "allocated_mw": mw,
        }
        for fid, mw in allocations.items()
    ]).sort_values(["tier", "facility_id"])

    total_allocated = alloc_df["allocated_mw"].sum()
    n_active = int((alloc_df["allocated_mw"] > 0).sum())

    with result_col:
        st.subheader("Dispatch Result")

        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">MW Redirected</div><div class="metric-value">{total_allocated:,.0f} MW</div></div>', unsafe_allow_html=True)
        with r2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Facilities Active</div><div class="metric-value">{n_active} / 30</div></div>', unsafe_allow_html=True)
        with r3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Unallocated MW</div><div class="metric-value">{remaining_mw:,.0f} MW</div></div>', unsafe_allow_html=True)

        st.markdown("")

        # Bar chart colored by tier
        tier_palette = {1: "#58A6FF", 2: "#F0883E", 3: "#F85149"}
        fig_alloc = go.Figure()

        for tier_num in [1, 2, 3]:
            tier_data = alloc_df[alloc_df["tier"] == tier_num]
            fig_alloc.add_trace(go.Bar(
                x=tier_data["facility_name"],
                y=tier_data["allocated_mw"],
                name=f"Tier {tier_num}",
                marker_color=tier_palette[tier_num],
                customdata=tier_data[["max_flex_mw"]].values,
                hovertemplate="<b>%{x}</b><br>Allocated: %{y:.1f} MW<br>Max Flex: %{customdata[0]:.1f} MW<extra></extra>"
            ))

        fig_alloc.update_layout(
            template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            barmode="stack", height=320, margin=dict(l=0, r=0, t=10, b=80),
            xaxis_tickangle=-45, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis_title="Allocated MW"
        )
        st.plotly_chart(fig_alloc, use_container_width=True)

    st.markdown("---")

    # --- TIER BREAKDOWN TABLE ---
    st.subheader("Allocation Breakdown by Tier")
    tier_summary = alloc_df.groupby("tier").agg(
        facilities_active=("allocated_mw", lambda x: (x > 0).sum()),
        total_allocated_mw=("allocated_mw", "sum"),
        max_possible_mw=("max_flex_mw", "sum"),
    ).reset_index()
    tier_summary["utilisation_pct"] = (tier_summary["total_allocated_mw"] / tier_summary["max_possible_mw"] * 100).round(1)
    tier_summary.columns = ["Tier", "Facilities Active", "Allocated MW", "Max Capacity MW", "Utilisation %"]
    st.dataframe(tier_summary, hide_index=True, use_container_width=True)

    st.markdown("")
    st.subheader("Per-Facility Allocation")
    display_df = alloc_df[alloc_df["allocated_mw"] > 0].copy()
    display_df["utilisation_pct"] = (display_df["allocated_mw"] / display_df["max_flex_mw"] * 100).round(1)
    display_df = display_df[["facility_name", "tier", "max_flex_mw", "allocated_mw", "utilisation_pct"]]
    display_df.columns = ["Facility", "Tier", "Max Flex MW", "Allocated MW", "Utilisation %"]
    st.dataframe(display_df, hide_index=True, use_container_width=True)

    # --- FULL YEAR CUMULATIVE CHART ---
    st.markdown("---")
    st.subheader("Full-Year Surplus: Redirectable MW Over Time")

    fig_full = go.Figure()
    fig_full.add_trace(go.Scatter(
        x=surplus_df.index, y=surplus_df["surplus_mw"],
        mode="lines", name="Curtailable Surplus (MW)",
        line=dict(color="#39D353", width=1.2),
        fill="tozeroy", fillcolor="rgba(57,211,83,0.12)"
    ))
    # Mark the selected timestamp if in data-pick mode
    if surplus_mode == "Pick from full-year data" and ts_key in surplus_df.index:
        fig_full.add_trace(go.Scatter(
            x=[ts_key], y=[sim_surplus_mw],
            mode="markers", name="Selected Hour",
            marker=dict(color="#F0883E", size=12, symbol="star")
        ))
    fig_full.update_layout(
        template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=300, margin=dict(l=0, r=0, t=10, b=0),
        yaxis_title="Surplus MW", xaxis_title="Date"
    )
    st.plotly_chart(fig_full, use_container_width=True)


# =====================================================================
# VIEW 3: INDUSTRIAL PARTNER PORTAL
# =====================================================================
elif view_selection == "Industrial Partner Portal":
    st.title("Industrial Partner Portal")
    st.markdown("Manage your facility commitment and view your settlement billing credit.")

    facility_names = facilities_df["facility_name"].tolist()
    selected_name = st.selectbox("Select Facility", facility_names)
    fac_data = facilities_df[facilities_df["facility_name"] == selected_name].iloc[0]

    tier_label = {1: "Tier 1 — Hydraulic (Fastest Response)", 2: "Tier 2 — Thermal Storage", 3: "Tier 3 — Batch Processing"}
    st.markdown(
        f"**Industry:** {fac_data['industry_type']} | "
        f"**{tier_label.get(fac_data['tier'], 'Unknown Tier')}** | "
        f"**Storage:** {fac_data['storage_type']}"
    )

    col_ctrl, col_table = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Commitment Control")
        offered_mw = st.slider(
            "Flexible Capacity Offered (MW)",
            min_value=0.0,
            max_value=float(fac_data["max_flex_mw"]),
            value=float(fac_data["max_flex_mw"]),
            step=1.0
        )
        pct_offered = offered_mw / fac_data["max_flex_mw"] * 100 if fac_data["max_flex_mw"] > 0 else 0
        st.markdown(f"Offering `{offered_mw:.0f} MW` — **{pct_offered:.0f}%** of maximum capacity.")
        st.info("By offering capacity, your facility is eligible to absorb renewable surplus at a heavily discounted energy rate.")

    with col_table:
        st.subheader("Settlement Ledger (Sample Winter Week)")
        fac_ledger = next((item for item in ledgers if item["facility_id"] == fac_data["facility_id"]), None)

        if fac_ledger:
            # Scale ledger values by the slider offering fraction
            scale = offered_mw / fac_data["max_flex_mw"] if fac_data["max_flex_mw"] > 0 else 0
            scaled_mwh = fac_ledger["mwh_absorbed"] * scale
            scaled_gross = fac_ledger["gross_savings_egp"] * scale
            scaled_credit = fac_ledger["factory_discount_credit_egp"] * scale

            df_ledger = pd.DataFrame([
                {"Metric": "Billing Period", "Value": fac_ledger["period"]},
                {"Metric": "Capacity Committed", "Value": f"{offered_mw:.0f} MW"},
                {"Metric": "Renewable Energy Absorbed", "Value": f"{scaled_mwh:,.2f} MWh"},
                {"Metric": "Gross System Savings (Gas + PPA)", "Value": f"{scaled_gross:,.2f} EGP"},
                {"Metric": "Your Factory Billing Credit (50%)", "Value": f"{scaled_credit:,.2f} EGP"},
            ])
            st.dataframe(df_ledger, use_container_width=True, hide_index=True)
            st.success(f"Estimated billing credit: **{scaled_credit:,.2f} EGP** for this period.")
        else:
            st.warning("No settlement record found. Run settlement_engine.ipynb first.")
