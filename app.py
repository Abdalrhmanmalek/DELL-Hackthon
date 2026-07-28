import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Shabaka Pulse VPP", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM THEMING (Dark / Teal) ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0E1117;
        color: #C9D1D9;
    }
    .css-1d391kg {
        background-color: #161B22;
    }
    h1, h2, h3 {
        color: #58A6FF !important;
    }
    .metric-card {
        background-color: #21262D;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        border-top: 4px solid #238636;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #39D353;
    }
    .metric-label {
        font-size: 1rem;
        color: #8B949E;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    surplus_df = pd.read_csv("data/surplus_forecast.csv", index_col="timestamp_utc", parse_dates=True)
    facilities_df = pd.read_csv("data/facilities_clustered.csv")
    
    try:
        with open("data/settlement_ledgers.json", "r") as f:
            ledgers = json.load(f)
    except FileNotFoundError:
        ledgers = []
        
    return surplus_df, facilities_df, ledgers

surplus_df, facilities_df, ledgers = load_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Shabaka Pulse")
st.sidebar.markdown("Virtual Power Plant Engine")
view_selection = st.sidebar.radio(
    "Navigation Mode",
    ["EETC Control Room", "Industrial Partner Portal"]
)

if view_selection == "EETC Control Room":
    st.title("🌐 EETC Control Room")
    st.markdown("Real-time view of national surplus, automated dispatch, and grid stability interlocks.")
    
    # 1. METRICS ROW
    sample_week = surplus_df.loc["2025-01-01":"2025-01-07"]
    peak_surplus = sample_week["surplus_mw"].max()
    
    total_mwh = sum(l["mwh_absorbed"] for l in ledgers)
    total_gas_savings = sum(l["gross_savings_egp"] for l in ledgers)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Grid Frequency</div><div class="metric-value">50.00 Hz</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Peak Surplus</div><div class="metric-value">{peak_surplus:,.0f} MW</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">MWh Absorbed (Week)</div><div class="metric-value">{total_mwh:,.0f}</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Gas Value Saved</div><div class="metric-value">EGP {total_gas_savings/1e6:,.1f}M</div></div>', unsafe_allow_html=True)
        
    st.markdown("---")
    
    # 2. CHARTS ROW
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.subheader("Renewable Generation vs. Transmission Headroom")
        
        fig_surplus = go.Figure()
        fig_surplus.add_trace(go.Scatter(x=sample_week.index, y=sample_week["total_pred_mw"], mode='lines', name='Total Generation', line=dict(color='#58A6FF', width=2)))
        fig_surplus.add_trace(go.Scatter(x=sample_week.index, y=sample_week["local_headroom_mw"], mode='lines', name='Headroom Limit', line=dict(color='#F85149', width=2, dash='dash')))
        
        # Fill surplus area
        surplus_mask = sample_week["total_pred_mw"] > sample_week["local_headroom_mw"]
        if surplus_mask.any():
            fig_surplus.add_trace(go.Scatter(
                x=sample_week[surplus_mask].index, 
                y=sample_week[surplus_mask]["total_pred_mw"],
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False
            ))
            fig_surplus.add_trace(go.Scatter(
                x=sample_week[surplus_mask].index,
                y=sample_week[surplus_mask]["local_headroom_mw"],
                fill='tonexty',
                mode='lines',
                fillcolor='rgba(57, 211, 83, 0.3)',
                line_color='rgba(0,0,0,0)',
                name='Curtailable Surplus'
            ))
            
        fig_surplus.update_layout(
            template="plotly_dark", 
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_surplus, use_container_width=True)

    with col_right:
        st.subheader("Fast Frequency Interlock (Digital Twin)")
        
        # Simulate frequency curve internally for the dashboard
        import numpy as np
        time_steps = np.arange(0, 30, 0.05)
        freq_hist = []
        f_curr = 50.0
        p_gen, p_load = 35000.0, 35000.0
        trigger = False
        
        for t in time_steps:
            if t >= 10.0 and p_gen == 35000.0:
                p_gen -= 1800.0
            if f_curr < 49.8 and not trigger:
                p_load -= 300.0
                trigger = True
            
            df_step = 50.0 * ((p_gen - p_load) / 59700.0) / (2.0 * 4.5) * 0.05
            f_curr += df_step
            freq_hist.append(f_curr)
            
        fig_freq = go.Figure()
        fig_freq.add_trace(go.Scatter(x=time_steps, y=freq_hist, mode='lines', name='Grid Frequency', line=dict(color='#39D353', width=2)))
        fig_freq.add_trace(go.Scatter(x=[0, 30], y=[50.0, 50.0], mode='lines', name='Nominal', line=dict(color='#8B949E', dash='dash')))
        fig_freq.add_trace(go.Scatter(x=[0, 30], y=[49.8, 49.8], mode='lines', name='Interlock Threshold', line=dict(color='#F85149', dash='dash')))
        
        fig_freq.update_layout(
            template="plotly_dark", 
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis_title="Frequency (Hz)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_freq, use_container_width=True)

    # 3. MAP ROW
    st.markdown("---")
    st.subheader("Active VPP Industrial Facilities")
    
    # Map colors based on operational tier
    tier_colors = {1: "blue", 2: "orange", 3: "red"}
    map_center = [facilities_df["lat"].mean(), facilities_df["lon"].mean()]
    
    m = folium.Map(location=map_center, zoom_start=6, tiles="CartoDB dark_matter")
    
    for _, row in facilities_df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=row["max_flex_mw"] / 5,  # Size by capacity
            color=tier_colors.get(row["tier"], "white"),
            fill=True,
            fill_opacity=0.7,
            popup=f"<b>{row['facility_name']}</b><br>Tier: {row['tier']}<br>Flex MW: {row['max_flex_mw']}<br>Storage: {row['storage_type']}"
        ).add_to(m)
        
    st_folium(m, width=1200, height=400)


elif view_selection == "Industrial Partner Portal":
    st.title("🏭 Industrial Partner Portal")
    st.markdown("Manage facility commitments and view real-time billing credits.")
    
    facility_names = facilities_df["facility_name"].tolist()
    selected_name = st.selectbox("Select Facility", facility_names)
    
    fac_data = facilities_df[facilities_df["facility_name"] == selected_name].iloc[0]
    
    st.markdown(f"**Industry Type:** {fac_data['industry_type']} | **Operational Tier:** {fac_data['tier']} | **Storage Type:** {fac_data['storage_type']}")
    
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
        
        st.info("By offering capacity, your facility becomes eligible to absorb renewable surplus at heavily discounted energy rates.")
        
    with col_table:
        st.subheader("Settlement Ledger (Past 7 Days)")
        
        # Find this facility in the ledger
        fac_ledger = next((item for item in ledgers if item["facility_id"] == fac_data["facility_id"]), None)
        
        if fac_ledger:
            df_ledger = pd.DataFrame([
                {"Metric": "Billing Period", "Value": fac_ledger["period"]},
                {"Metric": "Total Renewable Energy Absorbed", "Value": f"{fac_ledger['mwh_absorbed']} MWh"},
                {"Metric": "Gross System Savings (Gas + PPA)", "Value": f"{fac_ledger['gross_savings_egp']:,.2f} EGP"},
                {"Metric": "Your Factory Billing Credit (50%)", "Value": f"{fac_ledger['factory_discount_credit_egp']:,.2f} EGP"},
            ])
            st.dataframe(df_ledger, use_container_width=True, hide_index=True)
            st.success(f"🎉 Your active participation preserved grid stability and earned a discount of **{fac_ledger['factory_discount_credit_egp']:,.2f} EGP**.")
        else:
            st.warning("No settlement records found for this facility in the current period. Ensure the settlement engine has processed recent dispatch logs.")
