#!/usr/bin/env python3
"""
Simple Streamlit Dashboard for EV Battery Demo
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page config
st.set_page_config(page_title="EV Battery Dashboard", layout="wide")

# Generate mock data
@st.cache_data
def generate_demo_data():
    # Fleet data
    vehicles = []
    for i in range(20):
        soh = 85 + np.random.random() * 10
        rul = int(200 + np.random.random() * 800)
        risk = "High" if soh < 87 else "Medium" if soh < 90 else "Low"
        
        vehicles.append({
            'Vehicle ID': f'EV_{i:04d}',
            'SoH (%)': round(soh, 1),
            'RUL (days)': rul,
            'Risk': risk,
            'Location': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'])
        })
    
    # Timeline data for selected vehicle
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    timeline = []
    soh_start = 95
    for i, date in enumerate(dates):
        soh = soh_start - (i * 0.02) + np.random.normal(0, 0.5)
        timeline.append({
            'Date': date,
            'SoH (%)': max(80, soh),
            'Temperature (°C)': 25 + np.random.normal(0, 8),
            'Voltage (V)': 380 + np.random.normal(0, 10)
        })
    
    return pd.DataFrame(vehicles), pd.DataFrame(timeline)

# Load data
fleet_df, timeline_df = generate_demo_data()

# Title
st.title("🔋 EV Battery Health Dashboard")

# Sidebar
st.sidebar.header("Controls")
selected_vehicle = st.sidebar.selectbox("Select Vehicle", fleet_df['Vehicle ID'])

# Main dashboard
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Vehicles", len(fleet_df))

with col2:
    avg_soh = fleet_df['SoH (%)'].mean()
    st.metric("Average SoH", f"{avg_soh:.1f}%")

with col3:
    high_risk = len(fleet_df[fleet_df['Risk'] == 'High'])
    st.metric("High Risk Vehicles", high_risk)

with col4:
    avg_rul = fleet_df['RUL (days)'].mean()
    st.metric("Average RUL", f"{avg_rul:.0f} days")

# Fleet overview
st.subheader("Fleet Overview")
col1, col2 = st.columns([2, 1])

with col1:
    # Scatter plot
    fig = px.scatter(fleet_df, x='SoH (%)', y='RUL (days)', 
                    color='Risk', hover_data=['Vehicle ID', 'Location'],
                    title="Fleet Health Distribution")
    fig.add_hline(y=365, line_dash="dash", line_color="orange", 
                  annotation_text="1 Year Threshold")
    fig.add_vline(x=80, line_dash="dash", line_color="red", 
                  annotation_text="EOL Threshold")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Risk distribution
    risk_counts = fleet_df['Risk'].value_counts()
    fig_pie = px.pie(values=risk_counts.values, names=risk_counts.index, 
                     title="Risk Distribution")
    st.plotly_chart(fig_pie, use_container_width=True)

# Vehicle details
st.subheader(f"Vehicle Details: {selected_vehicle}")
vehicle_data = fleet_df[fleet_df['Vehicle ID'] == selected_vehicle].iloc[0]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Current SoH", f"{vehicle_data['SoH (%)']}%")
with col2:
    st.metric("Predicted RUL", f"{vehicle_data['RUL (days)']} days")
with col3:
    st.metric("Risk Level", vehicle_data['Risk'])

# Timeline chart
st.subheader("Battery Health Timeline")
fig_timeline = go.Figure()
fig_timeline.add_trace(go.Scatter(
    x=timeline_df['Date'], 
    y=timeline_df['SoH (%)'],
    mode='lines',
    name='SoH (%)',
    line=dict(color='blue', width=2)
))
fig_timeline.add_hline(y=80, line_dash="dash", line_color="red", 
                      annotation_text="EOL Threshold (80%)")
fig_timeline.update_layout(title="State of Health Over Time", 
                          xaxis_title="Date", yaxis_title="SoH (%)")
st.plotly_chart(fig_timeline, use_container_width=True)

# What-if analysis
st.subheader("What-If Analysis")
col1, col2 = st.columns([1, 2])

with col1:
    st.write("**Scenario Adjustments:**")
    fast_charge_reduction = st.slider("Reduce Fast Charging", 0, 50, 20, help="% reduction")
    temp_reduction = st.slider("Reduce Max Temperature", 0, 20, 10, help="°C reduction")
    
    # Calculate impact
    rul_improvement = fast_charge_reduction * 0.5 + temp_reduction * 1.2
    new_rul = vehicle_data['RUL (days)'] + rul_improvement
    
    st.metric("RUL Improvement", f"+{rul_improvement:.0f} days")
    st.metric("New Predicted RUL", f"{new_rul:.0f} days")

with col2:
    # Impact chart
    scenarios = ['Current', 'Optimized']
    rul_values = [vehicle_data['RUL (days)'], new_rul]
    
    fig_impact = px.bar(x=scenarios, y=rul_values, 
                       title="RUL Comparison: Current vs Optimized",
                       color=scenarios)
    fig_impact.update_layout(showlegend=False, yaxis_title="RUL (days)")
    st.plotly_chart(fig_impact, use_container_width=True)

# Fleet table
st.subheader("Fleet Status Table")
st.dataframe(fleet_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**EV Battery Prediction System** - Real-time monitoring and predictive maintenance")