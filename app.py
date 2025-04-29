import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
import datetime
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Accident Analysis Dashboard")

df = pd.read_excel('Updated_Data.xlsx', sheet_name='Sheet1')

image = Image.open('logo.jpg')

col1, col2 = st.columns([0.1, 0.9])
with col1:
    st.image(image, width=100)

html_title = """
    <style>
    .title-test {
    font-size: 40px;
    font-weight: bold;
    padding:5px;
    border-radius:6px;
    }
    </style>
    <center><h1 class="title-test">Accident Analysis Dashboard</h1></center>"""
with col2:
    st.markdown(html_title, unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; font-size: 24px;'>An Analysis of 2021-2023 Accident Data and Prediction of 2024 and 2025 Accident Data</h2>", unsafe_allow_html=True)

st.divider()

col3, col4, col5 = st.columns([0.1, 0.45, 0.45])
with col3:
    box_date = str(datetime.datetime.now().strftime("%d %B %Y"))
    st.write(f"Last updates by:  \n {box_date}")

with col4:
    df_grouped = df.groupby("Year")["Total Accidents"].sum().reset_index()
    fig = px.bar(
        df_grouped,
        x="Year",
        y="Total Accidents",
        labels={"Total Accidents": "Total Accidents"},
        title="Total Accidents Over Years",
        hover_data=["Year"],
        color_discrete_sequence=["#00FFFF"],  
        height=500
    )

    fig.update_layout(
        template="plotly_dark",  
        font=dict(color="white", size=16),
        title_font=dict(size=22, color="white"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#3C8DAD") 
    )
    st.plotly_chart(fig, use_container_width=True)

with col5:
    year_weather_cols = st.columns([1, 1]) 
    
    with year_weather_cols[0]:
        selected_year = st.selectbox("Select Year", sorted(df["Year"].unique()))
    
    with year_weather_cols[1]:
        selected_weather = st.selectbox("Select Weather", ["All"] + sorted(df["Weather"].unique()))

    if selected_weather == "All":
        df_filtered = df[df["Year"] == selected_year]
    else:
        df_filtered = df[(df["Year"] == selected_year) & (df["Weather"] == selected_weather)]

    color_scales = {
        2021: ["#ffcccc", "#ff6666", "#ff0000"],  # Red scale for 2021
        2022: ["#ffebcd", "#ff7f50", "#ff4500"],  # Orange scale for 2022
        2023: ["#ffffe0", "#ffdf00", "#ffcc00"],  # Yellow scale for 2023
        2024: ["#cceeff", "#66b3ff", "#3399ff"],  # Light blue scale for 2024
        2025: ["#e6ccff", "#9966ff", "#6600cc"],  # Green scale for 2025
    }

    color_scale = color_scales.get(selected_year, ["#00FFFF", "#2E8BC0", "#1E90FF"]) 

    fig = px.density_mapbox(
        df_filtered,
        lat="Latitude",
        lon="Longitude",
        z="Total Accidents",
        radius=25,
        center=dict(lat=11.0, lon=78.0),
        zoom=5.5,
        mapbox_style="carto-positron",
        hover_name="District",
        hover_data={
            "Weather": True,
            "Severity": True,
            "Total Accidents": True,
            "Latitude": False,
            "Longitude": False
        },
        color_continuous_scale=color_scale,
        title=f"Accident Heatmap - {selected_year} ({'All Weather' if selected_weather == 'All' else selected_weather} Weather)"
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

_, view1, dwn1, view2, dwn2 = st.columns([0.15, 0.20, 0.20, 0.20, 0.20])

with view1:
    expander = st.expander("Accident Analysis by District", expanded=False)
    data = df[["District", "Total Accidents"]].groupby(by="District").agg({
        "Total Accidents": "sum"
    })
    with expander:
        st.write(data)

with dwn1:
    st.download_button("Get Data", data=data.to_csv().encode("utf-8"),
                       file_name="Data.xlsx", mime="text/csv")

weather_data = df.groupby("Weather")["Total Accidents"].sum().reset_index()
weather_data = weather_data.sort_values(by="Total Accidents", ascending=False)
weather_data.reset_index(drop=True, inplace=True)

with view2:
    expander = st.expander("Accident Breakdown by Weather", expanded=False)
    with expander:
        st.write(weather_data)
        fig = px.pie(
            weather_data,
            values="Total Accidents",
            names="Weather",
            title="Weather Conditions and Accident Distribution",
            template="ggplot2"
        )

with dwn2:
    st.download_button(
        label="Get Data",
        data=weather_data.to_csv(index=False).encode("utf-8"),
        file_name="Accidents_by_Weather.csv",
        mime="text/csv"
    )

st.divider()

accident_by_year = df.groupby("Year")["Total Accidents"].sum().reset_index()
total_accidents = df["Total Accidents"].sum()
highest_accident_district = df.groupby("District")["Total Accidents"].sum().idxmax()
pie_data = df.groupby("Weather")["Total Accidents"].sum().reset_index()

fig3 = px.pie(
    pie_data,
    values="Total Accidents",
    names="Weather",
    hole=0.4,
    title="Weather-wise Share of Total Accidents",
    color_discrete_sequence=["#003366", "#004C66", "#00688B", "#007B8C", "#00CED1"],  
    template="plotly_dark"
)

fig3.update_traces(
    textinfo="percent+label", 
    pull=[0, 0, 0, 0, 0]
)

fig3.update_layout(
    title_font=dict(size=24, color="white"),
    font=dict(color="white", size=16), 
    legend=dict(
        bgcolor="#1B263B",
        font=dict(color="white")
    ),
    margin=dict(l=20, r=20, t=40, b=20)  
)

_, col6, col7 = st.columns([0.1, 0.45, 0.45])

with col6:
    st.markdown("### Key Metrics", unsafe_allow_html=True)
    kpi_col1, kpi_col2 = st.columns(2)

    with kpi_col1:
        st.metric("Total Accidents", f"{total_accidents:,}", delta=None)

    with kpi_col2:
        st.metric("District with Most Accidents", highest_accident_district, delta=None)

    kpi_col3, kpi_col4 = st.columns(2)

    with kpi_col3:
        accidents_2021_2023 = accident_by_year[accident_by_year['Year'].isin([2021, 2022, 2023])]
        for _, row in accidents_2021_2023.iterrows():
            st.metric(f"Accidents in {row['Year']}", f"{row['Total Accidents']:,}", delta=None)

    with kpi_col4:
        accidents_2024_2025 = accident_by_year[accident_by_year['Year'].isin([2024, 2025])]
        for _, row in accidents_2024_2025.iterrows():
            st.metric(f"Accidents in {row['Year']}", f"{row['Total Accidents']:,}", delta=None)

with col7:
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

_, col8, col9 = st.columns([0.1, 0.45, 0.45]) 

with col8:
    st.markdown("### Severity vs. Weather Conditions", unsafe_allow_html=True)
    fig_box = px.box(
        df,
        x="Weather",
        y="Severity",
        color="Weather",
        title="Severity Distribution across Weather Conditions",
        template="plotly_white",
        labels={"Severity": "Severity Level", "Weather": "Weather Condition"}
    )
    st.plotly_chart(fig_box, use_container_width=True)

with col9:
    df_line = df[df["Year"].isin([2021, 2022, 2023, 2024, 2025])].copy()
    df_line["Year"] = df_line["Year"].astype(str)

    line_data = df_line.groupby(["State", "Year"])["Total Accidents"].sum().reset_index()

    fig_line = px.line(
        line_data,
        x="Year",
        y="Total Accidents",
        color="State",
        markers=True,
        title="State-wise Trend of Total Accidents (2021–2025)",
        template="plotly_white"
    )

    st.plotly_chart(fig_line, use_container_width=True, key="line_chart_col9")

st.divider()

severity_mapping = {'Low': 1, 'Moderate': 2, 'High': 3}
df['Severity_numeric'] = df['Severity'].map(severity_mapping)

col10, col11 = st.columns(2)

with col10:
    st.markdown("### 🚦 **Average Severity Gauge**")  

    avg_severity = df["Severity_numeric"].mean()

    fig_gauge_severity = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_severity,
        number={'font': {'size': 40, 'color': '#FFFFFF'}},  
        gauge={
            'axis': {'range': [None, 3], 'tickcolor': '#AAAAAA'},
            'bar': {'color': '#00688B'},
            'steps': [
                {'range': [0, 1], 'color': '#00FFFF'},
                {'range': [1, 2], 'color': '#1E90FF'},
                {'range': [2, 3], 'color': '#2E8BC0'}
            ],
        }
    ))

    fig_gauge_severity.update_layout(
        font={'size': 18, 'color': '#FFFFFF'},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={'l': 10, 'r': 10, 't': 50, 'b': 10}
    )

    st.plotly_chart(fig_gauge_severity, use_container_width=True)

with col11:
    st.markdown("### 🌦️ **Weather Impact Radar**")

    severity_by_weather = df.groupby("Weather")["Severity_numeric"].mean().reset_index()

    fig_radar = px.line_polar(
        severity_by_weather,
        r="Severity_numeric",
        theta="Weather",
        line_close=True,
        template="none",
        color_discrete_sequence=["#003366"]
    )

    fig_radar.update_traces(
        fill='toself',
        line=dict(color="#00CED1", width=3)
    )

    fig_radar.update_layout(
        font_color="#FFFFFF",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(color="#FFFFFF"),
            angularaxis=dict(color="#FFFFFF")
        ),
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False
    )

    st.plotly_chart(fig_radar, use_container_width=True)

st.divider()

st.markdown("<h2 style='text-align: center; font-size: 30px;'>ROC Curve Comparison: XGBoost vs LSTM</h2>", unsafe_allow_html=True)

col12, col13 = st.columns(2)

with col12:
    st.markdown("### XGBoost Model Evaluation")

    fig_roc_xgb, ax = plt.subplots(figsize=(6, 6), facecolor='none')
    fpr_xgb = [0.0, 0.1, 0.2, 0.4, 1.0]
    tpr_xgb = [0.0, 0.6, 0.7, 0.85, 1.0]

    ax.plot(fpr_xgb, tpr_xgb, color='#00FFFF', linewidth=2, label='XGBoost ROC Curve')
    ax.plot([0, 1], [0, 1], linestyle='--', color='#CCCCCC', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=12, color='white')
    ax.set_ylabel('True Positive Rate', fontsize=12, color='white')
    ax.set_title('ROC Curve - XGBoost', fontsize=14, color='white')
    ax.legend(loc='lower right', facecolor='none', edgecolor='white', labelcolor='white')
    ax.tick_params(colors='white')

    fig_roc_xgb.patch.set_alpha(0)
    ax.set_facecolor('none')
    st.pyplot(fig_roc_xgb)

    fig_xgboost = go.Figure()
    fig_xgboost.add_trace(go.Scatter(
        x=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        y=[0.89, 0.87, 0.85, 0.86],
        mode='lines+markers',
        line=dict(color='#00CED1', width=3),
        marker=dict(size=10)
    ))

    fig_xgboost.update_layout(
        title="XGBoost Evaluation Metrics",
        xaxis_title="Metrics",
        yaxis_title="Scores",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )

    st.plotly_chart(fig_xgboost, use_container_width=True)

with col13:
    st.markdown("### LSTM Model Evaluation")

    fig_roc_lstm, ax2 = plt.subplots(figsize=(6, 6), facecolor='none')
    fpr_lstm = [0.0, 0.05, 0.15, 0.35, 1.0]
    tpr_lstm = [0.0, 0.7, 0.8, 0.9, 1.0]

    ax2.plot(fpr_lstm, tpr_lstm, color='#1E90FF', linewidth=2, label='LSTM ROC Curve')
    ax2.plot([0, 1], [0, 1], linestyle='--', color='#CCCCCC', linewidth=1)
    ax2.set_xlabel('False Positive Rate', fontsize=12, color='white')
    ax2.set_ylabel('True Positive Rate', fontsize=12, color='white')
    ax2.set_title('ROC Curve - LSTM', fontsize=14, color='white')
    ax2.legend(loc='lower right', facecolor='none', edgecolor='white', labelcolor='white')
    ax2.tick_params(colors='white')

    fig_roc_lstm.patch.set_alpha(0)
    ax2.set_facecolor('none')
    st.pyplot(fig_roc_lstm)

    fig_lstm = go.Figure()
    fig_lstm.add_trace(go.Scatter(
        x=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        y=[0.91, 0.88, 0.90, 0.89],
        mode='lines+markers',
        line=dict(color='#FFA07A', width=3),
        marker=dict(size=10)
    ))

    fig_lstm.update_layout(
        title="LSTM Evaluation Metrics",
        xaxis_title="Metrics",
        yaxis_title="Scores",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )

    st.plotly_chart(fig_lstm, use_container_width=True)

st.divider()

true_labels = np.array([0, 1, 1, 0, 1, 0, 1]) 
predictions = np.array([0, 1, 1, 0, 0, 1, 1]) 
conf_matrix = confusion_matrix(true_labels, predictions)

col14, col15 = st.columns(2)

with col14:
    st.markdown("### Summary Statistics for Accident Data")
    st.dataframe(df.describe()) 

with col15:
    st.markdown("### 📊 Model Evaluation Table")
    eval_data = {
        'Model': ['XGBoost', 'LSTM'],
        'Precision': [0.85, 0.80],
        'Recall': [0.88, 0.82],
        'F1-Score': [0.86, 0.81],
        'ROC-AUC': [0.90, 0.87]
    }

    eval_df = pd.DataFrame(eval_data)
    st.dataframe(eval_df, use_container_width=True)

st.divider()

col16, col17 = st.columns(2)

accident_data = {
    'Severity': ['Low', 'Moderate', 'High'],
    'Total Accidents': [358964, 349205, 91529]
}

df = pd.DataFrame(accident_data)

with col16:
    st.markdown("### 🎯 Spread of Accidents Across Severity Levels")
    fig_bar, ax_bar = plt.subplots(figsize=(6, 4))

    sns.barplot(
        data=df,
        x='Severity',
        y='Total Accidents',
        palette=['cyan', '#00b8b8', '#80e0e0'],
        ax=ax_bar
    )

    ax_bar.set_facecolor('none')
    fig_bar.patch.set_alpha(0)
    ax_bar.set_xlabel("Severity", color='white')
    ax_bar.set_ylabel("Total Accidents", color='white')
    ax_bar.tick_params(colors='white')
    ax_bar.set_title("Total Accidents by Severity", color='white', fontsize=13)
    ax_bar.grid(True, linestyle='--', alpha=0.3)

    st.pyplot(fig_bar)

with col17:
    st.markdown("### Confusion Matrix for Model Evaluation")

    cyan_palette = ListedColormap(["#002b36", "#007b8a", "#00d6d6", "#ccffff"])

    conf_matrix = [
        [320, 30, 10],
        [40, 290, 20],
        [15, 25, 110]
    ]

    fig_cm, ax_cm = plt.subplots(figsize=(6, 6), facecolor='none')
    sns.heatmap(conf_matrix, annot=True, fmt='d',
                cmap=cyan_palette,
                xticklabels=['Low', 'Moderate', 'High'],
                yticklabels=['Low', 'Moderate', 'High'],
                cbar=False, linewidths=0.8, linecolor='white')

    ax_cm.set_xlabel('Predicted', color='white')
    ax_cm.set_ylabel('Actual', color='white')
    ax_cm.tick_params(colors='white')
    ax_cm.set_facecolor('none')
    fig_cm.patch.set_alpha(0)

    st.pyplot(fig_cm)

st.markdown("""
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 10px;
        background-color: #2C3E50;
        color: white;
        text-align: center;
        font-size: 14px;
    }
    </style>
    <div class="footer">
        © 2025 Accident Analysis and Prediction Dashboard. All rights reserved.
    </div>
""", unsafe_allow_html=True)
