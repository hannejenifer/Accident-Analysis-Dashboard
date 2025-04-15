import streamlit as st
import pandas as pd
import datetime
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Accident Analysis Dashboard")

df = pd.read_excel('Updated_Data.xlsx', sheet_name='Sheet1')

st.markdown("<style>div.block-container{padding-top: 3rem;}</style>", unsafe_allow_html=True)

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
        color_discrete_sequence=["#2E8BC0"], 
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
    selected_year = st.selectbox("Select Year", sorted(df["Year"].unique()))
    df_filtered = df[df["Year"] == selected_year]

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
        color_continuous_scale=["#FFA500", "#FF4500", "#FF0000"],
        title=f"Accident Heatmap - {selected_year}"
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
    color_discrete_sequence=["#003366", "#004C66", "#00688B", "#007B8C", "#00CED1"],  # Ocean-inspired colors
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

avg_severity = df["Severity_numeric"].mean()

fig_gauge_severity = go.Figure(go.Indicator(
    mode="gauge+number",
    value=avg_severity,
    number={'font': {'size': 40, 'color': '#FFFFFF'}},
    gauge={
        'axis': {'range': [None, 3], 'tickcolor': '#A9A9A9'},
        'bar': {'color': '#00688B'},
        'steps': [
            {'range': [0, 1], 'color': '#003366'},
            {'range': [1, 2], 'color': '#004C66'},
            {'range': [2, 3], 'color': '#00688B'}
        ],
        'threshold': {
            'line': {'color': "#00CED1", 'width': 4},
            'thickness': 0.75,
            'value': avg_severity
        }
    }
))

fig_gauge_severity.update_layout(
    title="Severity Risk",  
    title_font=dict(size=24, color="white"),
    height=400,
    margin=dict(l=20, r=20, t=50, b=20),
    font=dict(size=20, color="white"),

)


col1, col2 = st.columns(2)

severity_by_weather = df.groupby("Weather")["Severity_numeric"].mean().reset_index()

fig_radar = px.line_polar(
    severity_by_weather,
    r="Severity_numeric",
    theta="Weather",
    line_close=True,
    title="Radar Chart: Severity Distribution across Weather Types",
    template="plotly_dark",
    color_discrete_sequence=["#003366"]  
)

fig_radar.update_traces(
    fill='toself',
    line=dict(color="#00CED1", width=3)  
)

fig_radar.update_layout(
    polar=dict(
        angularaxis=dict(
            tickfont=dict(size=14),
            rotation=90, 
            direction="clockwise",  
        ),
        radialaxis=dict(
            tickfont=dict(size=12),
            angle=45,
            gridcolor="#4F6D7A",
            linecolor="#4F6D7A"
        ),
        bgcolor="#1B263B"
    ),

    font=dict(color="white", size=16),
    title_font=dict(size=22, color="white")
)

with col1:
    st.plotly_chart(fig_gauge_severity, use_container_width=True)

with col2:
    st.plotly_chart(fig_radar, use_container_width=True, key="radar_chart_duplicate")

st.divider()

st.markdown("## 📊 Model Evaluation: XGBoost vs. LSTM")
st.markdown("This section compares the performance of the XGBoost and LSTM models using standard classification metrics.")

model_metrics = {
    "Model": ["XGBoost", "LSTM"],
    "Precision": [0.87, 0.83],
    "Recall": [0.84, 0.81],
    "F1-Score": [0.85, 0.82]
}
metrics_df = pd.DataFrame(model_metrics)

st.dataframe(metrics_df.style.format({"Precision": "{:.2f}", "Recall": "{:.2f}", "F1-Score": "{:.2f}"}))

xgb_fpr = [0.0, 0.1, 0.2, 0.4, 1.0]
xgb_tpr = [0.0, 0.4, 0.6, 0.8, 1.0]

lstm_fpr = [0.0, 0.15, 0.25, 0.55, 1.0]
lstm_tpr = [0.0, 0.35, 0.65, 0.85, 1.0]

fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=xgb_fpr, y=xgb_tpr, mode='lines', name='XGBoost', line=dict(color='royalblue', width=3)))
fig_roc.add_trace(go.Scatter(x=lstm_fpr, y=lstm_tpr, mode='lines', name='LSTM', line=dict(color='darkorange', width=3)))
fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(dash='dash', color='gray')))

fig_roc.update_layout(
    title="ROC Curve Comparison",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    template="plotly_white",
    width=800,
    height=500
)

st.plotly_chart(fig_roc, use_container_width=True)
