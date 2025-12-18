# ACCIDENT ANALYSIS AND PREDICTION DASHBOARD (2021-25)
A user-friendly dashboard analyzing road accidents in Tamil Nadu, Kerala, and Karnataka from 2021–2023, and predicting trends for 2024–2025. It provides insights on accident patterns, severity, weather influence, district-wise hotspots, and state comparisons using interactive charts and machine learning models.

# DASHBOARD IMAGES
![image alt](https://github.com/hannejenifer/Accident-Analysis-Dashboard/blob/2afd92a0db3d3386d2e65882558bbbaafad0b4bf/STATE%20WISE%20COMPARISON.png)
![image alt](https://github.com/hannejenifer/Accident-Analysis-Dashboard/blob/2afd92a0db3d3386d2e65882558bbbaafad0b4bf/WEATHER%20WISE%20ACCIDENT%20COMPARISON.png)

# EXECUTIVE SUMMARY
The Accident Analysis & Prediction Dashboard visualizes and analyzes road accidents in Tamil Nadu, Kerala, and Karnataka from 2021–2023, while forecasting trends for 2024–2025. Through interactive charts, heatmaps, and machine learning models (XGBoost & LSTM), it highlights high-risk districts, shows the impact of weather and accident severity, and provides state-wise comparisons. This empowers authorities and organizations to make data-driven decisions to improve road safety and allocate resources effectively.

# BUSINESS PROBLEM
Road accidents in Southern India lead to significant human, economic, and property losses. Despite the availability of data, insights are often underutilized, limiting proactive safety measures. This project addresses:
 - Identifying high-risk districts and states for targeted interventions.
 - Understanding accident trends by severity, weather, and historical patterns.
 - Predicting future accidents to support proactive planning and resource allocation.
 - Enabling strategic decision-making for government bodies, insurers, and road safety organizations.

# NORTH STAR METRICS
NSM: **Reduction in Accidents per District per Year**
 - This metric captures the core value of your dashboard: helping authorities and stakeholders **prevent accidents and improve road safety.**
 - t’s actionable and measurable, and all other analyses (trends, severity, weather impact) feed into **understanding and improving this metric**.

# SUPPORTING DIMENSIONS
These are the dimensions or sub-metrics that feed into the NSM and help analyze the problem in detail:
  1. **Geographical Dimensions:**
        - District-wise total accidents
        - State-wise trends

  2. **Temporal Dimensions:**
        - Yearly trends (2021–2025)
        - Seasonal or monthly accident patterns

  3. **Contextual Dimensions:**
        - Weather conditions 
        - Accident severity (Low, Moderate, High)

  4. **Model & Prediction Metrics:**
        - Model Precision, Recall, F1-Score, ROC-AUC
        - Accuracy of 2024–2025 accident predictions

  5. **Impact Dimensions:**
        - High-risk districts identified
        - KPIs like “District with Most Accidents”

# METHODOLOGY
The project follows a structured approach to analyze and predict road accidents:
 - **Data Collection & Cleaning:** Compiled 2021–2023 accident data for Tamil Nadu, Kerala, and Karnataka; handled missing and inconsistent values.
 - **Exploratory Data Analysis (EDA):** Visualized yearly trends, district-level distribution, and weather impact; created heatmaps and radar charts to highlight high-risk areas.
 - **Feature Engineering:** Converted severity levels into numeric values and aggregated metrics by district, state, and weather conditions.
 - **Modeling & Prediction:** Built XGBoost and LSTM models to forecast accidents for 2024–2025; evaluated using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
 - **Dashboard Development:** Developed an interactive Streamlit dashboard to display visualizations, KPIs, and model predictions with dynamic filters for year, weather, and district.

# SKILLS
This project highlights expertise in:
 - **Programming & Data Handling:** Python, Pandas, NumPy
 - **Data Visualization:** Plotly, Matplotlib, Seaborn
 - **Machine Learning & Predictive Modeling:** XGBoost, LSTM, scikit-learn
 - **Dashboard Development:** Streamlit for interactive reporting
 - **Data Analysis & Insights:** Feature engineering, EDA, KPI generation
 - **Reporting & Communication:** Presenting insights for business and policy decisions

# RESULTS
The dashboard delivers key insights from 2021–2023 accident data and predictions for 2024–2025:
 - **Accident Overview:** Total accidents summarized by year, district, and state to identify high-risk areas.
 - **High-Risk Districts:** Interactive KPIs and tables highlight districts with the most accidents.
 - **Weather Impact:** Visualizations show which weather conditions contribute most to accidents.
 - **Severity Analysis:** Box plots and bar charts display accident severity trends across districts and weather.
 - **State-wise Trends**: Line charts illustrate accident progression across Tamil Nadu, Kerala, and Karnataka.
 - **Predictions (2024–2025):** XGBoost and LSTM models forecast future accidents with high precision and reliability.
 - **Visualization Highlights:** Heatmaps, severity gauges, and ROC curves provide actionable insights for stakeholders.

# BUSINESS RECOMMENDATIONS
Based on historical trends and model predictions:
 - **Targeted Safety Measures:** Focus campaigns and monitoring on high-risk districts.
 - **Weather-specific Precautions:** Implement preventive measures during high-risk weather conditions.
 - **Resource Allocation:** Prioritize emergency response resources in predicted high-accident districts.
 - **Policy Planning:** Use insights to improve infrastructure and enforce road safety regulations.
 - **Insurance Strategy:** Adjust risk assessments and premiums based on district and state-level accident trends.

# NEXT STEPS

To enhance the dashboard and predictive capabilities:
 + **Incorporate More Data:** Add traffic density, road conditions, and demographic factors to improve accuracy.
 + **Advanced Modeling:** Explore ensemble methods, GRU, or transformer-based models for better forecasts.
 + **Real-time Integration:** Connect to live traffic and accident feeds for dynamic analysis.
 + **Interactive Alerts:** Implement automated notifications for districts with rising accident trends.
 + **Web & Mobile Deployment:** Make the dashboard accessible via responsive web and mobile apps.
 + **Enhanced Visualizations:** Add geospatial clustering, predictive heatmaps, and time-lapse animations for deeper insights.

# SUMMARY AND INSIGHTS

The Accident Analysis Dashboard analyzes road accident data from 2021–2023 in Tamil Nadu, Kerala, and Karnataka and predicts trends for 2024–2025. It highlights high-risk districts, weather impact, and accident severity using interactive charts, heatmaps, and predictive models (XGBoost and LSTM).

**Key Insights:**
 - Certain districts consistently have the highest accidents.
 - Accidents increase under adverse weather conditions.
 - Most accidents are of moderate severity.
 - Predictions help authorities plan preventive measures and allocate resources effectively.
