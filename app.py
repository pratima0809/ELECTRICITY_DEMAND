import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import shap
import os
import warnings

warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(page_title="Electricity Demand Forecasting", page_icon="⚡", layout="wide")

# Custom CSS for Professional Design
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to load data
@st.cache_data
def load_data():
    data_path = 'electricity_demand_forecasting_1500.csv'
    if not os.path.exists(data_path):
        # Import the generator from train_model if data is missing
        from train_model import generate_mock_data
        return generate_mock_data(data_path)
    return pd.read_csv(data_path)

def main():
    # Sidebar Navigation
    st.sidebar.title("⚡ Navigation")
    pages = [
        "Dashboard Home",
        "Dataset Explorer",
        "Exploratory Data Analysis",
        "Model Training",
        "Model Comparison",
        "Prediction",
        "Explainability",
        "About Project"
    ]
    choice = st.sidebar.radio("Go to", pages)

    df = load_data()

    if choice == "Dashboard Home":
        render_dashboard(df)
    elif choice == "Dataset Explorer":
        render_dataset_explorer(df)
    elif choice == "Exploratory Data Analysis":
        render_eda(df)
    elif choice == "Model Training":
        render_model_training(df)
    elif choice == "Model Comparison":
        render_model_comparison()
    elif choice == "Prediction":
        render_prediction(df)
    elif choice == "Explainability":
        render_explainability(df)
    elif choice == "About Project":
        render_about()

# -----------------------------------------------------------------
# Page Render Functions
# -----------------------------------------------------------------

def render_dashboard(df):
    st.title("⚡ Electricity Demand Forecasting System")
    st.markdown("### Project Overview")
    st.write("Welcome to the Electricity Demand Forecasting System. This application leverages machine learning to predict electricity demand based on various environmental and temporal factors.")
    
    st.markdown("### Objective")
    st.info("To accurately forecast electricity demand (in MWh) using predictive modeling techniques, aiding in efficient power generation and distribution planning.")
    
    st.markdown("### Key Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Records", value=f"{len(df):,}")
    with col2:
        st.metric(label="Avg Demand (MWh)", value=f"{df['Electricity_Demand_MWh'].mean():.2f}")
    with col3:
        st.metric(label="Max Demand (MWh)", value=f"{df['Electricity_Demand_MWh'].max():.2f}")
    with col4:
        st.metric(label="Min Demand (MWh)", value=f"{df['Electricity_Demand_MWh'].min():.2f}")
        
    st.markdown("### Dataset Summary")
    st.write(df.describe())

def render_dataset_explorer(df):
    st.title("📊 Dataset Explorer")
    
    # Search and Filter
    st.sidebar.subheader("Filter Data")
    min_temp, max_temp = float(df['Temperature'].min()), float(df['Temperature'].max())
    temp_range = st.sidebar.slider("Temperature Range", min_temp, max_temp, (min_temp, max_temp))
    
    selected_months = st.sidebar.multiselect("Select Month(s)", options=sorted(df['Month'].unique()), default=sorted(df['Month'].unique()))
    
    filtered_df = df[(df['Temperature'] >= temp_range[0]) & (df['Temperature'] <= temp_range[1]) & (df['Month'].isin(selected_months))]
    
    st.write(f"Showing {len(filtered_df)} records.")
    st.dataframe(filtered_df, use_container_width=True)
    
    # Download Option
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name='filtered_electricity_data.csv',
        mime='text/csv',
    )

def render_eda(df):
    st.title("📈 Exploratory Data Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Correlation", "Trend & Monthly", "Weather Impact", "Weekend Analysis"])
    
    with tab1:
        st.subheader("Correlation Heatmap")
        fig = px.imshow(df.corr(), text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.subheader("Monthly Demand")
        fig2 = px.box(df, x='Month', y='Electricity_Demand_MWh', color='Month', title="Demand Distribution by Month")
        st.plotly_chart(fig2, use_container_width=True)
        
        st.subheader("Demand Distribution")
        fig_dist = px.histogram(df, x='Electricity_Demand_MWh', nbins=50, title="Distribution of Electricity Demand")
        st.plotly_chart(fig_dist, use_container_width=True)
        
    with tab3:
        st.subheader("Weather Impact on Demand")
        col1, col2 = st.columns(2)
        with col1:
            fig3 = px.scatter(df, x='Temperature', y='Electricity_Demand_MWh', color='Season' if 'Season' in df.columns else 'Month', title="Temperature vs Demand")
            st.plotly_chart(fig3, use_container_width=True)
        with col2:
            fig4 = px.scatter(df, x='Humidity', y='Electricity_Demand_MWh', color='Rainfall', title="Humidity vs Demand")
            st.plotly_chart(fig4, use_container_width=True)
            
    with tab4:
        st.subheader("Weekend vs Weekday Demand")
        weekend_avg = df.groupby('Weekend')['Electricity_Demand_MWh'].mean().reset_index()
        weekend_avg['Weekend'] = weekend_avg['Weekend'].map({0: 'Weekday', 1: 'Weekend'})
        fig5 = px.bar(weekend_avg, x='Weekend', y='Electricity_Demand_MWh', color='Weekend', title="Average Demand: Weekday vs Weekend")
        st.plotly_chart(fig5, use_container_width=True)

def render_model_training(df):
    st.title("⚙️ Model Training")
    st.write("Train various machine learning models and evaluate their performance.")
    
    if st.button("Start Training Models"):
        with st.spinner('Preprocessing data and training models... This may take a moment.'):
            # Preprocessing
            X = df.drop(columns=['Electricity_Demand_MWh'])
            y = df['Electricity_Demand_MWh']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
                "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
            }
            
            results = []
            best_model = None
            best_r2 = -float('inf')
            
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_test_scaled)
                
                mae = mean_absolute_error(y_test, preds)
                mse = mean_squared_error(y_test, preds)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, preds)
                
                results.append({
                    "Model": name,
                    "MAE": mae,
                    "MSE": mse,
                    "RMSE": rmse,
                    "R²": r2
                })
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = model
            
            # Save models and results
            joblib.dump(best_model, 'model.pkl')
            joblib.dump(scaler, 'scaler.pkl')
            pd.DataFrame(results).to_csv('model_results.csv', index=False)
            
        st.success("Training completed! Models and metrics have been saved.")
        
        st.subheader("Training Metrics")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df.style.highlight_max(subset=['R²'], color='lightgreen').highlight_min(subset=['MAE', 'RMSE'], color='lightgreen'), use_container_width=True)

def render_model_comparison():
    st.title("🏆 Model Comparison")
    
    if os.path.exists('model_results.csv'):
        results_df = pd.read_csv('model_results.csv')
        
        st.subheader("Model Leaderboard")
        st.dataframe(results_df.sort_values(by="R²", ascending=False), use_container_width=True)
        
        # Interactive Comparison Chart
        fig = px.bar(results_df, x='Model', y='R²', color='Model', title="R² Score Comparison")
        st.plotly_chart(fig, use_container_width=True)
        
        best_model_name = results_df.sort_values(by="R²", ascending=False).iloc[0]['Model']
        st.info(f"**Best Model Auto-Selected:** {best_model_name}")
    else:
        st.warning("No model results found. Please train the models in the 'Model Training' page first.")

def render_prediction(df):
    st.title("🔮 Electricity Demand Prediction")
    st.write("Provide the inputs below to predict the electricity demand.")
    
    # Check if model exists
    if not os.path.exists('model.pkl') or not os.path.exists('scaler.pkl'):
        st.error("Model or Scaler not found! Please train the models first.")
        return
        
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # User Inputs
    col1, col2 = st.columns(2)
    with col1:
        temp = st.number_input("Temperature (°C)", value=float(df['Temperature'].mean()))
        humidity = st.number_input("Humidity (%)", value=float(df['Humidity'].mean()))
        wind = st.number_input("Wind Speed (km/h)", value=float(df['Wind Speed'].mean()))
        rain = st.number_input("Rainfall (mm)", value=float(df['Rainfall'].mean()))
    with col2:
        day = st.selectbox("Day Of Week (0=Monday, 6=Sunday)", options=list(range(7)))
        month = st.selectbox("Month", options=list(range(1, 13)))
        weekend = st.radio("Weekend?", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        holiday = st.radio("Holiday?", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        
    if st.button("Predict Demand"):
        input_data = pd.DataFrame({
            'Temperature': [temp],
            'Humidity': [humidity],
            'Wind Speed': [wind],
            'Rainfall': [rain],
            'Day Of Week': [day],
            'Month': [month],
            'Holiday': [holiday],
            'Weekend': [weekend]
        })
        
        # Ensure correct column order matching training data
        feature_cols = df.drop(columns=['Electricity_Demand_MWh']).columns
        input_data = input_data[feature_cols]
        
        try:
            scaled_input = scaler.transform(input_data)
            prediction = model.predict(scaled_input)[0]
            
            st.success("Prediction Successful!")
            st.markdown(f"<div class='metric-card' style='background-color: #d4edda; color: #155724;'><h3>Predicted Electricity Demand</h3><h2>{prediction:.2f} MWh</h2></div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

def render_explainability(df):
    st.title("🧠 Model Explainability")
    
    if not os.path.exists('model.pkl') or not os.path.exists('scaler.pkl'):
        st.warning("Please train the models first to view explainability.")
        return
        
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    X = df.drop(columns=['Electricity_Demand_MWh'])
    feature_cols = X.columns
    
    # Feature Importance
    st.subheader("Feature Importance")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feat_imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values(by='Importance', ascending=True)
        fig = px.bar(feat_imp_df, x='Importance', y='Feature', orientation='h', title="Feature Importance (Tree-based model)")
        st.plotly_chart(fig, use_container_width=True)
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
        feat_imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values(by='Importance', ascending=True)
        fig = px.bar(feat_imp_df, x='Importance', y='Feature', orientation='h', title="Feature Importance (Linear model coefficients)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("The selected best model does not have a native feature importance attribute.")
        
    # SHAP Analysis
    st.subheader("SHAP Analysis")
    st.write("SHAP (SHapley Additive exPlanations) values break down a prediction to show the impact of each feature.")
    
    if st.button("Generate SHAP Summary Plot (May take a moment)"):
        with st.spinner("Calculating SHAP values..."):
            try:
                # Sample a background dataset for SHAP to save time
                X_sampled = shap.utils.sample(X, 100)
                X_sampled_scaled = scaler.transform(X_sampled)
                
                explainer = shap.Explainer(model.predict, X_sampled_scaled)
                shap_values = explainer(X_sampled_scaled)
                
                fig, ax = plt.subplots()
                shap.summary_plot(shap_values, X_sampled, feature_names=feature_cols, show=False)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Could not generate SHAP values for the current model. Error: {str(e)}")

def render_about():
    st.title("ℹ️ About the Project")
    
    st.markdown("""
    ### Problem Statement
    Accurately predicting electricity demand is critical for grid stability, resource planning, and cost optimization. Overestimating demand leads to wasted energy and increased costs, while underestimating can result in power outages and grid failures. This project aims to build a robust machine learning model to forecast demand using weather and calendar data.
    
    ### Methodology
    1. **Data Collection & Preprocessing:** The dataset consists of historical demand, weather variables (Temperature, Humidity, Wind, Rain), and temporal variables. Missing values and outliers are handled, and features are standardized.
    2. **Exploratory Data Analysis (EDA):** Visualizing correlations, seasonal trends, and weather impacts on electricity usage.
    3. **Model Selection:** We evaluate multiple models including Linear Regression, Random Forest, Gradient Boosting, and XGBoost.
    4. **Evaluation:** Models are compared using MAE, MSE, RMSE, and R² scores. The best model is chosen for deployment.
    5. **Deployment:** The best model is served via a Streamlit web application, allowing interactive predictions and explainability.
    
    ### Team Information
    - **Developer:** Senior Machine Learning Engineer
    - **Tech Stack:** Python, Streamlit, Scikit-Learn, XGBoost, Plotly, Pandas
    """)

if __name__ == "__main__":
    main()
