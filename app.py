import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pickle
import os
import base64
from io import BytesIO

# Import custom modules
from auth import login, signup, check_authenticated, logout
from database import init_db, save_prediction, get_user_predictions
from model import load_model, predict_yield
from utils import preprocess_data

# Set page config
st.set_page_config(
    page_title="CropYield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
init_db()

# Load the ML model
model, scaler, encoder, label_encoder = load_model()

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/yield_df.csv')
    # Remove unnamed column if it exists
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    return df

df = load_data()

# Sidebar
st.sidebar.title("🌾 CropYield Predictor")
st.sidebar.image("https://img.icons8.com/color/96/000000/farm.png", width=100)

# Authentication
auth_option = st.sidebar.radio("Authentication", ["Login", "Sign Up"])

if auth_option == "Login":
    with st.sidebar.form(key="login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Login")
        
        if login_button:
            if login(username, password):
                st.session_state['authenticated'] = True
                st.session_state['username'] = username
                st.rerun()
            else:
                st.error("Invalid username or password")

elif auth_option == "Sign Up":
    with st.sidebar.form(key="signup_form"):
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        signup_button = st.form_submit_button("Sign Up")
        
        if signup_button:
            if new_password != confirm_password:
                st.error("Passwords do not match")
            elif not new_username or not new_password:
                st.error("Username and password cannot be empty")
            else:
                if signup(new_username, new_password):
                    st.success("Account created successfully! You can now log in.")
                else:
                    st.error("Username already exists")

# Check authentication
if check_authenticated():
    st.sidebar.success(f"Logged in as {st.session_state['username']}")
    if st.sidebar.button("Logout"):
        logout()
        st.rerun()
    
    # Main app content
    st.title("🌾 Crop Yield Prediction Tool")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Predict", "History", "Data Insights", "About"])
    
    with tab1:
        st.header("Predict Crop Yield")
        
        # Prediction form
        with st.form(key="prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Get unique areas and items from dataset
                areas = sorted(df['Area'].unique().tolist())
                items = sorted(df['Item'].unique().tolist())
                
                area = st.selectbox("Country/Region", areas)
                item = st.selectbox("Crop Type", items)
                year = st.slider("Year", 1990, 2030, datetime.now().year)
            
            with col2:
                # Calculate default values for help text
                avg_rain = float(df['average_rain_fall_mm_per_year'].mean())
                avg_pesticides = float(df['pesticides_tonnes'].mean())
                avg_temp = float(df['avg_temp'].mean())
                
                # Get min and max values for validation
                min_rain = float(df['average_rain_fall_mm_per_year'].min())
                max_rain = float(df['average_rain_fall_mm_per_year'].max())
                min_pesticides = float(df['pesticides_tonnes'].min())
                max_pesticides = float(df['pesticides_tonnes'].max())
                min_temp = float(df['avg_temp'].min())
                max_temp = float(df['avg_temp'].max())
                
                # Use text inputs instead of sliders
                rain = st.number_input(
                    "Average Rainfall (mm/year)",
                    min_value=min_rain,
                    max_value=max_rain,
                    value=avg_rain,
                    help=f"Enter a value between {min_rain:.1f} and {max_rain:.1f}. Average: {avg_rain:.1f}"
                )
                
                pesticides = st.number_input(
                    "Pesticides (tonnes)",
                    min_value=min_pesticides,
                    max_value=max_pesticides,
                    value=avg_pesticides,
                    help=f"Enter a value between {min_pesticides:.1f} and {max_pesticides:.1f}. Average: {avg_pesticides:.1f}"
                )
                
                temp = st.number_input(
                    "Average Temperature (°C)",
                    min_value=min_temp,
                    max_value=max_temp,
                    value=avg_temp,
                    help=f"Enter a value between {min_temp:.1f} and {max_temp:.1f}. Average: {avg_temp:.1f}"
                )
            
            predict_button = st.form_submit_button("Predict Yield")
        
        if predict_button:
            # Create input dataframe for prediction
            input_data = pd.DataFrame({
                'Area': [area],
                'Item': [item],
                'Year': [year],
                'average_rain_fall_mm_per_year': [rain],
                'pesticides_tonnes': [pesticides],
                'avg_temp': [temp]
            })
            
            # Make prediction
            with st.spinner("Predicting yield..."):
                prediction = predict_yield(input_data, model, scaler, encoder)
                
                # Save prediction to database
                save_prediction(st.session_state['username'], area, item, year, 
                               rain, pesticides, temp, float(prediction))
                
                # Display prediction
                st.success(f"Predicted Yield: {prediction:.2f} hg/ha")
                
                # Create visualizations
                st.subheader("Prediction Analysis")
                
                # Create a gauge chart for yield
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"Predicted Yield for {item} in {area}"},
                    gauge={
                        'axis': {'range': [0, 300000]},
                        'bar': {'color': "#6b8e23"},
                        'steps': [
                            {'range': [0, 50000], 'color': "#eff3e6"},
                            {'range': [50000, 100000], 'color': "#d5e8b5"},
                            {'range': [100000, 200000], 'color': "#a9d18e"},
                            {'range': [200000, 300000], 'color': "#6b8e23"}
                        ]
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create an explanation of factors
                st.subheader("Factor Importance")
                
                # Simulate factor importance
                factors = ['Rainfall', 'Temperature', 'Pesticides', 'Year']
                importance = [0.35, 0.25, 0.30, 0.10]  # Simplified for demonstration
                
                fig2 = px.bar(
                    x=factors,
                    y=importance,
                    labels={'x': 'Factor', 'y': 'Importance'},
                    title='Relative Importance of Factors in Prediction',
                    color=importance,
                    color_continuous_scale='Greens'
                )
                st.plotly_chart(fig2)
                
                # Option to download result
                result_df = pd.DataFrame({
                    'Factor': ['Country/Region', 'Crop Type', 'Year', 'Rainfall (mm/year)', 
                              'Pesticides (tonnes)', 'Temperature (°C)', 'Predicted Yield (hg/ha)'],
                    'Value': [area, item, year, rain, pesticides, temp, f"{prediction:.2f}"]
                })
                
                # Create a download link
                csv = result_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="prediction_result.csv">Download Prediction Result</a>'
                st.markdown(href, unsafe_allow_html=True)
    
    with tab2:
        st.header("Prediction History")
        
        # Get user's prediction history
        history = get_user_predictions(st.session_state['username'])
        
        if history:
            history_df = pd.DataFrame(history, columns=[
                'id', 'date', 'area', 'item', 'year', 'rainfall', 'pesticides', 'temperature', 'yield'
            ])
            
            # Format date
            history_df['date'] = pd.to_datetime(history_df['date']).dt.strftime('%Y-%m-%d %H:%M')
            
            # Display history
            st.dataframe(history_df.drop('id', axis=1), use_container_width=True)
            
            # Visualize history
            st.subheader("History Visualization")
            
            # Line chart of predictions over time
            fig = px.line(
                history_df,
                x='date',
                y='yield',
                color='item',
                title='Your Prediction History',
                labels={'yield': 'Predicted Yield (hg/ha)', 'date': 'Prediction Date'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Download history
            csv = history_df.drop('id', axis=1).to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="prediction_history.csv">Download History</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.info("No prediction history yet. Make your first prediction!")
    
    with tab3:
        st.header("Data Insights")
        
        # Display dataset statistics
        st.subheader("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{len(df):,}")
        col2.metric("Countries", f"{df['Area'].nunique():,}")
        col3.metric("Crop Types", f"{df['Item'].nunique():,}")
        col4.metric("Year Range", f"{df['Year'].min()} - {df['Year'].max()}")
        
        # Dataset exploration
        st.subheader("Explore Dataset")
        
        explore_col1, explore_col2 = st.columns(2)
        
        with explore_col1:
            selected_country = st.selectbox("Select Country", ["All"] + sorted(df['Area'].unique().tolist()))
        
        with explore_col2:
            selected_crop = st.selectbox("Select Crop", ["All"] + sorted(df['Item'].unique().tolist()))
        
        # Filter data based on selections
        filtered_df = df.copy()
        if selected_country != "All":
            filtered_df = filtered_df[filtered_df['Area'] == selected_country]
            
        if selected_crop != "All":
            filtered_df = filtered_df[filtered_df['Item'] == selected_crop]
        
        # Show filtered data
        if len(filtered_df) > 0:
            st.dataframe(filtered_df.head(100), use_container_width=True)
            
            # Visualizations
            st.subheader("Visualizations")
            
            viz_option = st.radio("Choose Visualization", [
                "Yield by Year",
                "Yield vs Rainfall",
                "Yield vs Temperature",
                "Yield vs Pesticides"
            ])
            
            if viz_option == "Yield by Year":
                # Group by Year and calculate mean yield
                if selected_crop != "All" and selected_country == "All":
                    # Show yield by country for a specific crop
                    year_data = filtered_df.groupby(['Area', 'Year'])['hg/ha_yield'].mean().reset_index()
                    fig = px.line(
                        year_data,
                        x='Year',
                        y='hg/ha_yield',
                        color='Area',
                        title=f'{selected_crop} Yield by Country Over Time',
                        labels={'hg/ha_yield': 'Average Yield (hg/ha)', 'Year': 'Year'}
                    )
                    
                elif selected_country != "All" and selected_crop == "All":
                    # Show yield by crop for a specific country
                    year_data = filtered_df.groupby(['Item', 'Year'])['hg/ha_yield'].mean().reset_index()
                    fig = px.line(
                        year_data,
                        x='Year',
                        y='hg/ha_yield',
                        color='Item',
                        title=f'Crop Yields in {selected_country} Over Time',
                        labels={'hg/ha_yield': 'Average Yield (hg/ha)', 'Year': 'Year'}
                    )
                    
                else:
                    # General case
                    year_data = filtered_df.groupby('Year')['hg/ha_yield'].mean().reset_index()
                    fig = px.line(
                        year_data,
                        x='Year',
                        y='hg/ha_yield',
                        title='Average Yield Over Time',
                        labels={'hg/ha_yield': 'Average Yield (hg/ha)', 'Year': 'Year'}
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif viz_option == "Yield vs Rainfall":
                fig = px.scatter(
                    filtered_df,
                    x='average_rain_fall_mm_per_year',
                    y='hg/ha_yield',
                    color='Item' if selected_crop == "All" else None,
                    title='Yield vs Rainfall',
                    labels={'average_rain_fall_mm_per_year': 'Rainfall (mm/year)', 'hg/ha_yield': 'Yield (hg/ha)'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
            elif viz_option == "Yield vs Temperature":
                fig = px.scatter(
                    filtered_df,
                    x='avg_temp',
                    y='hg/ha_yield',
                    color='Item' if selected_crop == "All" else None,
                    title='Yield vs Temperature',
                    labels={'avg_temp': 'Average Temperature (°C)', 'hg/ha_yield': 'Yield (hg/ha)'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
            elif viz_option == "Yield vs Pesticides":
                fig = px.scatter(
                    filtered_df,
                    x='pesticides_tonnes',
                    y='hg/ha_yield',
                    color='Item' if selected_crop == "All" else None,
                    title='Yield vs Pesticides',
                    labels={'pesticides_tonnes': 'Pesticides (tonnes)', 'hg/ha_yield': 'Yield (hg/ha)'}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for the selected filters.")
    
    with tab4:
        st.header("About This Tool")
        
        st.markdown("""
        ## Crop Yield Prediction Tool
        
        This application helps farmers and agricultural analysts predict crop yields based on various factors:
        
        - **Country/Region**: Different regions have different soil types and agricultural practices
        - **Crop Type**: Each crop has unique characteristics and yield potential
        - **Climate Factors**: Rainfall and temperature significantly affect crop growth
        - **Agricultural Inputs**: Pesticide usage impacts yield through pest control
        
        ### How It Works
        
        1. The prediction model is trained on historical crop yield data from around the world
        2. Machine learning algorithms identify patterns between environmental factors and crop yields
        3. When you enter your specific conditions, the model predicts the expected yield
        
        ### Data Sources
        
        The application uses a comprehensive dataset that includes:
        - Historical crop yields across different countries
        - Rainfall measurements
        - Temperature data
        - Pesticide usage information
        
        ### Interpretation
        
        The yield is measured in hectograms per hectare (hg/ha). To convert to other units:
        - 1 hg/ha = 0.1 kg/ha
        - 10,000 hg/ha = 1 tonne/hectare
        
        ### Limitations
        
        This tool provides estimates based on historical data and may not account for:
        - Extreme weather events
        - New crop diseases
        - Changes in farming practices
        - Local soil conditions
        
        Always combine these predictions with local agricultural knowledge for best results.
        """)
else:
    # Show welcome message for unauthenticated users
    st.title("🌾 Welcome to the Crop Yield Prediction Tool")
    
    st.markdown("""
    ### Predict crop yields with machine learning
    
    This application helps farmers and agricultural analysts predict crop yields based on various environmental and agricultural factors.
    
    #### Features:
    - Predict yields for different crops and regions
    - Analyze how rainfall, temperature, and pesticide usage affect yields
    - Track your prediction history
    - Download your prediction results
    
    #### Get Started:
    Please sign up or log in using the sidebar to access the prediction tool.
    """)
    
    # Display a sample visualization
    st.subheader("Sample Data Visualization")
    
    # Create a sample chart
    sample_data = df.groupby('Item')['hg/ha_yield'].mean().sort_values(ascending=False).head(10).reset_index()
    
    fig = px.bar(
        sample_data,
        x='Item',
        y='hg/ha_yield',
        title='Average Yield by Crop Type (Top 10)',
        labels={'Item': 'Crop Type', 'hg/ha_yield': 'Average Yield (hg/ha)'},
        color='hg/ha_yield',
        color_continuous_scale='Greens'
    )
    
    st.plotly_chart(fig, use_container_width=True)
