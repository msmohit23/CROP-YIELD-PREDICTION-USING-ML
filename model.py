import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

def train_model():
    """Train a machine learning model on the crop yield dataset"""
    # Load the data
    df = pd.read_csv('data/yield_df.csv')
    
    # Remove unnamed column if it exists
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)
    
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    # Split features and target
    X = df.drop('hg/ha_yield', axis=1)
    y = df['hg/ha_yield']
    
    # Define categorical and numerical features
    categorical_features = ['Area', 'Item']
    numerical_features = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # Create label encoder for visualization
    label_encoder = LabelEncoder()
    label_encoder.fit(df['Item'])
    
    # Create and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocess the data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Train the model
    model.fit(X_train_processed, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_processed)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model RMSE: {rmse}")
    print(f"Model R²: {r2}")
    
    # Create encoder for later use (without fitting, since it's already fit)
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(X[categorical_features])
    
    # Create scaler for later use (without fitting, since it's already fit)
    scaler = StandardScaler()
    scaler.fit(X[numerical_features])
    
    # Save the model, scaler, encoder, and label encoder
    with open('model.pkl', 'wb') as f:
        pickle.dump((model, preprocessor), f)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    return model, preprocessor

def load_model():
    """Load the trained model or train if it doesn't exist"""
    if not os.path.exists('model.pkl'):
        model, preprocessor = train_model()
        
        # Load the individual components
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
    else:
        # Load the model and preprocessor
        with open('model.pkl', 'rb') as f:
            model, preprocessor = pickle.load(f)
        
        # Load the scaler
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Load the encoder
        with open('encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        
        # Load the label encoder
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
    
    return model, scaler, encoder, label_encoder

def predict_yield(input_df, model, scaler, encoder):
    """Make a prediction using the trained model"""
    # Extract features
    categorical_features = ['Area', 'Item']
    numerical_features = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
    
    # Scale numerical features
    numerical_data = scaler.transform(input_df[numerical_features])
    
    # Encode categorical features
    categorical_data = encoder.transform(input_df[categorical_features])
    
    # Combine features
    processed_data = np.hstack([numerical_data, categorical_data.toarray()])
    
    # Make prediction
    prediction = model.predict(processed_data)
    
    return prediction[0]
