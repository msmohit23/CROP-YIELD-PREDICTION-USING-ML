import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_data(df):
    """Preprocess the data for analysis"""
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Remove unnamed column if it exists
    if 'Unnamed: 0' in df_processed.columns:
        df_processed.drop('Unnamed: 0', axis=1, inplace=True)
    
    # Drop duplicates
    df_processed.drop_duplicates(inplace=True)
    
    return df_processed

def encode_categorical(df, columns):
    """Encode categorical variables"""
    # Create a copy to avoid modifying the original
    df_encoded = df.copy()
    
    # Create encoder
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    
    # Fit and transform
    encoded_data = encoder.fit_transform(df_encoded[columns])
    
    # Get feature names
    feature_names = encoder.get_feature_names_out(columns)
    
    # Create DataFrame with encoded variables
    encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=df_encoded.index)
    
    # Remove original columns and add encoded ones
    df_encoded = df_encoded.drop(columns, axis=1)
    df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
    
    return df_encoded, encoder

def scale_numerical(df, columns):
    """Scale numerical variables"""
    # Create a copy to avoid modifying the original
    df_scaled = df.copy()
    
    # Create scaler
    scaler = StandardScaler()
    
    # Fit and transform
    df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
    
    return df_scaled, scaler

def create_features(df):
    """Create new features from existing ones"""
    # Create a copy to avoid modifying the original
    df_features = df.copy()
    
    # Example: Create a rainfall to temperature ratio
    df_features['rain_temp_ratio'] = df_features['average_rain_fall_mm_per_year'] / df_features['avg_temp']
    
    # Example: Create pesticides per rainfall
    df_features['pesticides_per_rainfall'] = df_features['pesticides_tonnes'] / df_features['average_rain_fall_mm_per_year']
    
    # Example: Create year groups
    df_features['year_group'] = pd.cut(df_features['Year'], 
                                      bins=[1989, 1995, 2000, 2005, 2010, 2015, 2020, 2025, 2030],
                                      labels=['1990-1995', '1996-2000', '2001-2005', '2006-2010', 
                                              '2011-2015', '2016-2020', '2021-2025', '2026-2030'])
    
    return df_features
