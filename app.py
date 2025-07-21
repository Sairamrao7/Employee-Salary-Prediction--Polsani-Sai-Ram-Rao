import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import pickle
import os

# --- 1. Model Training and Preprocessing (Run once to create the model file) ---
def train_and_save_model(data_path='Salary_Data.csv'):
    """
    Loads data, preprocesses it, trains a RandomForestRegressor model,
    and saves the model and encoder to disk.
    """
    # Load the dataset
    df = pd.read_csv(data_path)

    # --- Preprocessing ---
    # Impute missing numerical values with the mean
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    # Exclude Salary from imputation if it's a feature, but it's our target
    if 'Salary' in numerical_cols:
        numerical_cols.remove('Salary')
        
    numerical_imputer = SimpleImputer(strategy='mean')
    df[numerical_cols] = numerical_imputer.fit_transform(df[numerical_cols])

    # Impute missing categorical values with the mode
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

    # Drop rows where 'Salary' is still NaN
    df.dropna(subset=['Salary'], inplace=True)

    # Identify categorical columns for encoding
    categorical_cols_to_encode = df.select_dtypes(include='object').columns.tolist()

    # Apply one-hot encoding
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_categorical_features = encoder.fit_transform(df[categorical_cols_to_encode])
    encoded_categorical_df = pd.DataFrame(encoded_categorical_features, columns=encoder.get_feature_names_out(categorical_cols_to_encode))

    # Drop original categorical columns and concatenate encoded features
    df_encoded = df.drop(columns=categorical_cols_to_encode)
    df_encoded = pd.concat([df_encoded.reset_index(drop=True), encoded_categorical_df.reset_index(drop=True)], axis=1)

    # Separate features (X) and target variable (y)
    X = df_encoded.drop(columns=['Salary'])
    y = df_encoded['Salary']

    # --- Model Training ---
    # Using the best model from your notebook: RandomForestRegressor
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)

    # --- Save Model and Encoder ---
    with open('salary_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    with open('salary_encoder.pkl', 'wb') as encoder_file:
        pickle.dump(encoder, encoder_file)
    with open('model_columns.pkl', 'wb') as columns_file:
        pickle.dump(X.columns, columns_file)
        
    print("Model, encoder, and columns have been trained and saved successfully.")

# --- 2. Streamlit Application ---
def run_app():
    """
    Runs the Streamlit web application interface.
    """
    # Load the trained model, encoder, and columns
    try:
        with open('salary_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('salary_encoder.pkl', 'rb') as encoder_file:
            encoder = pickle.load(encoder_file)
        with open('model_columns.pkl', 'rb') as columns_file:
            model_columns = pickle.load(columns_file)
    except FileNotFoundError:
        st.error("Model files not found. Please run the training function first.")
        return

    # --- Streamlit UI ---
    st.set_page_config(page_title="Salary Predictor", page_icon="💼", layout="wide")
    
    st.title("💼 Employee Salary Predictor")
    st.markdown("Enter the employee's details below to get an estimated salary prediction.")
    
    # Create columns for a cleaner layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Demographics & Experience")
        age = st.slider("Age", 20, 65, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        years_of_experience = st.slider("Years of Experience", 0, 40, 5)

    with col2:
        st.subheader("Education & Role")
        education_level = st.selectbox("Education Level", ["Bachelor's", "Master's", "PhD"])
        # For simplicity, using a text input for job title. The one-hot encoder will handle new titles.
        job_title = st.text_input("Job Title", "Software Engineer")

    # Prediction button
    if st.button("Predict Salary", use_container_width=True):
        # --- Create a DataFrame from user input ---
        input_data = pd.DataFrame({
            'Age': [age],
            'Years of Experience': [years_of_experience],
            'Gender': [gender],
            'Education Level': [education_level],
            'Job Title': [job_title]
        })

        # --- Preprocess the input data ---
        # One-hot encode the categorical features
        encoded_input_categorical = encoder.transform(input_data[['Gender', 'Education Level', 'Job Title']])
        encoded_input_df = pd.DataFrame(encoded_input_categorical, columns=encoder.get_feature_names_out(['Gender', 'Education Level', 'Job Title']))

        # Combine numerical features with encoded categorical features
        input_numerical = input_data[['Age', 'Years of Experience']].reset_index(drop=True)
        processed_input = pd.concat([input_numerical, encoded_input_df], axis=1)
        
        # Ensure the processed input has all the columns the model was trained on
        processed_input = processed_input.reindex(columns=model_columns, fill_value=0)

        # --- Make Prediction ---
        prediction = model.predict(processed_input)[0]

        # --- Display the result ---
        st.success(f"Predicted Salary: **${prediction:,.2f}**")

# --- Main execution block ---
if __name__ == "__main__":
    # Create a dummy CSV for the first run if it doesn't exist
    if not os.path.exists('Salary_Data.csv'):
        st.info("Creating a dummy 'Salary_Data.csv'. Please replace it with your actual data.")
        dummy_data = {
            'Age': [32, 28, 45, 36, 52],
            'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'Education Level': ["Bachelor's", "Master's", "PhD", "Bachelor's", "Master's"],
            'Job Title': ['Software Engineer', 'Data Analyst', 'Senior Manager', 'Sales Associate', 'Director'],
            'Years of Experience': [5.0, 3.0, 15.0, 7.0, 20.0],
            'Salary': [90000.0, 65000.0, 150000.0, 60000.0, 200000.0]
        }
        pd.DataFrame(dummy_data).to_csv('Salary_Data.csv', index=False)

    # Check if the model files exist. If not, train and save them.
    if not (os.path.exists('salary_model.pkl') and os.path.exists('salary_encoder.pkl')):
        train_and_save_model('Salary_Data.csv')
    
    # Run the Streamlit app
    run_app()
