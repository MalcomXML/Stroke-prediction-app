import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import streamlit as st
from sklearn.metrics import accuracy_score



df = pd.read_csv("df_encoded.csv")

# Separate features and target variable
X = df.drop(['ID', 'STROKE'], axis=1)
y = df['STROKE']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess categorical variables
categorical_features = ['GENDER', 'MARRIED', 'WK', 'RT', 'SS']
categorical_transformer = OneHotEncoder(drop='first')

# Preprocess numerical variables
numerical_features = ['AGE', 'AGL', 'BMI']
numerical_transformer = StandardScaler()

# Preprocess binary variables
binary_features = ['HT', 'HD']
binary_transformer = StandardScaler()  # Use StandardScaler for binary variables

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        ('bin', binary_transformer, binary_features)
    ])

# Create a pipeline with the logistic regression model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Fit the model
model.fit(X_train, y_train)

# Streamlit UI
st.title("Stroke Prediction App")

# User input
gender = st.radio("Gender", df['GENDER'].unique())
age = st.slider("Age", int(df['AGE'].min()), int(df['AGE'].max()), int(df['AGE'].median()))
married = st.radio("Marital Status", df['MARRIED'].unique())
work_type = st.selectbox("Work Type", df['WK'].unique())
residence_type = st.selectbox("Residence Type", df['RT'].unique())
avg_glucose_level = st.slider("Average Glucose Level", float(df['AGL'].min()), float(df['AGL'].max()), float(df['AGL'].median()))
bmi = st.slider("BMI", float(df['BMI'].min()), float(df['BMI'].max()), float(df['BMI'].median()))
smoking_status = st.selectbox("Smoking Status", df['SS'].unique())

# Button to trigger prediction
if st.button("Predict"):
    # Convert user input to a DataFrame
    user_data = pd.DataFrame({
        'GENDER': [gender],
        'AGE': [age],
        'MARRIED': [married],
        'WK': [work_type],
        'RT': [residence_type],
        'AGL': [avg_glucose_level],
        'BMI': [bmi],
        'SS': [smoking_status],
        'HT': [0],  # Placeholder value for binary variables
        'HD': [0]   # Placeholder value for binary variables
    })

    # Make a prediction
    prediction = model.predict(user_data)

    # Display prediction
    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.error("The model predicts a stroke.")
    else:
        st.success("The model predicts no stroke.")