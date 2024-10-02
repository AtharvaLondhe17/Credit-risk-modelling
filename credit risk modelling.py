import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Step 1: Generate a Larger Random Dataset
def generate_random_dataset(num_samples=10000):
    # Set a random seed for reproducibility
    np.random.seed(42)

    # Generate random data
    data = {
        'age': np.random.randint(18, 70, size=num_samples),
        'income': np.random.randint(10000, 150000, size=num_samples),
        'loan_amount': np.random.randint(1000, 100000, size=num_samples),
        'credit_score': np.random.randint(300, 850, size=num_samples),
    }

    # Simulate loan default status based on some criteria
    data['loan_default'] = np.where(
        (data['credit_score'] < 600) | (data['loan_amount'] > 50000), 1, 0
    )

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv('random_credit_data_large.csv', index=False)
    print("Larger random dataset generated and saved as 'random_credit_data_large.csv'.")

# Generate the dataset (uncomment this line to generate a new dataset)
# generate_random_dataset()

# Step 2: Streamlit Application for Credit Risk Model
def run_credit_risk_app():
    # Load dataset
    data = pd.read_csv('random_credit_data_large.csv')

    # Data cleaning and preprocessing
    data.fillna(data.mean(), inplace=True)

    # Convert categorical variables if necessary
    data = pd.get_dummies(data, drop_first=True)

    # Define features and target variable
    X = data.drop('loan_default', axis=1)
    y = data['loan_default']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # User Input for Prediction
    st.title("Credit Risk Prediction")
    
    # Option to display charts
    show_charts = st.checkbox("Show Charts")
    
    if show_charts:
        st.write("### Data Visualization")
        
        # Visualizing loan default distribution
        plt.figure(figsize=(8, 4))
        sns.countplot(x='loan_default', data=data)
        plt.title('Loan Default Distribution')
        st.pyplot(plt)

        # Visualizing age distribution
        plt.figure(figsize=(8, 4))
        sns.histplot(data['age'], bins=30, kde=True)
        plt.title('Age Distribution')
        st.pyplot(plt)

        # Visualizing income distribution
        plt.figure(figsize=(8, 4))
        sns.histplot(data['income'], bins=30, kde=True)
        plt.title('Income Distribution')
        st.pyplot(plt)

    st.write("### Make a Prediction")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Income", min_value=1000, max_value=150000, value=50000)
    loan_amount = st.number_input("Loan Amount", min_value=1000, max_value=100000, value=10000)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)

    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'age': [age],
        'income': [income],
        'loan_amount': [loan_amount],
        'credit_score': [credit_score]
    })

    # Make prediction
    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[:, 1][0]
        if prediction == 1:
            st.write(f"The model predicts a **Loan Default** with a probability of **{probability:.2f}**.")
        else:
            st.write(f"The model predicts **No Loan Default** with a probability of **{1 - probability:.2f}**.")

# Run the dataset generation and the Streamlit app
# generate_random_dataset()  # Uncomment this to generate a new dataset
run_credit_risk_app()

# Run this Streamlit app by executing the following command in your terminal:
# streamlit run your_script_name.py
