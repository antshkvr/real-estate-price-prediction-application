import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# Application title
st.title("Real Estate Price Prediction Application")

# File uploader for dataset
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

if uploaded_file:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    categorical_columns = data.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
    st.write("Dataset Preview:")
    st.dataframe(data.head())

    # Check for null values
    st.write("Null values in each column:")
    st.write(data.isnull().sum())

    # Filter null values
    if st.checkbox("Drop rows with null values"):
        st.write(f"Shape before dropping null values: {data.shape}")
        data = data.dropna()
        st.write(f"Shape after dropping null values: {data.shape}")

    # Select target variable
    target_col = st.selectbox("Select the target column", data.columns)
    feature_cols = [col for col in data.columns if col != target_col]
    X = data[feature_cols]
    y = data[target_col]

    # Train-test split
    test_size = st.slider("Test size (%)", 10, 50, 20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42)

    # Model selection
    model_name = st.selectbox("Select a model", ["LightGBM", "XGBoost"])

    if st.button("Train Model"):

        if model_name == "LightGBM":
            model = lgb.LGBMRegressor(random_state=42)
        else:
            model = xgb.XGBRegressor(random_state=42, use_label_encoder=False, eval_metric='rmse')

        # Train the model
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        st.write(f"Model Evaluation for {model_name}:")
        st.write(f"MAE: {mae:.2f}")
        st.write(f"MSE: {mse:.2f}")
        st.write(f"RMSE: {rmse:.2f}")
        st.write(f"R^2 Score: {r2:.2f}")

        # Interactive Scatter Plot
        st.subheader("True vs Predicted Values")
        fig = px.scatter(x=y_test, y=predictions, title=f"True vs Predicted Values ({model_name})",
                         labels={"x": "True Values", "y": "Predicted Values"})
        st.plotly_chart(fig)

        # Additional Analysis with Plotly
        st.subheader("Target Variable Distribution")
        fig = px.histogram(data, x=target_col, title=f"Distribution of {target_col}")
        st.plotly_chart(fig)

        st.subheader("Boxplot for Target Variable")
        fig = px.box(data, y=target_col, title=f"Boxplot of {target_col}")
        st.plotly_chart(fig)

        st.subheader("Correlation Heatmap")
        fig = px.imshow(data.corr(), text_auto=False, title="Correlation Heatmap", width=800, height=800)
        st.plotly_chart(fig)
