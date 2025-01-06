from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


# Декомпозиція часу (Trend, Seasonal, Residual)
def decompose_time_series(data, column, target_column):
    from statsmodels.tsa.seasonal import seasonal_decompose

    # Перетворення в часову серію, агреговану за середніми значеннями
    ts = data.set_index(column)[target_column]
    decomposition = seasonal_decompose(ts, model='additive', period=52)
    return decomposition.trend, decomposition.seasonal, decomposition.resid


# Application title
st.title("Real Estate Price Prediction Application")

# File uploader for dataset
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

if uploaded_file:
    # Load the dataset
    data = pd.read_csv(uploaded_file)


    def convert_week_to_date(week_str):
        year, week = int(week_str[:4]), int(week_str[5:])
        return datetime.strptime(f'{year}-W{week - 1}-1', "%Y-W%W-%w")


    data['date'] = data['week_no'].apply(convert_week_to_date)

    categorical_columns = data.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
    st.write("Dataset Preview:")
    st.dataframe(data)

    # Check for null values
    st.write("Null values in each column:")
    st.write(data.isnull().sum())

    # Options for handling null values
    fill_methods = st.selectbox("Select method to handle null values", ["None", "Mean", "Median", "Mode", "Drop"])

    # Create new dataset based on selected method
    if fill_methods != "None":
        new_data = data.copy()
        if fill_methods == "Mean":
            new_data.fillna(new_data.mean(), inplace=True)
        elif fill_methods == "Median":
            new_data.fillna(new_data.median(), inplace=True)
        elif fill_methods == "Mode":
            new_data.fillna(new_data.mode().iloc[0], inplace=True)
        elif fill_methods == "Drop":
            new_data = new_data.dropna()
    else:
        new_data = data.copy()
        # Check for null values
    data_to_train = new_data.copy()
    data_to_train = data_to_train.drop(columns="date")

    st.write("Null values in each column:")
    st.write(new_data.isnull().sum())
    st.write(f"Shape of dataset before handling NULLs: {new_data.shape}")
    st.write(f"Shape of dataset after handling NULLs: {new_data.shape}")

    st.write("Updated Dataset Preview:")
    st.dataframe(new_data.head())

    # Select target variable
    target_col = st.selectbox("Select the target column", data_to_train.columns)
    feature_cols = [col for col in data_to_train.columns if col != target_col]
    X = data_to_train[feature_cols]
    y = data_to_train[target_col]

    # Вибираємо колонки, які є числовими та мають часову залежність (наприклад, week_no або дата)
    time_columns = [col for col in new_data.columns if 'date' in col.lower() and target_col != 'week_no']

    # Обрання часової колонки для декомпозиції
    if time_columns:
        feature_col = st.selectbox("Select a feature column for decomposition", time_columns)

        if st.checkbox(f"Decompose {feature_col}"):

            trend, seasonal, resid = decompose_time_series(new_data, feature_col, target_col)

            method = st.selectbox("Select analysis", ["trend", "seasonal", "residual"])
            if st.button("Analyze"):
                if method == "trend":
                    st.write("Trend")
                    st.line_chart(trend)
                elif method == "seasonal":
                    st.write("Seasonal Component:")
                    st.line_chart(seasonal)
                elif method == "residual":
                    st.write("Residual Component:")
                    st.line_chart(resid)
        if st.checkbox("Fit ARIMA Model"):
            new_data['date'] = pd.to_datetime(new_data['date'])
            new_data.set_index('date', inplace=True)

            # Визначення розміру тестової вибірки
            train_size_arima = st.slider("Test size (%)", 10, 80, 20, key='0')
            train_size = int(len(new_data) * train_size_arima / 100)
            train, test = new_data[:train_size], new_data[train_size:]
            size = len(test)



            # Визначення параметрів ARIMA
            st.sidebar.header("ARIMA Parameters")
            p = st.sidebar.slider("p (AR order)", 0, 5, 2)
            d = st.sidebar.slider("d (Differencing order)", 0, 2, 1)
            q = st.sidebar.slider("q (MA order)", 0, 5, 2)
            daterange = pd.date_range(train.index[0], periods=50)
            model = ARIMA(train[target_col], order=(p, d, q))
            model_fit = model.fit()
            st.write(model_fit.summary())
            forecast_series = model_fit.predict(start=len(train)+1, end=len(train) + len(test))
            st.dataframe(forecast_series)
            residuals = test[target_col].values - forecast_series.values

            mae = mean_absolute_error(test[target_col], forecast_series.values)
            mse = mean_squared_error(test[target_col], forecast_series.values)
            rmse = np.sqrt(mse)
            r2 = r2_score(test[target_col], forecast_series.values)

            st.write(f"Model Evaluation for ARIMA:")
            st.write(f"MAE: {mae}")
            st.write(f"MSE: {mse}")
            st.write(f"RMSE: {rmse}")
            st.write(f"R^2 Score: {r2}")

            # # Візуалізація прогнозу
            st.header("Forecast vs Actual")
            plt.figure(figsize=(10, 6))
            plt.plot(train.index, train[target_col].values,  label='Train')
            plt.plot(test.index, test[target_col].values, label='Test', color='orange')
            plt.plot(test.index, forecast_series.values, label='Test', color='blue')
            plt.title('ARIMA Forecast vs Actual')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt.gcf())




            st.dataframe(residuals)
            # Гістограма залишків
            st.header("Residuals Distribution")
            st.bar_chart(residuals)


            # ACF залишків
            st.header("Residuals Autocorrelation")
            st.line_chart(residuals)

        else:
            st.write(f"No decomposition performed for {feature_col}.")
    else:
        st.write("No suitable columns for time series decomposition.")
    target_col = st.selectbox("Select the target column", data.columns, 
                             index=list(data.columns).index("median_price_pln") if "median_price_pln" in data.columns else 0)
    feature_cols = [col for col in data.columns if col != target_col]
    X = data[feature_cols]
    y = data[target_col]

    # Train-test split
    test_size = st.slider("Test size (%)", 10, 80, 20, key='2')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42)

    # Model selection
    model_name = st.selectbox("Select a model", ["LightGBM", "XGBoost", "Random Forest"])

    if st.button("Train Model"):

        if model_name == "LightGBM":
            model = lgb.LGBMRegressor(random_state=42)
        elif model_name == "XGBoost":
            model = xgb.XGBRegressor(random_state=42, use_label_encoder=False, eval_metric='rmse')
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Train the model
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Additional Analysis with Plotly
        st.subheader("Target Variable Distribution")
        fig = px.histogram(data_to_train, x=target_col, title=f"Distribution of {target_col}")
        st.plotly_chart(fig)

        st.subheader("Boxplot for Target Variable")
        fig = px.box(data_to_train, y=target_col, title=f"Boxplot of {target_col}")
        st.plotly_chart(fig)

        st.subheader("Correlation Heatmap")
        fig = px.imshow(data_to_train.corr(), text_auto=True, title="Correlation Heatmap")
        st.plotly_chart(fig)

        # Interactive Scatter Plot
        st.subheader("True vs Predicted Values")
        fig = px.scatter(x=y_test, y=predictions, title=f"True vs Predicted Values ({model_name})",
                         labels={"x": "True Values", "y": "Predicted Values"})
        fig.add_shape(
            type="line",
            line=dict(color="red", dash="dash"),
            x0=min(y_test),
            x1=max(y_test),
            y0=min(y_test),
            y1=max(y_test)
        )
        st.plotly_chart(fig)

        # Metrics
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        st.write(f"Model Evaluation for {model_name}:")
        st.write(f"MAE: {mae}")
        st.write(f"MSE: {mse}")
        st.write(f"RMSE: {rmse}")
        st.write(f"R^2 Score: {r2}")
