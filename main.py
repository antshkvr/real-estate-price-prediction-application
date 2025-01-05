from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import plotly.express as px
import json
import requests
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from catboost import CatBoostRegressor


# Декомпозиція часу (Trend, Seasonal, Residual)
def decompose_time_series(data, column, target_column):
    from statsmodels.tsa.seasonal import seasonal_decompose

    # Перетворення в часову серію, агреговану за середніми значеннями
    ts = data.set_index(column)[target_column]
    decomposition = seasonal_decompose(ts, model='additive', period=365)
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

    # Keep original county names for mapping
    county_column = None
    if 'county' in categorical_columns:
        county_column = data['county'].copy()

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
            new_data = data.dropna()
    else:
        new_data = data.copy()
        # Check for null values
    data_to_train = new_data.copy()
    data_to_train = data_to_train.drop(columns="date")

    st.write("Null values in each column:")
    st.write(new_data.isnull().sum())
    st.write(f"Shape of dataset before handling NULLs: {data.shape}")
    st.write(f"Shape of dataset after handling NULLs: {new_data.shape}")

    st.write("Updated Dataset Preview:")
    st.dataframe(new_data.head())

    # Distribution Chart Builder
    st.subheader("Distribution Charts")
    chart_type = st.selectbox("Select chart type", ["Histogram", "Box Plot", "County Map"])

    # Show column selector only for non-map visualizations
    if chart_type != "County Map":
        selected_column = st.selectbox("Select column to visualize", data.columns)
    else:
        selected_column = st.selectbox("Select value to show on map",
                                       [col for col in data.columns if col not in ['county', 'county_original']])

    # Outlier filtering (only for non-map visualizations)
    if chart_type != "County Map":
        col1, col2 = st.columns(2)
        with col1:
            filter_outliers = st.checkbox("Filter outliers")
        with col2:
            if filter_outliers:
                outlier_method = st.selectbox("Outlier detection method", ["Z-Score", "IQR"])
                threshold = st.slider("Outlier threshold (%)", 0, 20, 5)

                filtered_data = data.copy()
                if outlier_method == "Z-Score":
                    z_scores = np.abs(
                        (filtered_data[selected_column] - filtered_data[selected_column].mean()) / filtered_data[
                            selected_column].std())
                    filtered_data = filtered_data[z_scores <= np.percentile(z_scores, 100 - threshold)]
                elif outlier_method == "IQR":
                    Q1 = filtered_data[selected_column].quantile(threshold / 200)
                    Q3 = filtered_data[selected_column].quantile(1 - threshold / 200)
                    filtered_data = filtered_data[
                        (filtered_data[selected_column] >= Q1) & (filtered_data[selected_column] <= Q3)]

                st.write(
                    f"Removed {len(data) - len(filtered_data)} outliers ({(len(data) - len(filtered_data)) / len(data) * 100:.1f}% of data)")
            else:
                filtered_data = data
    else:
        filtered_data = data

    if chart_type == "County Map":
        if 'county_original' not in data.columns:
            # Restore county names for mapping
            if county_column is not None:
                data['county_original'] = county_column
            # st.error("No 'county' column found in the dataset")
            # st.stop()

        # Load Poland GeoJSON (counties)
        try:
            geojson_url = "https://raw.githubusercontent.com/ppatrzyk/polska-geojson/refs/heads/master/powiaty/powiaty-min.geojson"
            response = requests.get(geojson_url)
            response.raise_for_status()
            counties_geojson = json.loads(response.text)

            # Prepare data for the map using original county names
            county_stats = filtered_data.groupby('county_original')[selected_column].agg(
                ['mean', 'count']).reset_index()
            county_stats['county'] = county_stats['county_original'].apply(
                lambda x: f"powiat {x}" if not x.startswith("powiat ") else x)

            # Create choropleth map
            fig = px.choropleth_mapbox(
                county_stats,
                geojson=counties_geojson,
                locations='county',
                featureidkey="properties.nazwa",
                color='mean',
                hover_data=['count'],
                color_continuous_scale="RdYlBu_r",
                range_color=[county_stats['mean'].min(), county_stats['mean'].max()],
                mapbox_style="carto-positron",
                zoom=5,
                center={"lat": 52.0, "lon": 19.0},
                opacity=0.7,
                title=f"Distribution of {selected_column} across Polish Counties<br><sup>Hover for details</sup>"
            )
            fig.update_layout(
                margin={"r": 0, "t": 30, "l": 0, "b": 0},
                height=600,
                coloraxis_colorbar=dict(
                    title=dict(text=f"Average {selected_column}"),
                    len=0.8,
                )
            )

            # Add summary statistics
            st.write("Summary Statistics:")
            stats_df = pd.DataFrame({
                'Metric': ['Minimum', 'Maximum', 'Mean', 'Median', 'Counties with data'],
                'Value': [
                    f"{county_stats['mean'].min():.2f}",
                    f"{county_stats['mean'].max():.2f}",
                    f"{county_stats['mean'].mean():.2f}",
                    f"{county_stats['mean'].median():.2f}",
                    f"{len(county_stats)}"
                ]
            })
            st.dataframe(stats_df)

        except Exception as e:
            st.error(f"Error loading map data: {str(e)}")
            st.stop()

    elif chart_type == "Histogram":
        bins = st.slider("Number of bins", 5, 100, 30)
        fig = px.histogram(filtered_data, x=selected_column, nbins=bins, title=f"Histogram of {selected_column}")
    elif chart_type == "Box Plot":
        fig = px.box(filtered_data, y=selected_column, title=f"Box Plot of {selected_column}")
    else:
        st.error("Invalid chart type")
        st.stop()

    st.plotly_chart(fig)

    # Select target variable
    target_col = st.selectbox("Select the target column", data_to_train.columns)
    feature_cols = [col for col in data_to_train.columns if col != target_col]
    X = data_to_train[feature_cols]
    y = data_to_train[target_col]

    # Вибираємо колонки, які є числовими та мають часову залежність (наприклад, week_no або дата)
    time_columns = [col for col in new_data.columns if 'date' in col.lower() and target_col != 'week_no']

    # Обрання часової колонки для декомпозиції
    if time_columns:
        feature_col = st.selectbox("Select a feature column for decomposition", time_columns, key='decomposition')

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
            daterange = pd.date_range(train.index[0], periods=7)
            model = ARIMA(train[target_col], order=(p, d, q))
            model_fit = model.fit()
            st.write(model_fit.summary())
            forecast_series = model_fit.predict(start=len(train) + 1, end=len(train) + len(test))
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
            plt.plot(train.index, train[target_col].values, label='Train')
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
                              index=list(data.columns).index(
                                  "median_price_pln") if "median_price_pln" in data.columns else 0)
    # Exclude date and county_original from features
    feature_cols = [col for col in new_data.columns if col != target_col
                    and col != 'county_original'
                    and col != 'date']
    X = new_data[feature_cols]
    y = new_data[target_col]

    # Train-test split
    test_size = st.slider("Test size (%)", 10, 80, 20, key='2')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42)

    # Model selection
    model_name = st.selectbox("Select a model",
                              ["LightGBM", "XGBoost", "Random Forest", "CatBoost",
                               "Ridge Regression", "Lasso Regression", "SVR"])

    if st.button("Train Model"):
        if model_name == "LightGBM":
            model = lgb.LGBMRegressor(random_state=42)
        elif model_name == "XGBoost":
            model = xgb.XGBRegressor(random_state=42, use_label_encoder=False, eval_metric='rmse')
        elif model_name == "Random Forest":
            model = RandomForestRegressor(random_state=42, n_jobs=-1)
        elif model_name == "CatBoost":
            model = CatBoostRegressor(random_state=42, verbose=False)
        elif model_name == "Ridge Regression":
            model = Ridge(random_state=42)
        elif model_name == "Lasso Regression":
            model = Lasso(random_state=42)
        elif model_name == "SVR":
            model = SVR(kernel='rbf', max_iter=1000)
        else:
            st.error("Invalid model selection")
            st.stop()

        # Train the model
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Additional Analysis with Plotly
        st.subheader("Target Variable Distribution")
        fig = px.histogram(data_to_train, x=target_col, title=f"Distribution of {target_col}")
        st.plotly_chart(fig, key="dist_1")

        st.subheader("Boxplot for Target Variable")
        fig = px.box(data_to_train, y=target_col, title=f"Boxplot of {target_col}")
        st.plotly_chart(fig, key="box_1")

        st.subheader("Correlation Heatmap")
        numeric_cols = data_to_train.select_dtypes(include=['int64', 'float64']).columns
        correlation_matrix = data_to_train[numeric_cols].corr()
        fig = px.imshow(correlation_matrix, text_auto=True, title="Correlation Heatmap")
        st.plotly_chart(fig, key="corr_1")

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
        st.plotly_chart(fig, key="scatter_1")

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

        # Feature Importance Analysis
        if hasattr(model, 'feature_importances_'):
            st.subheader("Feature Importance Analysis")
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)

            fig = px.bar(importance_df, x='Feature', y='Importance',
                         title='Feature Importance',
                         labels={'Importance': 'Importance Score', 'Feature': 'Feature Name'})
            st.plotly_chart(fig, key="importance_1")
