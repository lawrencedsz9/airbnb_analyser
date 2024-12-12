import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from streamlit_folium import st_folium
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.dates as mdates

# Title
st.title("Airbnb Listings Analysis")

# File Upload
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Data Cleaning
    data['last_review'] = pd.to_datetime(data['last_review'], errors='coerce')
    data['price'].fillna(data['price'].median(), inplace=True)
    data['reviews_per_month'].fillna(0, inplace=True)
    data_cleaned = data.drop(columns=['license'], errors='ignore')

    # EDA
    st.subheader("Exploratory Data Analysis")
    st.write(data_cleaned.describe())

    # Price Distribution Plot
    st.subheader("Price Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data_cleaned['price'], bins=50, kde=True, color='blue', ax=ax)
    st.pyplot(fig)

    # Map
    st.subheader("Listings Map")
    map_center = [data_cleaned['latitude'].mean(), data_cleaned['longitude'].mean()]
    map_airbnb = folium.Map(location=map_center, zoom_start=12)
    marker_cluster = MarkerCluster().add_to(map_airbnb)
    for idx, row in data_cleaned.iterrows():
        folium.Marker(location=[row['latitude'], row['longitude']],
                      popup=f"Price: ${row['price']}\nRoom Type: {row['room_type']}\nNeighborhood: {row['neighbourhood']}").add_to(marker_cluster)
    st_folium(map_airbnb, width=800, height=500)

    # Model Training and Evaluation
    st.subheader("Predictive Modeling")
    features = ['minimum_nights', 'number_of_reviews', 'reviews_per_month', 'availability_365']
    X = data_cleaned[features]
    y = data_cleaned['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"**Mean Absolute Error (MAE):** {mae}")
    st.write(f"**Mean Squared Error (MSE):** {mse}")
    st.write(f"**R-squared (R²):** {r2}")

    # Feature Importance
    st.subheader("Feature Importance")
    feature_importances = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feature_importances, x='Importance', y='Feature', palette='viridis', ax=ax)
    st.pyplot(fig)

    # Export Results
    if st.button("Export Results"):
        feature_importances.to_csv('feature_importances.csv', index=False)
        st.write("Feature importance exported to 'feature_importances.csv'")

    # Time-Series Analysis
    if 'last_review' in data_cleaned.columns:
        st.subheader("Time-Series Analysis: Price Trends")

        # Filter and Resample Data for Time-Series Analysis
        time_series_data = data_cleaned[['last_review', 'price']].dropna()
        time_series_data = time_series_data.sort_values(by='last_review')
        time_series_data = time_series_data.set_index('last_review')
        time_series_data = time_series_data.resample('M').mean()
        time_series_data['price'] = time_series_data['price'].interpolate(method='linear')  # Handle missing values

        # Plot Time-Series Data
        st.write("**Monthly Average Prices**")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(time_series_data.index, time_series_data['price'], marker='o', linestyle='-', label='Price')
        ax.set_title('Monthly Average Price Trends', fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Seasonal Decomposition
        st.write("**Seasonal Decomposition**")
        decomposition = seasonal_decompose(time_series_data['price'], model='additive', period=12)
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        decomposition.trend.plot(ax=axes[0], title='Trend', color='blue')
        decomposition.seasonal.plot(ax=axes[1], title='Seasonal', color='green')
        decomposition.resid.plot(ax=axes[2], title='Residual', color='red')
        plt.xlabel('Date')
        st.pyplot(fig)