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
    data_cleaned = data.drop(columns=['license'])

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
