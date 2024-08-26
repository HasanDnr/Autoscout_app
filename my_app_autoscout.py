import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Load data
df = pd.read_csv("Ready_to_ML.csv")

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Visualization", "Prediction"])

# Home Page
if page == "Home":
    st.title("Welcome to the Car Price Prediction App")
    st.write("""
    This application helps you predict the price of a car based on various features such as age, mileage, engine size, and more. 
    You can also visualize how different features affect the price.
    """)
    st.image("car_image.jpeg", caption="Predict your car price with our app!",use_column_width=True )  

    st.subheader("How it Works")
    st.write("""
    - **Visualization**: Explore how different features impact car prices.
    - **Prediction**: Enter the details of a car to get a price prediction.
    """)

# Visualization Page
elif page == "Visualization":
    st.title("Feature Impact on Car Prices")

    st.write("Here you can see how various features like age, mileage, and engine size affect the car price.")
    
    # Price distribution
    st.subheader("Price Distribution")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price'], kde=True, bins=30)
    st.pyplot(plt)
    
    # Age vs. Price
    st.subheader("Price vs. Age")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='age', y='price', data=df)
    st.pyplot(plt)
    
    # Mileage vs. Price
    st.subheader("Price vs. Mileage")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='mileage', y='price', data=df)
    st.pyplot(plt)
    
    # Engine size vs. Price
    st.subheader("Price vs. Engine Size")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='engine_size', y='price', data=df)
    st.pyplot(plt)

# Prediction Page
elif page == "Prediction":
    st.title("Predict the Price of Your Car")

    # Load or train the model
    def load_model():
        try:
            with open("auto_scout_advanced_model.pkl", "rb") as file:
                model = pickle.load(file)
        except FileNotFoundError:
            model = train_model()
        return model

    def train_model():
        X = df[['age', 'mileage', 'engine_size', 'make_model', 'gearbox', 'drivetrain']]
        y = df['price']

        numeric_features = ['age', 'mileage', 'engine_size']
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

        categorical_features = ['make_model', 'gearbox', 'drivetrain']
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', xgb.XGBRegressor(objective='reg:squarederror'))])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        param_grid = {
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [3, 5, 7],
            'regressor__learning_rate': [0.01, 0.1, 0.3],
        }

        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_absolute_error')
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        with open("auto_scout_advanced_model.pkl", "wb") as file:
            pickle.dump(best_model, file)

        return best_model

    model = load_model()

    # User Input Section
    make_model = st.selectbox("Make-Model", df['make_model'].unique())
    age = st.number_input("Age:", min_value=0, max_value=30, value=5)
    mileage = st.number_input("Mileage:", min_value=0, max_value=300000, value=50000)
    engine_size = st.number_input("Engine size(in liters):", min_value=0.0, max_value=8.0, value=2.0)
    gearbox = st.selectbox("Gearbox", df['gearbox'].unique())
    drivetrain = st.selectbox("Drivetrain", df['drivetrain'].unique())

    if st.button("Predict Price"):
        input_data = pd.DataFrame({
            'age': [age],
            'mileage': [mileage],
            'engine_size': [engine_size],
            'make_model': [make_model],
            'gearbox': [gearbox],
            'drivetrain': [drivetrain]
        })

        prediction = model.predict(input_data)
        st.write(f"Predicted price of the car: €{prediction[0]:,.2f}")
