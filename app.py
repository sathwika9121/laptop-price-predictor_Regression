import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Page Configuration
st.set_page_config(
    page_title="Laptop Price Prediction",
    page_icon="💻",
    layout="centered"
)

# Title
st.title("💻 Laptop Price Prediction using Random Forest")
st.write("Predict laptop prices based on laptop specifications.")

# Load Dataset
data = pd.read_csv("laptop_price.csv", encoding="latin-1")

# Drop unnecessary columns
for col in ["laptop_ID", "Product"]:
    if col in data.columns:
        data.drop(col, axis=1, inplace=True)

# Features and Target
X = data.drop("Price_euros", axis=1)
y = data["Price_euros"]

# Convert categorical columns
X = pd.get_dummies(X, drop_first=True)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Sidebar Inputs
st.sidebar.header("Enter Laptop Specifications")

company = st.sidebar.selectbox(
    "Company",
    ["Dell", "HP", "Lenovo", "Asus", "Acer", "Apple"]
)

typename = st.sidebar.selectbox(
    "Laptop Type",
    ["Notebook", "Gaming", "Ultrabook", "2 in 1 Convertible"]
)

inches = st.sidebar.slider(
    "Screen Size (Inches)",
    10.0,
    20.0,
    15.6
)

ram = st.sidebar.selectbox(
    "RAM",
    ["4GB", "8GB", "16GB", "32GB"]
)

weight = st.sidebar.slider(
    "Weight (KG)",
    0.5,
    5.0,
    2.0
)

opsys = st.sidebar.selectbox(
    "Operating System",
    ["Windows 10", "macOS", "Linux", "No OS"]
)

# Create Input Data
input_data = pd.DataFrame({
    "Company": [company],
    "TypeName": [typename],
    "Inches": [inches],
    "Ram": [ram],
    "Weight": [str(weight) + "kg"],
    "OpSys": [opsys]
})

# Convert categorical columns
input_data = pd.get_dummies(input_data)

# Add missing columns
for col in X.columns:
    if col not in input_data.columns:
        input_data[col] = 0

# Arrange columns correctly
input_data = input_data[X.columns]

# Prediction
prediction = model.predict(input_data)

# Display Result
st.subheader("💰 Predicted Laptop Price")

st.success(f"Estimated Price: € {prediction[0]:,.2f}")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Random Forest Regression")