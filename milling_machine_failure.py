import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Load the dataset
dta = pd.read_csv('milling_dataset.csv')
def load_data():
    data = shuffle(dta)
    return data

data_load_state = st.text('Loading data...')
data = load_data()
data_load_state.text("Data loaded successfully!")

# Preprocessing
# Drop irrelevant columns like UDI, Product ID, and Type
data.drop(['UDI', 'Product ID', 'Type', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1, inplace=True)

# data = data[['Air temperature (K)', 'Process temperature (K)', 'Rotational speed (rpm)', 'Torque (Nm)', 'Tool wear (min)', 'Machine failure', 'HDF', 'OSF', 'PWF', 'RNF', 'TWF']]

# Sidebar for user input
st.sidebar.header('Adjust Input Values')

# Input fields for adjusting values
air_temp = st.sidebar.slider('Air Temperature (K)', 295.0, 305.0, 300.0, 1.0)
process_temp = st.sidebar.slider('Process Temperature (K)', 306.0, 314.0, 310.0, 1.0)
rotational_speed = st.sidebar.slider('Rotational Speed (rpm)', 1168, 2886, 2000, 10)
torque = st.sidebar.slider('Torque (Nm)', 3.8, 76.6, 40.0, 0.1)
tool_wear = st.sidebar.slider('Tool Wear (min)', 0, 253, 120, 1)

# Prepare user input for prediction
user_input = pd.DataFrame({
    'Air temperature (K)': [air_temp],
    'Process temperature (K)': [process_temp],
    'Rotational speed (rpm)': [rotational_speed],
    'Torque (Nm)': [torque],
    'Tool wear (min)': [tool_wear]
})

# Model Training
X = data.drop('Machine failure', axis=1)
y = data['Machine failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
prediction = model.predict(user_input)
prediction_proba = model.predict_proba(user_input)

# Display prediction result
st.header('Predictive Maintenance Result')
if prediction[0] == 1:
    st.error('Machine failure is predicted!')
else:
    st.success('No machine failure predicted.')

st.subheader('Prediction Probability')
st.write('Probability of machine failure:', prediction_proba[0][1])
