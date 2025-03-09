import streamlit as st

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
import tempfile
from tensorflow.keras.models import load_model


# ✅ Load Crime Dataset
df = pd.read_csv(r"D:\shruthi-work\final year project\Major_Crime_Indicators_Open_Data_-3805566126367379926.csv")

# ✅ Load Trained Model
crime_model = joblib.load("crime_prediction_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# ✅ Streamlit UI
st.title("Crime Prediction & Analysis Dashboard")

# 🔹 Show Raw Data
if st.sidebar.checkbox("Show Raw Data"):
    st.subheader("Raw Crime Data")
    st.write(df.head())

# 🔹 Crime Count by Year
st.subheader("Crime Count by Year")
crime_by_year = df["OCC_YEAR"].value_counts().sort_index()
st.line_chart(crime_by_year)

# 🔹 Heatmap of Crimes
# 🔹 Crime Heatmap
st.subheader("Crime Heatmap")

# Select only numeric columns
numeric_df = df.select_dtypes(include=["number"])

plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
st.pyplot(plt)

st.subheader("Crime Type Distribution")

# Aggregate crime types
crime_counts = df["MCI_CATEGORY"].value_counts()

# Plot bar chart
plt.figure(figsize=(10, 5))
sns.barplot(x=crime_counts.index, y=crime_counts.values, palette="viridis")
plt.xticks(rotation=45)
plt.xlabel("Crime Type")
plt.ylabel("Count")
plt.title("Crime Distribution by Category")
st.pyplot(plt)  # Show bar chart in Streamlit


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"D:\shruthi-work\final year project\Major_Crime_Indicators_Open_Data_-3805566126367379926.csv")

# Ensure required columns exist
required_columns = ['OCC_HOUR', 'MCI_CATEGORY']
if not all(col in df.columns for col in required_columns):
    st.error("❌ Required columns are missing in the dataset!")
    st.stop()

# Count number of crimes per hour and category
hour_crime_group = df.groupby(['OCC_HOUR', 'MCI_CATEGORY']).size().reset_index(name='Crime_Count')

# Streamlit UI
st.title("📊 Crime Data Visualization")
st.subheader("⏰ Crime Types by Hour of Day in Toronto")

# Initialize the plot
fig, ax = plt.subplots(figsize=(15, 10))

# Iterate through each crime type and plot separately
for crime_type, group_data in hour_crime_group.groupby('MCI_CATEGORY'):
    ax.plot(group_data['OCC_HOUR'], group_data['Crime_Count'], label=crime_type, linewidth=2)

# Labels and title
ax.set_xlabel('Hour')
ax.set_ylabel('Number of Occurrences')
ax.set_title('Crime Types by Hour of Day in Toronto', color='red', fontsize=18)

# Add legend to show crime types
ax.legend(title='Crime Types')

# Display the plot in Streamlit
st.pyplot(fig)




# Sample dataset (Replace with your actual DataFrame)
df_2024_grouped = pd.DataFrame({
    "MCI_CATEGORY": ["Assault", "Auto Theft", "Break and Enter", "Robbery", "Theft Over"],
    "Number of Cases": [15800, 7200, 5000, 2300, 1400]
}).set_index("MCI_CATEGORY")

# Display title in Streamlit
st.title("Major Crimes Reported in Toronto (2024)")

# Plot the bar chart
fig, ax = plt.subplots(figsize=(12, 6))
df_2024_grouped.plot(kind="barh", legend=True, ax=ax, title="Number of Major Crimes Reported in Toronto in 2024")

# Show the plot in Streamlit
st.pyplot(fig)



st.subheader("Anomaly Detection Results")

# Path to extracted frame (Update with actual path)
image_path = "extracted_frames/stealing_uccrime_Robbery099_x264_frame_17.jpg"

# Display Image in Streamlit
st.image(image_path, caption="Detected Anomaly", use_column_width=True)

# Aggregate Data by Month & Year
mci_distribution = df.groupby(['OCC_YEAR', 'OCC_MONTH'], as_index=False).size()

# Convert Year to String
mci_distribution['OCC_YEAR'] = mci_distribution['OCC_YEAR'].astype(str)

# Create 'Month-Year' Column
mci_distribution['monthYear'] = mci_distribution['OCC_MONTH'] + ', ' + mci_distribution['OCC_YEAR']

# Plot the Time Series Distribution
st.subheader("📊 Time Series Distribution of Crime (Month-wise)")

plt.figure(figsize=(20, 6))
plt.grid(True)
plt.plot(mci_distribution['monthYear'], mci_distribution['size'], marker='o')

# Labels
plt.xlabel('Month Stream')
plt.ylabel('Count of Crime')
plt.title('Time Series Distribution of Crime [Month-wise]')

# Set X-Ticks (for better visibility)
plt.xticks(np.arange(0, mci_distribution.shape[0], 3), rotation=90)

# Display the Plot in Streamlit
st.pyplot(plt)


import streamlit as st
import cv2
import numpy as np
import tempfile
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load pre-trained Autoencoder model
autoencoder = load_model("autoencoder_model.h5", compile=False)

# Function to preprocess video frames
def preprocess_frame(frame):
    frame = cv2.resize(frame, (128, 128))  # Resize to match model input
    frame = frame.astype("float32") / 255.0  # Normalize
    return frame

# Function to enhance frame clarity
def enhance_frame(frame):
    """Enhance frame clarity using contrast, sharpening, and noise reduction."""
    
    # Convert to grayscale for better enhancement
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Histogram Equalization for contrast enhancement
    enhanced = cv2.equalizeHist(gray)

    # Convert back to BGR for visualization
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    # Apply Unsharp Masking for sharpening
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)

    return sharpened

# Function to detect anomalies in video frames
def detect_anomaly(frame, threshold=0.05):
    processed_frame = preprocess_frame(frame)
    reconstructed = autoencoder.predict(np.expand_dims(processed_frame, axis=0))
    
    error = np.mean(np.abs(reconstructed - processed_frame))
    
    return error > threshold  # Returns True if anomaly detected

# Streamlit UI
st.title("📹 Video Anomaly Detection")

uploaded_file = st.file_uploader("📤 Upload a Video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    anomaly_detected = False
    anomaly_frame = None  # Store the anomaly frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if detect_anomaly(frame):
            anomaly_detected = True
            anomaly_frame = frame.copy()  # Store the frame where anomaly is detected
            break  # Stop at the first detected anomaly

    cap.release()

    if anomaly_detected:
        st.error("🚨 Anomaly Detected in the Video!")
        
        # Enhance and display the anomaly frame
        if anomaly_frame is not None:
            enhanced_frame = enhance_frame(anomaly_frame)
            st.image(enhanced_frame, channels="BGR", use_container_width=True, caption="🔍 Detected Anomaly Frame")
    
    else:
        st.success("✅ No Anomaly Detected!")


# 🔹 Crime Prediction
st.subheader("Predict Crime Type")
occ_year = st.number_input("Occurrence Year", min_value=2000, max_value=2030, value=2024)
occ_month = st.selectbox("Occurrence Month", df["OCC_MONTH"].unique())
occ_hour = st.slider("Occurrence Hour", 0, 23, 12)
division = st.selectbox("Division", df["DIVISION"].unique())
location_type = st.selectbox("Location Type", df["LOCATION_TYPE"].unique())
premises_type = st.selectbox("Premises Type", df["PREMISES_TYPE"].unique())

# Convert Inputs using Label Encoding
input_data = pd.DataFrame([[occ_year, occ_month, occ_hour, division, location_type, premises_type]],
                          columns=["OCC_YEAR", "OCC_MONTH", "OCC_HOUR", "DIVISION", "LOCATION_TYPE", "PREMISES_TYPE"])
for col in ["OCC_MONTH", "DIVISION", "LOCATION_TYPE", "PREMISES_TYPE"]:
    input_data[col] = label_encoders[col].transform(input_data[col])

# Predict Crime Type
if st.button("Predict Crime"):
    prediction = crime_model.predict(input_data)[0]
    predicted_crime = label_encoders["MCI_CATEGORY"].inverse_transform([prediction])[0]
    st.success(f"Predicted Crime: **{predicted_crime}**")
    
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ✅ Function to Load Data with Automatic Processing
@st.cache_data
def load_data(filepath):
    df = pd.read_csv(r"C:\Users\nirur\Downloads\Major_Crime_Indicators_Open_Data_-3805566126367379926.csv")

    # Set date column
    df["REPORT_DATE"] = pd.to_datetime(df["REPORT_DATE"], errors="coerce")
    df = df.set_index("REPORT_DATE").sort_index()

    # ✅ Create a 'Crime Count' column (count crimes per day)
    df["Crime Count"] = 1
    df = df.resample("D")["Crime Count"].sum()

    return df

# 📌 Load Data (Update the file path)
csv_filepath = r"C:\Users\nirur\Downloads\Major_Crime_Indicators_Open_Data_-3805566126367379926.csv"
df = load_data(csv_filepath)

# ✅ Select 'Crime Count' as our target variable
y = df

# ✅ Get the last date from the dataset
last_date = y.index.max()

# ✅ Generate future dates for forecasting
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=365, freq='D')

# ✅ Fit SARIMAX model
st.write("⏳ Training SARIMAX model... Please wait.")
model = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,12))
SARIMAXresults = model.fit()

# ✅ Get predictions
pred_uc = SARIMAXresults.get_forecast(steps=365)
pred_ci = pred_uc.conf_int()

# ✅ Assign correct date index to predictions
pred_uc_mean = pd.Series(pred_uc.predicted_mean.values, index=forecast_dates)
pred_ci.index = forecast_dates

# ✅ Create an interactive Plotly figure
fig = go.Figure()

# Observed Data
fig.add_trace(go.Scatter(
    x=y.index, y=y.values, 
    mode='lines', name='Observed', line=dict(color='blue')
))

# Forecasted Data
fig.add_trace(go.Scatter(
    x=pred_uc_mean.index, y=pred_uc_mean.values,
    mode='lines', name='Forecast', line=dict(color='red')
))

# Confidence Intervals
fig.add_trace(go.Scatter(
    x=pred_ci.index, y=pred_ci.iloc[:, 0],
    mode='lines', name='Lower Bound', line=dict(color='gray'), opacity=0.5
))
fig.add_trace(go.Scatter(
    x=pred_ci.index, y=pred_ci.iloc[:, 1],
    mode='lines', name='Upper Bound', line=dict(color='gray'), opacity=0.5
))

# Fill confidence interval
fig.add_trace(go.Scatter(
    x=list(pred_ci.index) + list(pred_ci.index[::-1]),
    y=list(pred_ci.iloc[:, 1]) + list(pred_ci.iloc[:, 0][::-1]),
    fill='toself', fillcolor='rgba(200,200,200,0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    showlegend=False
))

# ✅ Customize layout
fig.update_layout(
    title="📈 Crime Count Forecast with SARIMAX",
    xaxis_title="Date",
    yaxis_title="Crime Count",
    hovermode="x"
)

# ✅ Streamlit App UI
st.title("📊 Crime Forecasting Dashboard")
st.plotly_chart(fig)


import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ✅ Load Data Function
@st.cache_data
def load_data(filepath):
    df = pd.read_csv(r"D:\shruthi-work\final year project\Major_Crime_Indicators_Open_Data_-3805566126367379926.csv"")

    # ✅ Convert REPORT_DATE to DateTime
    df["REPORT_DATE"] = pd.to_datetime(df["REPORT_DATE"], errors="coerce")

    # ✅ Check if 'City' column exists, else handle alternative
    city_column = "City" if "City" in df.columns else "NEIGHBOURHOOD_158"

    # ✅ Count crimes per city per date
    df["Crime Count"] = 1
    df = df.groupby([city_column, "REPORT_DATE"]).size().reset_index(name="Crime Count")

    return df, city_column

# 📌 Load Data
csv_filepath = r"D:\shruthi-work\final year project\Major_Crime_Indicators_Open_Data_-3805566126367379926.csv"
df, city_column = load_data(csv_filepath)

# ✅ Streamlit App UI
st.title("📊 City-Based Crime Forecasting")

# 📍 User selects a city
selected_city = st.selectbox("Select a City", df[city_column].unique())

# ✅ Filter data for the selected city
city_data = df[df[city_column] == selected_city].copy()

# ✅ Set date column as index
city_data.set_index("REPORT_DATE", inplace=True)

# ✅ Aggregate by month (avoid summing non-numeric columns)
city_data = city_data.resample('M').sum(numeric_only=True)

# ✅ Ensure 'Crime Count' exists
if "Crime Count" not in city_data:
    st.error("No 'Crime Count' column found in dataset.")
    st.stop()

y = city_data["Crime Count"]

# ✅ Get last date for forecasting
last_date = y.index.max()

# ✅ Generate future dates
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=12, freq='M')

# ✅ Train SARIMAX model
st.write(f"⏳ Training SARIMAX model for {selected_city}... Please wait.")
model = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,12))
SARIMAXresults = model.fit()

# ✅ Get predictions
pred_uc = SARIMAXresults.get_forecast(steps=12)
pred_ci = pred_uc.conf_int()

# ✅ Assign date index to predictions
pred_uc_mean = pd.Series(pred_uc.predicted_mean.values, index=forecast_dates)
pred_ci.index = forecast_dates

# ✅ Create an interactive Plotly figure
fig = go.Figure()

# 📈 Observed Data
fig.add_trace(go.Scatter(
    x=y.index, y=y.values, 
    mode='lines', name='Observed', line=dict(color='blue')
))

# 🔴 Forecasted Data
fig.add_trace(go.Scatter(
    x=pred_uc_mean.index, y=pred_uc_mean.values,
    mode='lines', name='Forecast', line=dict(color='red')
))

# ⚠️ Confidence Intervals
fig.add_trace(go.Scatter(
    x=pred_ci.index, y=pred_ci.iloc[:, 0],
    mode='lines', name='Lower Bound', line=dict(color='gray'), opacity=0.5
))
fig.add_trace(go.Scatter(
    x=pred_ci.index, y=pred_ci.iloc[:, 1],
    mode='lines', name='Upper Bound', line=dict(color='gray'), opacity=0.5
))

# 🔳 Fill confidence interval
fig.add_trace(go.Scatter(
    x=list(pred_ci.index) + list(pred_ci.index[::-1]),
    y=list(pred_ci.iloc[:, 1]) + list(pred_ci.iloc[:, 0][::-1]),
    fill='toself', fillcolor='rgba(200,200,200,0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    showlegend=False
))

# ✅ Customize layout
fig.update_layout(
    title=f"Crime Forecast for {selected_city}",
    xaxis_title="Date",
    yaxis_title="Crime Count",
    hovermode="x"
)

# ✅ Display the chart in Streamlit
st.plotly_chart(fig)



st.sidebar.markdown("Built using Streamlit")

