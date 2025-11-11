import streamlit as st
import numpy as np
import pandas as pd
import joblib
import requests
import os
import tensorflow as tf
from train import train_irrigation_model
import shutil
from datetime import datetime
import plotly.express as px

# --- CONFIGURATION & PAGE SETUP ---
st.set_page_config(page_title="Smart Irrigation System", layout="wide", initial_sidebar_state="expanded")
HISTORY_FILE = 'prediction_history.csv'
API_KEY = "3a82b8373a4be272a226c6175f8492a8"
CITIES_FILE = 'indian_cities.csv'
TFLITE_MODEL_PATH = 'irrigation_model.tflite'
SCALER_PATH = 'scaler.pkl'

# --- NEW: Function to load and inject CSS ---
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def apply_dark_mode_styles():
    dark_mode_css = """
    <style>
        body {
            --primary-color: #0068c9; --background-color: #f0f2f6;
            --secondary-background-color: #ffffff; --text-color: #262730;
        }
        body.dark-mode {
            --primary-color: #1c83e1; --background-color: #0e1117;
            --secondary-background-color: #1e1e1e; --text-color: #fafafa;
        }
    </style>
    """
    st.markdown(dark_mode_css, unsafe_allow_html=True)
    dark_mode_js = f"""
    <script>
        const body = window.parent.document.querySelector('body');
        if ({st.session_state.dark_mode}) {{
            body.classList.add('dark-mode');
        }} else {{
            body.classList.remove('dark-mode');
        }}
    </script>
    """
    st.components.v1.html(dark_mode_js, height=0)

# --- HELPER FUNCTIONS ---
@st.cache_data
def load_city_list():
    try:
        df = pd.read_csv(CITIES_FILE, usecols=[0], header=None)
        return sorted(list(df[0].unique()))
    except FileNotFoundError:
        return ["Patna", "Gaya", "Darbhanga"]

@st.cache_resource
def load_interpreter_and_scaler(model_path, scaler_path):
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path); interpreter.allocate_tensors()
        scaler = joblib.load(scaler_path)
        return interpreter, scaler
    except: return None, None

def predict_with_tflite(interpreter, input_data):
    input_details = interpreter.get_input_details(); output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], np.array(input_data, dtype=np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0][0]

def get_weather_data(city, api_key):
    base_url = "http://api.openweathermap.org/data/2.5/weather"; params = {'q': city, 'appid': api_key, 'units': 'metric'}
    try:
        response = requests.get(base_url, params=params); response.raise_for_status(); data = response.json()
        return data['main']['temp'], data['main']['humidity'], None
    except Exception as e: return None, None, f"City not found or API key invalid. Error: {e}"

def log_prediction(inputs, prediction):
    log_entry = {'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")], 'Soil Moisture': [inputs[0]], 'Temperature': [inputs[1]], 'Humidity': [inputs[2]], 'Crop Type': [inputs[3]], 'Predicted Duration (min)': [f"{prediction:.2f}"]}
    log_df = pd.DataFrame(log_entry)
    if not os.path.exists(HISTORY_FILE): log_df.to_csv(HISTORY_FILE, index=False)
    else: log_df.to_csv(HISTORY_FILE, mode='a', header=False, index=False)

# --- UI PAGE: PREDICTOR ---
def show_predictor_page():
    st.title('🌾 Smart Agricultural Irrigation Predictor')
    col1, col2 = st.columns([1, 1.5])
    with col1:
        with st.container(border=True):
            st.header("Input Parameters")
            temp_default = st.session_state.get('live_temp', 30.0)
            hum_default = st.session_state.get('live_hum', 60.0)
            soil_moisture = st.slider('Soil Moisture (%)', 0, 100, 35)
            temperature = st.slider('Temperature (°C)', 10, 45, int(temp_default))
            humidity = st.slider('Air Humidity (%)', 20, 90, int(hum_default))
            crop_options = {0: 'Wheat', 1: 'Corn', 2: 'Rice'}
            crop_type_code = st.selectbox('Crop Type', options=list(crop_options.keys()), format_func=lambda x: crop_options[x])
            if st.button('Predict Irrigation', use_container_width=True):
                with st.spinner("Loading model for the first time..."):
                    interpreter, scaler = load_interpreter_and_scaler(TFLITE_MODEL_PATH, SCALER_PATH)
                if interpreter and scaler:
                    input_data = np.array([[soil_moisture, temperature, humidity, crop_type_code]])
                    input_scaled = scaler.transform(input_data)
                    duration = predict_with_tflite(interpreter, input_scaled)
                    st.session_state.last_prediction = {"duration": duration, "inputs": [soil_moisture, temperature, humidity, crop_options[crop_type_code]]}
                    log_prediction([soil_moisture, temperature, humidity, crop_options[crop_type_code]], duration)

    with col2:
        st.header("Prediction Results")
        if 'last_prediction' in st.session_state:
            res, duration = st.session_state.last_prediction, st.session_state.last_prediction['duration']
            st.markdown(f'<div class="card"><div class="metric-label">Predicted Irrigation Duration</div><div class="metric-value">{duration:.2f} minutes</div></div>', unsafe_allow_html=True)
            with st.container(border=True):
                if duration < 15: st.success("Recommendation: Low irrigation needed.")
                elif duration < 30: st.info("Recommendation: Moderate irrigation recommended.")
                elif duration < 45: st.warning("Recommendation: High irrigation needed.")
                else: st.error("Recommendation: Very high irrigation needed!")
            st.subheader("Factors Influencing Prediction")
            chart_data = pd.DataFrame({"Value": res['inputs'][:3]}, index=["Soil Moisture", "Temperature", "Humidity"])
            st.bar_chart(chart_data)

# --- UI PAGE: DASHBOARD ---
def show_dashboard_page():
    st.title("📊 Interactive Analytics Dashboard")
    st.write("Analyze historical prediction data to uncover trends and insights.")
    if not os.path.exists(HISTORY_FILE):
        st.warning("No prediction history found. Make some predictions first on the Predictor page."); return
    history_df = pd.read_csv(HISTORY_FILE)
    if 'Timestamp' not in history_df.columns:
        st.error("Your prediction history file is in an old format. Please delete 'prediction_history.csv' and make a new prediction."); return
    history_df['Timestamp'] = pd.to_datetime(history_df['Timestamp'])
    st.write("---"); st.subheader("Prediction History (Last 20 Entries)"); st.dataframe(history_df.tail(20))
    st.write("---"); st.subheader("Irrigation Duration Over Time")
    fig1 = px.line(history_df, x='Timestamp', y='Predicted Duration (min)', title='Trends in Irrigation Needs', markers=True); st.plotly_chart(fig1, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Average Irrigation by Crop Type")
        avg_duration_by_crop = history_df.groupby('Crop Type')['Predicted Duration (min)'].mean().reset_index()
        fig2 = px.bar(avg_duration_by_crop, x='Crop Type', y='Predicted Duration (min)', title='Average Water Needs per Crop', color='Crop Type'); st.plotly_chart(fig2, use_container_width=True)
    with col2:
        st.subheader("Temperature vs. Irrigation Duration")
        fig3 = px.scatter(history_df, x='Temperature', y='Predicted Duration (min)', title='Impact of Temperature', color='Crop Type', trendline="ols"); st.plotly_chart(fig3, use_container_width=True)

# --- UI PAGE: RETRAIN MODEL ---
def show_retrain_page():
    st.title("🔄 One-Click Model Retraining")
    st.write("Improve the model's accuracy by training it on new data.")
    st.info("Your uploaded CSV file must have the columns: `soil_moisture`, `temperature`, `humidity`, `crop_type`, `irrigation_duration_minutes`.")
    uploaded_file = st.file_uploader("Upload your new training data (CSV)", type="csv")
    if uploaded_file is not None:
        new_data_df = pd.read_csv(uploaded_file)
        st.write("Data Preview:"); st.dataframe(new_data_df.head())
        epochs = st.slider("Select number of training epochs", 10, 200, 50)
        if st.button("Start Retraining"):
            with st.spinner("Retraining in progress... This may take a moment."):
                temp_data_path = "temp_training_data.csv"; new_data_df.to_csv(temp_data_path, index=False)
                new_model_path = "temp_model.keras"; new_scaler_path = "temp_scaler.pkl"
                performance, history = train_irrigation_model(dataset_path=temp_data_path, model_output_path=new_model_path, scaler_output_path=new_scaler_path, epochs=epochs)
                st.session_state.retrain_results = {"performance": performance, "history": history}
                os.remove(temp_data_path)
    if 'retrain_results' in st.session_state:
        st.success("Retraining complete!")
        results = st.session_state.retrain_results
        st.subheader("New Model Performance"); st.json(results['performance'])
        st.subheader("Training & Validation Loss")
        fig = px.line(x=range(1, len(results['history'].history['loss']) + 1), y=[results['history'].history['loss'], results['history'].history['val_loss']]); fig.update_layout(xaxis_title="Epoch", yaxis_title="Loss", legend_title="Metric"); fig.data[0].name = "Training Loss"; fig.data[1].name = "Validation Loss"
        st.plotly_chart(fig, use_container_width=True)
        st.warning("The new model is trained but NOT YET ACTIVE. Click the button below to replace the current active model.")
        if st.button("Activate New Model"):
            # We need to also convert the new keras model to tflite for speed
            from model import HybridSwishReLU
            import tensorflow as tf
            new_keras_model = tf.keras.models.load_model("temp_model.keras", custom_objects={'HybridSwishReLU': HybridSwishReLU})
            converter = tf.lite.TFLiteConverter.from_keras_model(new_keras_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model_content = converter.convert()
            with open("temp_model.tflite", 'wb') as f:
                f.write(tflite_model_content)

            shutil.move("temp_model.tflite", TFLITE_MODEL_PATH)
            shutil.move("temp_scaler.pkl", SCALER_PATH)
            st.success("New model activated successfully! The app will now use the retrained model for predictions.")
            st.cache_resource.clear()
            del st.session_state.retrain_results

# --- MAIN APP ROUTER ---
if 'cities' not in st.session_state: st.session_state.cities = load_city_list()
if 'dark_mode' not in st.session_state: st.session_state.dark_mode = False

load_css("style.css"); apply_dark_mode_styles()

with st.sidebar:
    st.title("Navigation")
    page = st.sidebar.radio("Go to", ["Predictor", "Analytics Dashboard", "Retrain Model"])
    st.write("---"); st.header("Settings")
    st.session_state.dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    st.write("---"); st.header("🌦️ Live Weather Input")
    try: default_index = st.session_state.cities.index("Darbhanga")
    except: default_index = 0
    city = st.selectbox("Select Your City", options=st.session_state.cities, index=default_index)
    if st.button("Fetch Live Weather"):
        temp, hum, error = get_weather_data(city, API_KEY)
        if error: st.error(error)
        else:
            st.success(f"Weather for {city}: {temp}°C, {hum}% Humidity")
            st.session_state.live_temp, st.session_state.live_hum = temp, hum

if page == "Predictor": show_predictor_page()
elif page == "Analytics Dashboard": show_dashboard_page()
elif page == "Retrain Model": show_retrain_page()





# app_dark_ready.py
# Full Streamlit app with robust dark mode and color-matched theme
# import streamlit as st
# import streamlit.components.v1 as components
# import numpy as np
# import pandas as pd
# import joblib
# import requests
# import os
# import tensorflow as tf
# from train import train_irrigation_model
# import shutil
# from datetime import datetime
# import plotly.express as px
# import json

# # --- CONFIGURATION & PAGE SETUP ---
# st.set_page_config(page_title="Smart Irrigation System", layout="wide", initial_sidebar_state="expanded")
# HISTORY_FILE = 'prediction_history.csv'
# API_KEY = "3a82b8373a4be272a226c6175f8492a8"
# CITIES_FILE = 'indian_cities.csv'
# TFLITE_MODEL_PATH = 'irrigation_model.tflite'
# SCALER_PATH = 'scaler.pkl'

# # --- THEME COLORS (sampled / tuned from your screenshot) ---
# # Accent (blue) sampled from your image: #0060C0 (slightly adjusted)
# ACCENT_HEX = "#0060C0"
# # Dark theme
# DARK_BG = "#0f1720"
# DARK_CARD = "#111827"
# DARK_TEXT = "#e6eef6"
# DARK_MUTED = "#c2c8d1"
# # Light theme (kept subtle like your screenshot)
# LIGHT_BG = "#f2f5f7"
# LIGHT_CARD = "#ffffff"
# LIGHT_TEXT = "#222222"
# LIGHT_MUTED = "#5b6470"

# # ---------------------------
# # Utility: load external css file if present (keeps your previous function usable)
# # ---------------------------
# def load_css(file_name):
#     try:
#         with open(file_name) as f:
#             st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
#     except FileNotFoundError:
#         pass

# # ---------------------------
# # Robust dark-mode injection with JS bridge and CSS vars
# # ---------------------------
# def apply_dark_mode_styles():
#     css = f"""
#     <style>
#     :root{{
#       --bg: {LIGHT_BG};
#       --card: {LIGHT_CARD};
#       --muted: {LIGHT_MUTED};
#       --text: {LIGHT_TEXT};
#       --accent: {ACCENT_HEX};
#       --sidebar-bg: {LIGHT_CARD};
#       --border: rgba(0,0,0,0.08);
#       --card-shadow: 0 6px 18px rgba(13,20,25,0.04);
#     }}

#     body.dark-mode {{
#       --bg: {DARK_BG};
#       --card: {DARK_CARD};
#       --muted: {DARK_MUTED};
#       --text: {DARK_TEXT};
#       --accent: {ACCENT_HEX};
#       --sidebar-bg: #0b1220;
#       --border: rgba(255,255,255,0.06);
#       --card-shadow: 0 6px 18px rgba(0,0,0,0.6);
#     }}

#     [data-testid="stAppViewContainer"] {{
#       background-color: var(--bg) !important;
#       color: var(--text) !important;
#       transition: background-color .25s ease, color .2s ease;
#     }}

#     [data-testid="stSidebar"] {{
#       background-color: var(--sidebar-bg) !important;
#       color: var(--text) !important;
#     }}

#     .card {{
#       background: var(--card) !important;
#       border: 1px solid var(--border) !important;
#       color: var(--text) !important;
#       border-radius: 12px;
#       padding: 18px;
#       box-shadow: var(--card-shadow);
#     }}

#     .stButton>button, button[kind="primary"], .css-1emrehy.edgvbvh3 {{
#       background-color: var(--accent) !important;
#       color: white !important;
#       border: none !important;
#     }}

#     .muted {{ color: var(--muted) !important; }}

#     input[type=range]::-webkit-slider-thumb {{ background: var(--accent) !important; }}
#     input[type=range]::-webkit-slider-runnable-track {{ background: rgba(0,0,0,0.06) !important; }}

#     .js-plotly-plot .plotly .main-svg, .plotly-graph-div {{ background: transparent !important; }}

#     .invert-for-dark {{ filter: invert(1) hue-rotate(180deg); }}

#     </style>
#     """

#     js = """
#     <script>
#     (function(){
#       const storageKey = "saip_dark_mode_v1";
#       // On load: apply persisted mode
#       try {
#         const current = JSON.parse(localStorage.getItem(storageKey));
#         if (current === true) document.body.classList.add("dark-mode");
#         else document.body.classList.remove("dark-mode");
#       } catch(e){}
#       // Expose setter for Streamlit to call
#       window.setDarkMode = function(v){
#         try {
#           if (v) { document.body.classList.add("dark-mode"); localStorage.setItem(storageKey, JSON.stringify(true)); }
#           else { document.body.classList.remove("dark-mode"); localStorage.setItem(storageKey, JSON.stringify(false)); }
#         } catch(e){}
#       }
#     })();
#     </script>
#     """
#     components.html(css + js, height=0)

#     # call the setter to reflect current session_state immediately
#     dm = 'true' if st.session_state.get('dark_mode', False) else 'false'
#     components.html(f"<script>if(window.setDarkMode) window.setDarkMode({dm});</script>", height=0)


# # --- HELPER FUNCTIONS from your original app (kept largely intact) ---
# @st.cache_data
# def load_city_list():
#     try:
#         df = pd.read_csv(CITIES_FILE, usecols=[0], header=None)
#         return sorted(list(df[0].unique()))
#     except FileNotFoundError:
#         return ["Patna", "Gaya", "Darbhanga"]

# @st.cache_resource
# def load_interpreter_and_scaler(model_path, scaler_path):
#     try:
#         interpreter = tf.lite.Interpreter(model_path=model_path); interpreter.allocate_tensors()
#         scaler = joblib.load(scaler_path)
#         return interpreter, scaler
#     except Exception:
#         return None, None

# def predict_with_tflite(interpreter, input_data):
#     input_details = interpreter.get_input_details(); output_details = interpreter.get_output_details()
#     interpreter.set_tensor(input_details[0]['index'], np.array(input_data, dtype=np.float32))
#     interpreter.invoke()
#     return interpreter.get_tensor(output_details[0]['index'])[0][0]

# def get_weather_data(city, api_key):
#     base_url = "http://api.openweathermap.org/data/2.5/weather"; params = {'q': city, 'appid': api_key, 'units': 'metric'}
#     try:
#         response = requests.get(base_url, params=params); response.raise_for_status(); data = response.json()
#         return data['main']['temp'], data['main']['humidity'], None
#     except Exception as e:
#         return None, None, f"City not found or API key invalid. Error: {e}"

# def log_prediction(inputs, prediction):
#     log_entry = {'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")], 'Soil Moisture': [inputs[0]], 'Temperature': [inputs[1]], 'Humidity': [inputs[2]], 'Crop Type': [inputs[3]], 'Predicted Duration (min)': [f"{prediction:.2f}"]}
#     log_df = pd.DataFrame(log_entry)
#     if not os.path.exists(HISTORY_FILE):
#         log_df.to_csv(HISTORY_FILE, index=False)
#     else:
#         log_df.to_csv(HISTORY_FILE, mode='a', header=False, index=False)

# # ---------------------------
# # UI PAGES (predictor / dashboard / retrain)
# # ---------------------------
# def show_predictor_page():
#     st.title('🌾 Smart Agricultural Irrigation Predictor')

#     # logo swap based on theme
#     logo_light = "assets/logo_light.png"
#     logo_dark = "assets/logo_dark.png"
#     logo_path = logo_dark if st.session_state.get('dark_mode') else logo_light
#     try:
#         st.image(logo_path, width=140)
#     except Exception:
#         # fallback and invert if missing dark logo
#         st.image(logo_light, width=140)
#         if st.session_state.get('dark_mode'):
#             st.markdown("<style>.stImage img{ filter: invert(1) hue-rotate(180deg); }</style>", unsafe_allow_html=True)

#     col1, col2 = st.columns([1, 1.5])
#     with col1:
#         # card wrapper using the injected .card class
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.header("Input Parameters")
#         temp_default = st.session_state.get('live_temp', 30.0)
#         hum_default = st.session_state.get('live_hum', 60.0)
#         soil_moisture = st.slider('Soil Moisture (%)', 0, 100, 35)
#         temperature = st.slider('Temperature (°C)', 10, 45, int(temp_default))
#         humidity = st.slider('Air Humidity (%)', 20, 90, int(hum_default))
#         crop_options = {0: 'Wheat', 1: 'Corn', 2: 'Rice'}
#         crop_type_code = st.selectbox('Crop Type', options=list(crop_options.keys()), format_func=lambda x: crop_options[x])
#         st.markdown('</div>', unsafe_allow_html=True)
#         st.write("")  # spacer

#         if st.button('Predict Irrigation', use_container_width=True):
#             with st.spinner("Loading model for the first time..."):
#                 interpreter, scaler = load_interpreter_and_scaler(TFLITE_MODEL_PATH, SCALER_PATH)
#             if interpreter and scaler:
#                 input_data = np.array([[soil_moisture, temperature, humidity, crop_type_code]])
#                 try:
#                     input_scaled = scaler.transform(input_data)
#                     duration = predict_with_tflite(interpreter, input_scaled)
#                     st.session_state.last_prediction = {"duration": duration, "inputs": [soil_moisture, temperature, humidity, crop_options[crop_type_code]]}
#                     log_prediction([soil_moisture, temperature, humidity, crop_options[crop_type_code]], duration)
#                     st.success("Prediction logged.")
#                 except Exception as e:
#                     st.error(f"Prediction failed: {e}")
#             else:
#                 st.error("Model or scaler not found. Check model paths.")

#     with col2:
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.header("Prediction Results")
#         if 'last_prediction' in st.session_state:
#             res = st.session_state.last_prediction
#             duration = res['duration']
#             # custom metric look
#             st.markdown(f'''
#                 <div style="padding:12px;border-radius:10px;">
#                   <div style="font-size:14px;color:var(--muted)">Predicted Irrigation Duration</div>
#                   <div style="font-size:40px;font-weight:700;color:var(--accent)">{duration:.2f} minutes</div>
#                 </div>
#             ''', unsafe_allow_html=True)

#             with st.container():
#                 if duration < 15:
#                     st.success("Recommendation: Low irrigation needed.")
#                 elif duration < 30:
#                     st.info("Recommendation: Moderate irrigation recommended.")
#                 elif duration < 45:
#                     st.warning("Recommendation: High irrigation needed.")
#                 else:
#                     st.error("Recommendation: Very high irrigation needed!")

#             st.subheader("Factors Influencing Prediction")
#             chart_data = pd.DataFrame({"Value": res['inputs'][:3]}, index=["Soil Moisture", "Temperature", "Humidity"])
#             # Use plotly template that matches theme
#             plotly_template = "plotly_dark" if st.session_state.get('dark_mode') else "plotly_white"
#             fig = px.bar(chart_data.reset_index().rename(columns={'index':'Factor'}), x='Factor', y='Value', template=plotly_template)
#             fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10,r=10,t=30,b=10))
#             st.plotly_chart(fig, use_container_width=True)
#         else:
#             st.info("Make a prediction to see results.")
#         st.markdown('</div>', unsafe_allow_html=True)


# def show_dashboard_page():
#     st.title("📊 Interactive Analytics Dashboard")
#     st.write("Analyze historical prediction data to uncover trends and insights.")
#     if not os.path.exists(HISTORY_FILE):
#         st.warning("No prediction history found. Make some predictions first on the Predictor page.")
#         return

#     history_df = pd.read_csv(HISTORY_FILE)
#     if 'Timestamp' not in history_df.columns:
#         st.error("Your prediction history file is in an old format. Please delete 'prediction_history.csv' and make a new prediction.")
#         return

#     history_df['Timestamp'] = pd.to_datetime(history_df['Timestamp'])
#     st.write("---")
#     st.subheader("Prediction History (Last 20 Entries)")
#     st.dataframe(history_df.tail(20))

#     st.write("---")
#     st.subheader("Irrigation Duration Over Time")
#     plotly_template = "plotly_dark" if st.session_state.get('dark_mode') else "plotly_white"
#     fig1 = px.line(history_df, x='Timestamp', y='Predicted Duration (min)', title='Trends in Irrigation Needs', markers=True, template=plotly_template)
#     fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
#     st.plotly_chart(fig1, use_container_width=True)

#     col1, col2 = st.columns(2)
#     with col1:
#         st.subheader("Average Irrigation by Crop Type")
#         avg_duration_by_crop = history_df.groupby('Crop Type')['Predicted Duration (min)'].mean().reset_index()
#         fig2 = px.bar(avg_duration_by_crop, x='Crop Type', y='Predicted Duration (min)', title='Average Water Needs per Crop', color='Crop Type', template=plotly_template)
#         fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
#         st.plotly_chart(fig2, use_container_width=True)
#     with col2:
#         st.subheader("Temperature vs. Irrigation Duration")
#         fig3 = px.scatter(history_df, x='Temperature', y='Predicted Duration (min)', title='Impact of Temperature', color='Crop Type', trendline="ols", template=plotly_template)
#         fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
#         st.plotly_chart(fig3, use_container_width=True)


# def show_retrain_page():
#     st.title("🔄 One-Click Model Retraining")
#     st.write("Improve the model's accuracy by training it on new data.")
#     st.info("Your uploaded CSV file must have the columns: `soil_moisture`, `temperature`, `humidity`, `crop_type`, `irrigation_duration_minutes`.")
#     uploaded_file = st.file_uploader("Upload your new training data (CSV)", type="csv")
#     if uploaded_file is not None:
#         new_data_df = pd.read_csv(uploaded_file)
#         st.write("Data Preview:"); st.dataframe(new_data_df.head())
#         epochs = st.slider("Select number of training epochs", 10, 200, 50)
#         if st.button("Start Retraining"):
#             with st.spinner("Retraining in progress... This may take a moment."):
#                 temp_data_path = "temp_training_data.csv"; new_data_df.to_csv(temp_data_path, index=False)
#                 new_model_path = "temp_model.keras"; new_scaler_path = "temp_scaler.pkl"
#                 performance, history = train_irrigation_model(dataset_path=temp_data_path, model_output_path=new_model_path, scaler_output_path=new_scaler_path, epochs=epochs)
#                 st.session_state.retrain_results = {"performance": performance, "history": history}
#                 os.remove(temp_data_path)
#     if 'retrain_results' in st.session_state:
#         st.success("Retraining complete!")
#         results = st.session_state.retrain_results
#         st.subheader("New Model Performance"); st.json(results['performance'])
#         st.subheader("Training & Validation Loss")
#         fig = px.line(x=list(range(1, len(results['history'].history['loss']) + 1)),
#                       y=[results['history'].history['loss'], results['history'].history['val_loss']])
#         fig.update_layout(xaxis_title="Epoch", yaxis_title="Loss", legend_title="Metric")
#         fig.data[0].name = "Training Loss"; fig.data[1].name = "Validation Loss"
#         fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", template=("plotly_dark" if st.session_state.get('dark_mode') else "plotly_white"))
#         st.plotly_chart(fig, use_container_width=True)

#         st.warning("The new model is trained but NOT YET ACTIVE. Click the button below to replace the current active model.")
#         if st.button("Activate New Model"):
#             from model import HybridSwishReLU
#             import tensorflow as _tf
#             new_keras_model = _tf.keras.models.load_model("temp_model.keras", custom_objects={'HybridSwishReLU': HybridSwishReLU})
#             converter = _tf.lite.TFLiteConverter.from_keras_model(new_keras_model)
#             converter.optimizations = [_tf.lite.Optimize.DEFAULT]
#             tflite_model_content = converter.convert()
#             with open("temp_model.tflite", 'wb') as f:
#                 f.write(tflite_model_content)

#             shutil.move("temp_model.tflite", TFLITE_MODEL_PATH)
#             shutil.move("temp_scaler.pkl", SCALER_PATH)
#             st.success("New model activated successfully! The app will now use the retrained model for predictions.")
#             st.cache_resource.clear()
#             del st.session_state.retrain_results


# # --- MAIN APP ROUTER & Sidebar with theme toggle ---
# if 'cities' not in st.session_state:
#     st.session_state.cities = load_city_list()
# if 'dark_mode' not in st.session_state:
#     # Try to read persisted value from localStorage via small script
#     st.session_state.dark_mode = False

# # Load optional external style file
# load_css("style.css")

# # Inject our theme CSS + JS bridge
# apply_dark_mode_styles()

# with st.sidebar:
#     st.title("Navigation")
#     page = st.radio("Go to", ["Predictor", "Analytics Dashboard", "Retrain Model"])
#     st.write("---")
#     st.header("Settings")

#     # Theme checkbox (robust across Streamlit versions)
#     dark_choice = st.checkbox("🌙 Dark Mode", value=st.session_state.get('dark_mode', False))
#     st.session_state.dark_mode = dark_choice
#     # Immediately call JS setter to update body class & persist to localStorage
#     components.html(f"<script>if(window.setDarkMode) window.setDarkMode({str(dark_choice).lower()});</script>", height=0)

#     st.write("---")
#     st.header("🌦️ Live Weather Input")
#     try:
#         default_index = st.session_state.cities.index("Darbhanga")
#     except ValueError:
#         default_index = 0
#     city = st.selectbox("Select Your City", options=st.session_state.cities, index=default_index)
#     if st.button("Fetch Live Weather"):
#         temp, hum, error = get_weather_data(city, API_KEY)
#         if error:
#             st.error(error)
#         else:
#             st.success(f"Weather for {city}: {temp}°C, {hum}% Humidity")
#             st.session_state.live_temp, st.session_state.live_hum = temp, hum

# # Router
# if page == "Predictor":
#     show_predictor_page()
# elif page == "Analytics Dashboard":
#     show_dashboard_page()
# elif page == "Retrain Model":
#     show_retrain_page()

# End of file