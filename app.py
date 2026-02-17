import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Heart Risk Predictor",
    page_icon="❤️",
    layout="wide"
)

# ================= CUSTOM CSS (GLASS EFFECT) =================
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

.glass {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
}

.stButton>button {
    background: linear-gradient(45deg, #ff416c, #ff4b2b);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size:18px;
}

</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# ================= HEADER =================
st.markdown("<h1 style='text-align:center;'>❤️ AI Heart Disease Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Advanced ML Based Clinical Risk Analysis</p>", unsafe_allow_html=True)

# ================= SIDEBAR INPUT =================
with st.sidebar:
    st.header("🧾 Patient Details")

    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_bp = st.number_input("Resting BP", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol", 100, 600, 200)
    fasting_bs = st.selectbox("Fasting Sugar >120", [0, 1])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"])
    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    predict_btn = st.button("🔍 Analyze Risk")

# ================= MAIN AREA =================
col1, col2 = st.columns([1.2, 1])

if predict_btn:

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    try:
        prob = model.predict_proba(scaled_input)[0][1]
    except:
        prob = 0.7 if prediction == 1 else 0.2

    # ================= LEFT SIDE - GAUGE =================
    with col1:
        st.markdown("### 🧠 AI Risk Analysis")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Heart Disease Risk %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red" if prob > 0.5 else "green"},
                'steps': [
                    {'range': [0, 50], 'color': "#2ecc71"},
                    {'range': [50, 100], 'color': "#e74c3c"}
                ]
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

    # ================= RIGHT SIDE - CLEAN CARD =================
    with col2:

        if prediction == 1:

            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #3a1c71, #d76d77, #ffaf7b);
                padding: 25px;
                border-radius: 15px;
                color: white;
            ">
            <h2>⚠ High Risk Detected</h2>
            <h4>Recommended Actions:</h4>
            <ul style="font-size:17px;">
                <li>Consult Cardiologist Immediately</li>
                <li>Reduce Salt Intake</li>
                <li>Avoid Smoking & Alcohol</li>
                <li>Daily 30 min Walk</li>
                <li>Monitor Blood Pressure Regularly</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #0f9b0f, #000000);
                padding: 25px;
                border-radius: 15px;
                color: white;
            ">
            <h2>✅ Maintain Good Health</h2>
            <ul style="font-size:18px;">
                <li>Balanced Diet</li>
                <li>Regular Exercise</li>
                <li>Annual Health Checkup</li>
                <li>Maintain Healthy Weight</li>
                <li>Stay Hydrated</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("---")
st.markdown("<p style='text-align:center;'>Developed by Anmol | AI Powered Healthcare</p>", unsafe_allow_html=True)
