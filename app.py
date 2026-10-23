import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model


model = load_model("lstm_reservoir_inflow_model.keras")
scaler = joblib.load("reservoir_inflow_scaler.pkl")


st.set_page_config(
    page_title="Uzhavan with AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""



<style>
[data-testid="stSidebar"] {
    background: #061a18 !important;
}

[data-testid="stSidebar"] > div {
    background: #061a18 !important;
}

[data-testid="stSidebar"] section {
    background: #061a18 !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>


.stApp {
    background:
        radial-gradient(circle at top left, #123c3c 0%, transparent 35%),
        radial-gradient(circle at bottom right, #062d35 0%, transparent 40%),
        linear-gradient(135deg, #031f25, #052f32 50%, #031b24);
    color: white;
}


.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}


.hero-title {
    font-size: 58px;
    font-weight: 800;
    line-height: 1.05;
    color: white;
    margin-bottom: 8px;
}

.hero-title span {
    background: linear-gradient(90deg, #62ff9a, #32e6ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #a8c5c5;
    font-size: 17px;
    margin-bottom: 35px;
}


.section-number {
.white-number {
    color: white !important;
}
    color: #62ff9a;
    font-size: 14px;
    font-weight: 800;
    letter-spacing: 3px;
    margin-bottom: 5px;

}
.section-title {
    color: white;
    font-size: 30px;
    font-weight: 750;
    margin-bottom: 20px;
}


.glass-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.13);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border-radius: 24px;
    padding: 28px;
    box-shadow: 0 15px 45px rgba(0,0,0,0.25);
}



.kpi-card {
    border-radius: 22px;
    padding: 25px;
    min-height: 150px;
    box-shadow: 0 12px 35px rgba(0,0,0,0.3);
}

.kpi-green {
    background: linear-gradient(135deg, #3cff8f, #16c784);
    color: #03251b;
}

.kpi-blue {
    background: linear-gradient(135deg, #3de8ff, #238cff);
    color: white;
}

.kpi-purple {
    background: linear-gradient(135deg, #9b6cff, #6740d9);
    color: white;
}

.kpi-orange {
    background: linear-gradient(135deg, #ffbd5a, #ff7657);
    color: #301407;
}

.kpi-label {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    opacity: 0.85;
}

.kpi-value {
    font-size: 38px;
    font-weight: 800;
    margin-top: 15px;
}


div[data-baseweb="input"],
div[data-baseweb="select"] {
    background: rgba(255,255,255,0.08);
    border-radius: 12px;
}

label {
    color: #d5eeee !important;
}


.stButton > button {
    width: 100%;
    border: none;
    border-radius: 14px;
    padding: 13px 20px;
    font-weight: 700;
    color: #03251b;
    background: linear-gradient(90deg, #62ff9a, #35e5ff);
    box-shadow: 0 8px 25px rgba(50,230,200,0.2);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 30px rgba(50,230,200,0.35);
}

/* ===== DATAFRAME / TABLE ===== */
div[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
}

/* ===== ALERT BOX ===== */
div[data-testid="stAlert"] {
    border-radius: 16px;
}



[data-testid="stSidebar"] {
    background: #061a18 !important;
}

[data-testid="stSidebar"] > div:first-child {
    background: #061a18 !important;
}

[data-testid="stSidebar"] > div:first-child > div {
    background: #061a18 !important;
}

[data-testid="stSidebarContent"] {
    background: #061a18 !important;
}

[data-testid="stSidebarUserContent"] {
    background: #061a18 !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label {
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)



st.markdown(
    """
    <div class="section-number">SMART AGRICULTURE • AI POWERED</div>

    <div class="hero-title">
        Uzhavan <span>with AI</span>
    </div>

    <div class="hero-subtitle">
        AI-Powered Smart Water Distribution & Crop Recommendation System
        <br>
        Intelligent insights for sustainable farming across Tamil Nadu
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div style="margin-top: 10px; margin-bottom: 20px;">
    <div class="section-number">01</div>
    <div class="section-title">Reservoir Inflow Forecast</div>
</div>
""", unsafe_allow_html=True)

st.write(
    "Use the previous 4 reservoir inflow values to predict "
    "the next reservoir inflow using our trained LSTM model."
)

st.markdown('</div>', unsafe_allow_html=True)



st.markdown("""
<div style="margin-top: 25px; margin-bottom: 15px;">
    <div class="section-number">INPUT DATA</div>
    <div style="
        color: white;
        font-size: 22px;
        font-weight: 700;
    ">
        Previous Reservoir Inflow
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    value1 = st.number_input(
        "Inflow 1 (MCFT)",
        value=42.2
    )

with col2:
    value2 = st.number_input(
        "Inflow 2 (MCFT)",
        value=42.2
    )

with col3:
    value3 = st.number_input(
        "Inflow 3 (MCFT)",
        value=43.3
    )

with col4:
    value4 = st.number_input(
        "Inflow 4 (MCFT)",
        value=43.3
    )

st.write("")



st.markdown("""
<div style="
    margin-top: 20px;
    margin-bottom: 10px;
    color: #8affb8;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
">
    AI Prediction
</div>
""", unsafe_allow_html=True)

if st.button("🔮 Predict Reservoir Inflow", use_container_width=True):

 
    values = np.array([
    inflow_value,
    inflow_value,
    inflow_value,
    inflow_value
]).reshape(-1, 1)


    # Scale input
    scaled_values = scaler.transform(values)

    # Reshape for LSTM
    X_new = scaled_values.reshape(1, 4, 1)

    # Prediction
    prediction_scaled = model.predict(
        X_new,
        verbose=0
    )

    # Convert prediction back to MCFT
    predicted_mcft = scaler.inverse_transform(
        prediction_scaled
    )[0][0]

    st.markdown(
        f"""
        <div class="result">
            🌊 Predicted Reservoir Inflow<br>
            {predicted_mcft:.2f} MCFT
        </div>
        """,
        unsafe_allow_html=True
    )

    st.success(
        "LSTM prediction completed successfully!"
    )



st.divider()



c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    ### 💧 Smart Water Distribution
    AI-based water allocation for agricultural regions.
    """)

with c2:
    st.markdown("""
    ### 🌾 Crop Recommendation
    Recommend suitable crops based on season and water availability.
    """)

with c3:
    st.markdown("""
    ### 🌧️ Forecasting
    LSTM-based reservoir inflow forecasting using historical data.
    """)

st.divider()

st.caption(
    "Uzhavan with AI | Final Year Project | AI-Powered Agriculture"
)


st.divider()

st.markdown("""
<div style="margin-top: 35px; margin-bottom: 20px;">
    <div class="section-number">03</div>
    <div class="section-title">AI Crop Recommendation</div>
</div>
""", unsafe_allow_html=True)

st.write(
    "Select the district, season, and available water to get "
    "a suitable crop recommendation."
)

crop_col1, crop_col2, crop_col3 = st.columns(3)

with crop_col1:
    district = st.selectbox(
        "📍 Select District",
        [
            "Thanjavur",
            "Tiruvarur",
            "Pudukkottai",
            "Nagapattinam"
        ]
    )

with crop_col2:
    season = st.selectbox(
        "🌾 Select Season",
        [
            "Kuruvai",
            "Samba",
            "Navarai"
        ]
    )

with crop_col3:
    available_water = st.number_input(
        "💧 Available Water (MCFT)",
        min_value=0.0,
        value=100.0
    )

if st.button("🌱 Recommend Crop", use_container_width=True):

    if season == "Kuruvai":
        if available_water >= 80:
            crop = "Rice"
            requirement = 80
        else:
            crop = "Groundnut"
            requirement = 50

    elif season == "Samba":
        if available_water >= 100:
            crop = "Rice"
            requirement = 100
        else:
            crop = "Maize"
            requirement = 60

    else:  # Navarai
        if available_water >= 70:
            crop = "Rice"
            requirement = 70
        else:
            crop = "Groundnut"
            requirement = 50

    st.success(f"🌾 Recommended Crop: {crop}")

    st.info(
        f"Estimated Water Requirement: {requirement} MCFT"
    )

    if available_water >= requirement:
        st.write("✅ Available water is sufficient for the recommended crop.")
    else:
        st.warning(
            "⚠️ Water availability is low. Consider a lower-water crop."
        )

    st.caption(f"Selected District: {district}")


st.divider()

st.markdown("""
<div style="margin-top: 35px; margin-bottom: 20px;">
    <div class="section-number">04</div>
    <div class="section-title">Smart Water Allocation</div>
</div>
""", unsafe_allow_html=True)

st.write(
    "Fairly distribute available water among the four delta districts "
    "while protecting downstream water availability."
)

water_available = st.number_input(
    "💧 Total Available Water (MCFT)",
    min_value=0.0,
    value=400.0
)

if st.button("🚰 Allocate Water", use_container_width=True):

    districts = [
        "Thanjavur",
        "Tiruvarur",
        "Pudukkottai",
        "Nagapattinam"
    ]

    # Minimum protected allocation for each district
    minimum_water = 50.0

    total_minimum = minimum_water * len(districts)

    if water_available < total_minimum:
        st.error(
            "⚠️ Available water is too low to provide the minimum "
            "allocation to all districts."
        )
    else:
        remaining_water = water_available - total_minimum

        # Fair distribution of remaining water
        extra_water = remaining_water / len(districts)

        allocation = round(minimum_water + extra_water, 2)

        st.success("✅ Water allocation completed successfully!")

        for district in districts:
            st.write(
                f"🌾 **{district}: {allocation:.2f} MCFT**"
            )

        st.info(
            f"💧 Total Allocated Water: "
            f"{water_available:.2f} MCFT"
        )

        st.caption(
            "The allocation provides a minimum protected amount "
            "to each district and distributes the remaining water fairly."
        )


st.divider()

st.markdown("""
<div style="margin-top: 35px; margin-bottom: 20px;">
    <div class="section-number">07</div>
    <div class="section-title">Reservoir Inflow Forecast</div>
</div>
""", unsafe_allow_html=True)


chart_values = np.array([
    inflow_value,
    inflow_value,
    inflow_value,
    inflow_value
])


if "predicted_mcft" in locals():

    chart_data = np.append(
        chart_values,
        predicted_mcft
    )

    st.line_chart(chart_data)

    st.caption(
        "The chart shows the previous 4 inflow values "
        "and the predicted next inflow."
    )

else:

    st.info(
        "🔮 Click 'Predict Reservoir Inflow' "
        "to display the forecast graph."
    )


st.divider()

st.markdown("""
<div style="margin-top: 35px; margin-bottom: 20px;">
    <div class="section-number">05</div>
    <div class="section-title">Water Availability Alert</div>
</div>
""", unsafe_allow_html=True)

alert_water = st.number_input(
    "💧 Current Available Water (MCFT)",
    min_value=0.0,
    value=100.0
)

minimum_required = 200.0

if st.button("⚠️ Check Water Status", use_container_width=True):

    if alert_water < 100:
        st.error(
            "🚨 Critical Water Shortage! "
            "Immediate water conservation is required."
        )

    elif alert_water < minimum_required:
        st.warning(
            "⚠️ Low Water Availability! "
            "Consider reducing water-intensive crops."
        )

    else:
        st.success(
            "✅ Water Availability is Sufficient."
        )

    st.write(
        f"Current Available Water: **{alert_water:.2f} MCFT**"
    )



st.divider()

st.subheader("🌧️ Environmental & Reservoir Inputs")

input_col1, input_col2, input_col3 = st.columns(3)

with input_col1:
    rainfall = st.number_input(
        "🌧️ Rainfall (mm)",
        min_value=0.0,
        value=50.0
    )

with input_col2:
    storage = st.number_input(
        "💧 Reservoir Storage (%)",
        min_value=0.0,
        max_value=100.0,
        value=60.0
    )

with input_col3:
    crop_requirement = st.number_input(
        "🌾 Crop Water Requirement (MCFT)",
        min_value=0.0,
        value=80.0
    )

if st.button("📊 Analyze Water Condition", use_container_width=True):

    if storage < 30:
        st.error("🚨 Reservoir storage is critically low.")

    elif storage < 50:
        st.warning("⚠️ Reservoir storage is below normal.")

    else:
        st.success("✅ Reservoir storage is at a suitable level.")

    if rainfall < 20:
        st.warning("🌧️ Low rainfall detected. Additional irrigation may be required.")
    else:
        st.success("🌧️ Rainfall availability is satisfactory.")

    st.write(
        f"Crop Water Requirement: **{crop_requirement:.2f} MCFT**"
    )


st.divider()

st.markdown("""
<div style="margin-top: 35px; margin-bottom: 20px;">
    <div class="section-number">06</div>
    <div class="section-title">Smart Irrigation Schedule</div>
</div>
""", unsafe_allow_html=True)

st.write(
    "Generate an irrigation schedule based on crop water requirement "
    "and available water."
)

irrigation_col1, irrigation_col2 = st.columns(2)

with irrigation_col1:
    irrigation_crop = st.selectbox(
        "🌾 Select Crop",
        ["Rice", "Maize", "Groundnut"]
    )

with irrigation_col2:
 irrigation_water = st.number_input(
    "💧 Available Water (MCFT)",
    min_value=0.0,
    value=100.0,
    key="irrigation_water_input"
)

if st.button("💦 Generate Irrigation Schedule", use_container_width=True):

    if irrigation_crop == "Rice":
        required_water = 80.0
        frequency = "Every 2 days"

    elif irrigation_crop == "Maize":
        required_water = 60.0
        frequency = "Every 3 days"

    else:
        required_water = 50.0
        frequency = "Every 4 days"

    st.success("✅ Irrigation schedule generated successfully!")

    st.write(f"🌾 **Crop:** {irrigation_crop}")
    st.write(f"💧 **Required Water:** {required_water:.2f} MCFT")
    st.write(f"📅 **Irrigation Frequency:** {frequency}")

    if irrigation_water >= required_water:
        st.info("✅ Available water is sufficient for irrigation.")
    else:
        st.warning(
            "⚠️ Available water is insufficient. "
            "Water conservation is recommended."
        )


st.divider()

st.markdown("""
<div style="margin-top: 35px; margin-bottom: 20px;">
    <div class="section-number white-number">02</div>
    <div class="section-title">Smart Farming Analysis</div>
</div>
""", unsafe_allow_html=True)
summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

with summary_col1:
    st.markdown(f"""
    <div class="glass-card">
        <div style="color:#d5eeee; font-size:14px;">🌧️ Rainfall</div>
        <div style="color:white; font-size:32px; font-weight:800; margin-top:10px;">
            {rainfall:.1f} mm
        </div>
    </div>
    """, unsafe_allow_html=True)

with summary_col2:
    st.markdown(f"""
    <div class="glass-card">
        <div style="color:#d5eeee; font-size:14px;">💧 Storage</div>
        <div style="color:white; font-size:32px; font-weight:800; margin-top:10px;">
            {storage:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

with summary_col3:
    st.markdown(f"""
    <div class="glass-card">
        <div style="color:#d5eeee; font-size:14px;">🌾 Crop Requirement</div>
        <div style="color:white; font-size:32px; font-weight:800; margin-top:10px;">
            {crop_requirement:.1f} MCFT
        </div>
    </div>
    """, unsafe_allow_html=True)

with summary_col4:
    if "predicted_mcft" in locals():
        forecast_value = f"{predicted_mcft:.2f} MCFT"
    else:
        forecast_value = "Not available"

    st.markdown(f"""
    <div class="glass-card">
        <div style="color:#d5eeee; font-size:14px;">🌊 Predicted Inflow</div>
        <div style="color:white; font-size:32px; font-weight:800; margin-top:10px;">
            {forecast_value}
        </div>
    </div>
    """, unsafe_allow_html=True)


st.info(
    "🤖 AI Summary: The system combines rainfall, reservoir storage, "
    "crop water requirement, and LSTM-based inflow forecasting "
    "to support smart agricultural water management."
)



    
  



kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown("""
    <div class="glass-card">
        <div style="color:#d5eeee; font-size:14px;">🌧️ Rainfall</div>
        <div style="color:white; font-size:32px; font-weight:800; margin-top:10px;">
            50 mm
        </div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown("""
    <div class="glass-card">
        <div style="color:#d5eeee; font-size:14px;">💧 Reservoir Storage</div>
        <div style="color:white; font-size:32px; font-weight:800; margin-top:10px;">
            60%
        </div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown("""
    <div class="glass-card">
        <div style="color:#d5eeee; font-size:14px;">🌾 Crop Requirement</div>
        <div style="color:white; font-size:32px; font-weight:800; margin-top:10px;">
            80 MCFT
        </div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    if "predicted_mcft" in locals():
        forecast_value = f"{predicted_mcft:.2f} MCFT"
    else:
        forecast_value = "Not available"

    st.markdown(f"""
    <div class="glass-card">
        <div style="color:#d5eeee; font-size:14px;">🌊 LSTM Forecast</div>
        <div style="color:white; font-size:32px; font-weight:800; margin-top:10px;">
            {forecast_value}
        </div>
    </div>
    """, unsafe_allow_html=True)


st.divider()

st.subheader("🌾 About Uzhavan with AI")

st.write(
    "Uzhavan with AI is an AI-powered smart agriculture system "
    "designed to support efficient water management and crop planning. "
    "The system uses LSTM-based reservoir inflow forecasting, "
    "smart water distribution, crop recommendation, rainfall analysis, "
    "water shortage alerts, and irrigation scheduling."
)

about_col1, about_col2 = st.columns(2)

with about_col1:
    st.markdown("### 🎯 Project Objective")
    st.write(
        "To improve agricultural water management by predicting "
        "water availability and supporting fair water distribution."
    )

with about_col2:
    st.markdown("### 🤖 AI Technology")
    st.write(
        "LSTM is used to forecast reservoir inflow from historical "
        "time-series data and support better water management decisions."
    )


st.sidebar.title("🌾 Uzhavan with AI")
st.sidebar.caption("Smart Agriculture Dashboard")

st.sidebar.divider()

st.sidebar.markdown("### 🌱 System Modules")

st.sidebar.write("💧 Reservoir Forecast")
st.sidebar.write("🌾 Crop Recommendation")
st.sidebar.write("🚰 Smart Water Distribution")
st.sidebar.write("⚠️ Water Availability Alert")
st.sidebar.write("💦 Irrigation Schedule")

st.sidebar.divider()

with st.sidebar.expander("📌 Project Information"):
    st.write("**Project:** Uzhavan with AI")
    st.write("**Domain:** Smart Agriculture")
    st.write("**Technology:** Python, LSTM, Streamlit")
    st.write("**Focus:** Water Management & Crop Recommendation")
    st.write("**Academic Year:** 2026–2027")
