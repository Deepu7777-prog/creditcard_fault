
import streamlit as st
import pandas as pd
import pickle

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>

/* Main Background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b, #0f172a);
}

/* Main Container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}

/* Headings */
h1, h2, h3 {
    color: white !important;
}

/* Normal Text */
p, label {
    color: #e2e8f0 !important;
}

/* Header Card */
.header-box {
    background: linear-gradient(135deg, #1d4ed8, #7c3aed);
    padding: 35px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.3);
}

.header-box h1 {
    color: white !important;
    margin-bottom: 5px;
}

.header-box p {
    color: #e0e7ff !important;
    font-size: 18px;
}

/* Input Cards */
.input-card {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.12);
    margin-bottom: 20px;
}

/* Button */
.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 12px;
    border: none;
    font-size: 18px;
    font-weight: bold;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
}

.stButton > button:hover {
    transform: scale(1.02);
}

/* Success Result */
.safe-card {
    background: linear-gradient(135deg, #064e3b, #065f46);
    padding: 35px;
    border-radius: 20px;
    text-align: center;
    border: 2px solid #22c55e;
    margin-top: 20px;
}

.safe-card h1 {
    color: #86efac !important;
}

.safe-card p {
    color: white !important;
    font-size: 18px;
}

/* Fraud Result */
.fraud-card {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    padding: 35px;
    border-radius: 20px;
    text-align: center;
    border: 2px solid #ef4444;
    margin-top: 20px;
}

.fraud-card h1 {
    color: #fca5a5 !important;
}

.fraud-card p {
    color: white !important;
    font-size: 18px;
}

/* Footer */
.footer {
    text-align: center;
    color: #94a3b8 !important;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODEL
# ==========================================
with open("fraud_detection_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("preprocessor.pkl", "rb") as file:
    preprocessor = pickle.load(file)


# ==========================================
# HEADER
# ==========================================
st.markdown("""
<div class="header-box">
    <h1>🛡️ FraudShield AI</h1>
    <p>Intelligent Credit Card Fraud Detection System</p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# TRANSACTION INPUT SECTION
# ==========================================
st.markdown('<div class="input-card">', unsafe_allow_html=True)

st.subheader("💳 Transaction Details")

col1, col2 = st.columns(2)

with col1:

    amount_usd = st.number_input(
        "💰 Transaction Amount (USD)",
        min_value=0.0,
        value=100.0
    )

    merchant_category = st.selectbox(
        "🏪 Merchant Category",
        ['Restaurants', 'Online Retail', 'Groceries', 'Streaming',
         'Travel', 'Gift Cards', 'Electronics', 'Fuel',
         'Gaming', 'Utilities', 'Crypto Exchange', 'Healthcare']
    )

    card_type = st.selectbox(
        "💳 Card Type",
        ['Visa', 'Mastercard', 'Amex', 'RuPay', 'Discover']
    )

with col2:

    auth_method = st.selectbox(
        "🔐 Authentication Method",
        ['OTP', '3D Secure', 'No Authentication', 'Biometric', 'PIN']
    )

    channel = st.selectbox(
        "📱 Transaction Channel",
        ['Online', 'POS', 'In-App', 'Contactless', 'ATM']
    )

    device_type = st.selectbox(
        "💻 Device Type",
        ['Android Phone', 'Mac', 'iPhone', 'POS Terminal',
         'ATM Machine', 'Tablet', 'Windows PC', 'Smart Watch']
    )

st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# ANALYZE BUTTON
# ==========================================
if st.button("🔍 ANALYZE TRANSACTION"):

    # Create complete input for ML model
    input_data = pd.DataFrame({
        "amount_usd": [amount_usd],
        "merchant_category": [merchant_category],
        "card_type": [card_type],
        "auth_method": [auth_method],
        "channel": [channel],
        "device_type": [device_type],

        # Automatic risk analysis features
        "is_foreign_transaction": [0],
        "hours_since_last_txn": [5.0],
        "txn_count_last_24h": [2],
        "distance_from_home_km": [10.0],
        "card_age_months": [24],
        "customer_age": [25],
        "account_balance_usd": [1000.0],
        "is_new_merchant": [0],
        "used_vpn": [0],
        "ip_country_mismatch": [0],
        "billing_shipping_mismatch": [0],
        "cvv_retry_count": [0],
        "velocity_score": [0.2],
        "time_of_day_hour": [12],
        "day_of_week": [2],
        "is_ai_generated_scam_attempt": [0],
        "merchant_risk_score": [0.2],
        "prior_disputes": [0]
    })

    # Preprocess
    processed_data = preprocessor.transform(input_data)

    # Predict
    prediction = model.predict(processed_data)
    probability = model.predict_proba(processed_data)[0][1]

    # ==========================================
    # RESULT DISPLAY
    # ==========================================
    if prediction[0] == 1:

        st.markdown(f"""
        <div class="fraud-card">
            <h1>🚨 FRAUD ALERT!</h1>
            <p>Suspicious transaction detected</p>
            <h2>🔴 Fraud Risk: {probability:.2%}</h2>
            <p>⚠️ Additional verification is recommended.</p>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="safe-card">
            <h1>✅ TRANSACTION SAFE</h1>
            <p>No significant fraud indicators detected</p>
            <h2>🟢 Fraud Risk: {probability:.2%}</h2>
            <p>✓ Transaction appears safe based on AI analysis.</p>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# FOOTER
# ==========================================
st.markdown("""
<div class="footer">
    🛡️ Powered by Decision Tree Machine Learning Model
    <br>
    Credit Card Fraud Detection System
</div>
""", unsafe_allow_html=True)
