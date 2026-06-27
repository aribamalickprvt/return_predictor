"""
Streamlit Dashboard — Smart Return Predictor
=============================================
Interactive business dashboard for predicting e-commerce returns.
Calls the FastAPI backend to get predictions.
"""

import sys
import json
from pathlib import Path
from datetime import date, timedelta

import streamlit as st
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import (
    API_PORT, LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD,
    MODEL_ARTIFACT_PATH,
)

API_BASE = f"http://127.0.0.1:{API_PORT}"

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Smart Return Predictor",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252b3e);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #2e3450;
        text-align: center;
    }
    .metric-label { color: #8892b0; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.1em; }
    .metric-value { color: #e6f1ff; font-size: 2rem; font-weight: 700; margin-top: 0.3rem; }
    .risk-low    { background: linear-gradient(135deg, #0d3b2e, #1a5c42); border-color: #27ae60; color: #2ecc71; }
    .risk-medium { background: linear-gradient(135deg, #3b2a0d, #5c4019); border-color: #e67e22; color: #f39c12; }
    .risk-high   { background: linear-gradient(135deg, #3b0d0d, #5c1a1a); border-color: #c0392b; color: #e74c3c; }
    .action-box {
        background: #1a1f2e;
        border-left: 4px solid #6c63ff;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        margin-top: 0.8rem;
        color: #cdd6f4;
        font-size: 0.95rem;
    }
    .section-header {
        color: #6c63ff;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.5rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #6c63ff, #a78bfa);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# 🛍️ Smart Return Predictor")
st.markdown(
    "**Demarketing Engine** — Predict whether a customer will return a purchased item "
    "before it ships, and take targeted action to reduce reverse logistics costs."
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — API status & model info
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ System Status")

    try:
        resp = requests.get(f"{API_BASE}/health", timeout=3)
        health = resp.json()
        st.success("✅ API Connected")

        eval_data = health.get("evaluation_summary", [])
        if eval_data:
            st.markdown("#### 📊 Model Evaluation")
            for m in eval_data:
                with st.expander(m["model"]):
                    st.metric("F1-Score", m["f1_score"])
                    st.metric("Precision", m["precision"])
                    st.metric("Recall", m["recall"])
                    st.metric("PR-AUC", m["pr_auc"])
    except Exception:
        st.error("❌ API Offline\nStart the FastAPI server first:\n\n`uvicorn api.main:app`")

    st.divider()
    st.markdown("### 🎯 Risk Thresholds")
    st.markdown(f"🟢 **Low** — < {LOW_RISK_THRESHOLD:.0%}")
    st.markdown(f"🟡 **Medium** — {LOW_RISK_THRESHOLD:.0%} – {HIGH_RISK_THRESHOLD:.0%}")
    st.markdown(f"🔴 **High** — > {HIGH_RISK_THRESHOLD:.0%}")

    st.divider()
    st.markdown("### 📁 Feature Importance")

    fi_path = MODEL_ARTIFACT_PATH.parent / "feature_importance.csv"
    if fi_path.exists():
        fi_df = pd.read_csv(fi_path).head(10)
        st.bar_chart(fi_df.set_index("feature")["importance"])

# ─────────────────────────────────────────────────────────────────────────────
# Main form — two columns
# ─────────────────────────────────────────────────────────────────────────────

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<p class="section-header">🧑 Customer Profile</p>', unsafe_allow_html=True)

    customer_id = st.text_input("Customer ID", value="CUST_00042")
    zip_code = st.text_input("ZIP Code", value="10001")
    account_age = st.slider("Account Age (days)", 30, 2000, 365)
    past_purchases = st.number_input("Past Purchases", min_value=1, max_value=500, value=15)
    past_returns = st.number_input("Past Returns", min_value=0, max_value=500, value=2)

    st.markdown('<p class="section-header" style="margin-top:1.2rem">📦 Product Details</p>', unsafe_allow_html=True)

    category = st.selectbox(
        "Category",
        ["Clothing", "Electronics", "Footwear", "Home & Kitchen", "Books", "Toys", "Beauty"],
    )
    product_id = st.text_input("Product ID", value="PROD_1234")
    price = st.number_input("Price ($)", min_value=1.0, max_value=5000.0, value=89.99, step=0.01)
    discount = st.slider("Discount %", 0, 70, 15)
    review_score = st.select_slider("Review Score", options=[1, 2, 3, 4, 5], value=3)


with col_right:
    st.markdown('<p class="section-header">🚚 Shipping Details</p>', unsafe_allow_html=True)

    shipping_method = st.selectbox("Shipping Method", ["Standard", "Express", "Economy", "Next Day"])
    carrier = st.selectbox("Carrier", ["FedEx", "UPS", "USPS", "DHL", "BlueDart"])
    order_date = st.date_input("Order Date", value=date.today() - timedelta(days=3))
    delivery_date = st.date_input("Delivery Date (optional)", value=None)

    st.markdown('<p class="section-header" style="margin-top:1.2rem">👗 Size & Return Info</p>', unsafe_allow_html=True)

    size_ordered = None
    bought_two_sizes = 0
    if category in ["Clothing", "Footwear"]:
        size_ordered = st.selectbox("Size Ordered", ["XS", "S", "M", "L", "XL", "XXL"])
        bought_two_sizes = int(st.checkbox("⚠️ Customer bought TWO different sizes (size mismatch risk)"))
    else:
        st.info("Size fields are only relevant for Clothing and Footwear.")

    return_reason = st.text_input("Return Reason (if known)", value="")

# ─────────────────────────────────────────────────────────────────────────────
# Predict button
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
predict_col, _ = st.columns([1, 2])

with predict_col:
    predict_clicked = st.button("🔮 Predict Return Probability")

# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────

if predict_clicked:
    payload = {
        "Customer_ID": customer_id,
        "ZIP_Code": zip_code,
        "Account_Age_Days": int(account_age),
        "Past_Purchases": int(past_purchases),
        "Past_Returns": int(past_returns),
        "Product_ID": product_id,
        "Category": category,
        "Price": float(price),
        "Discount_Percentage": float(discount),
        "Shipping_Method": shipping_method,
        "Carrier": carrier,
        "Order_Date": str(order_date),
        "Delivery_Date": str(delivery_date) if delivery_date else None,
        "Size_Ordered": size_ordered,
        "Bought_Two_Sizes": bought_two_sizes,
        "Review_Score": int(review_score),
        "Return_Reason": return_reason,
        "Is_Returned": 0,
    }

    with st.spinner("Running inference..."):
        try:
            r = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
            r.raise_for_status()
            result = r.json()

            prob = result["return_probability"]
            risk = result["risk_tier"]
            action = result["recommended_action"]

            st.divider()
            st.markdown("## 📊 Prediction Results")

            r1, r2, r3 = st.columns(3)

            with r1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Return Probability</div>
                    <div class="metric-value">{prob:.1%}</div>
                </div>""", unsafe_allow_html=True)

            with r2:
                risk_class = {"LOW": "risk-low", "MEDIUM": "risk-medium", "HIGH": "risk-high"}[risk]
                risk_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}[risk]
                st.markdown(f"""
                <div class="metric-card {risk_class}">
                    <div class="metric-label">Risk Tier</div>
                    <div class="metric-value">{risk_emoji} {risk}</div>
                </div>""", unsafe_allow_html=True)

            with r3:
                return_ratio = past_returns / max(past_purchases, 1)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Customer Return Ratio</div>
                    <div class="metric-value">{return_ratio:.1%}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="action-box">
                <strong>💡 Recommended Action:</strong><br>{action}
            </div>
            """, unsafe_allow_html=True)

            # Probability gauge
            st.markdown("### Return Probability Gauge")
            st.progress(prob)
            gauge_col1, gauge_col2, gauge_col3 = st.columns(3)
            gauge_col1.markdown("🟢 Low Risk (< 30%)")
            gauge_col2.markdown("🟡 Medium (30–60%)")
            gauge_col3.markdown("🔴 High Risk (> 60%)")

            # Key drivers summary
            st.markdown("### 🔍 Key Factors")
            factors = []
            if bought_two_sizes:
                factors.append(("⚠️ Size Mismatch", "Customer ordered 2 sizes — very strong return signal", "high"))
            if return_ratio > 0.25:
                factors.append(("📉 High Return History", f"Past return ratio: {return_ratio:.1%}", "high"))
            if discount >= 30:
                factors.append(("💸 Impulse Buy Risk", f"{discount}% discount may indicate impulse purchase", "medium"))
            if review_score <= 2:
                factors.append(("⭐ Low Review Score", f"Review score {review_score}/5 — quality concern", "medium"))
            if category in ["Clothing", "Footwear"]:
                factors.append(("👗 High-Return Category", f"{category} items have higher return rates", "low"))

            if factors:
                for name, desc, level in factors:
                    color = {"high": "#e74c3c", "medium": "#f39c12", "low": "#27ae60"}[level]
                    st.markdown(f"""
                    <div style="border-left: 3px solid {color}; padding: 0.5rem 0.8rem; margin: 0.3rem 0; background: #1a1f2e; border-radius: 0 6px 6px 0;">
                        <strong style="color:{color}">{name}</strong><br>
                        <span style="color:#8892b0; font-size:0.9rem">{desc}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No high-risk factors detected.")

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to the API. Make sure the FastAPI server is running.")
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ API Error: {e.response.text}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Batch prediction section
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
with st.expander("📂 Batch Prediction (Upload CSV)"):
    st.markdown("Upload a CSV file with the same columns as the prediction form to run bulk predictions.")
    uploaded = st.file_uploader("Upload orders CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(5))
        st.info(f"Loaded {len(df)} rows. Batch prediction via API is supported — extend this section with a loop over `/predict`.")
