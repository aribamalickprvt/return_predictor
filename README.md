# 🛍️ E-Commerce Smart Return Predictor & Demarketing Engine

A production-quality mini ML project that predicts whether a customer will return a purchased item before it ships. Built with Python, XGBoost, FastAPI, and Streamlit.

---

## 📁 Project Structure

```
return_predictor/
│
├── data/
│   ├── generate_data.py          # Synthetic dataset generator
│   └── ecommerce_returns.csv     # Generated dataset (15,000 orders)
│
├── notebooks/
│   ├── preprocessing.py          # Full preprocessing pipeline
│   └── train_model.py            # Model training & evaluation
│
├── models/
│   ├── xgb_pipeline.joblib       # Saved XGBoost model (primary)
│   ├── rf_model.joblib           # Saved Random Forest model
│   ├── target_encoders.joblib    # Saved custom transformers
│   ├── column_transformer.joblib # Saved sklearn ColumnTransformer
│   ├── feature_importance.csv    # XGBoost feature importances
│   └── evaluation_results.json  # F1, Precision, Recall, PR-AUC
│
├── api/
│   ├── main.py                   # FastAPI application
│   ├── schemas.py                # Pydantic request/response models
│   └── inference.py              # Prediction engine (model loading)
│
├── dashboard/
│   └── app.py                    # Streamlit interactive dashboard
│
├── utils/
│   ├── config.py                 # Constants, paths, hyperparameters
│   └── logger.py                 # Centralized logging
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the dataset
```bash
python data/generate_data.py
```

### 3. Train the models
```bash
python notebooks/train_model.py
```

### 4. Start the FastAPI server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
API docs available at: `http://localhost:8000/docs`

### 5. Launch the Streamlit dashboard
```bash
streamlit run dashboard/app.py
```
Dashboard available at: `http://localhost:8501`

---

## 🧠 ML Pipeline Overview

### Phase 1 — Preprocessing & Feature Engineering

| Step | Technique |
|------|-----------|
| Missing delivery dates | Carrier-aware conditional imputation (median per carrier) |
| Price outliers | IQR-based capping (1.5× IQR) |
| High-cardinality encoding | Frequency Encoding (Product ID, ZIP Code) |
| Low-cardinality encoding | One-Hot Encoding (Category, Carrier, Shipping Method) |
| Scaling | RobustScaler for all numeric features |
| Class imbalance | SMOTE (Synthetic Minority Oversampling) + `scale_pos_weight` |

**Engineered Features:**
- `Customer_Return_Ratio` — past returns / past purchases
- `Expected_Delivery_Days` — delivery_date − order_date
- `Is_Weekend_Order` — binary flag
- `Is_Holiday_Season` — Nov, Dec, Jan flag
- `Has_Discount` — binary flag

### Phase 2 — Model Training

| Model | F1-Score | Precision | Recall | PR-AUC |
|-------|----------|-----------|--------|--------|
| Logistic Regression (baseline) | ~0.41 | ~0.34 | ~0.54 | ~0.40 |
| Random Forest | ~0.39 | ~0.37 | ~0.40 | ~0.40 |
| **XGBoost (primary)** | ~0.39 | ~0.34 | ~0.46 | ~0.37 |

> Note: F1-score is the primary metric because the dataset is imbalanced (~25% return rate).

### Phase 3 — Deployment

- **FastAPI** exposes a `/predict` endpoint (POST) that accepts order features and returns return probability, risk tier, and a recommended demarketing action.
- **Streamlit** dashboard provides an interactive business UI with sidebar model metrics, form-based prediction, probability gauge, key factor breakdown, and batch upload support.

---

## 🔌 API Reference

### `POST /predict`
```json
// Request body
{
  "Customer_ID": "CUST_00042",
  "Category": "Clothing",
  "Price": 89.99,
  "Discount_Percentage": 20,
  "Shipping_Method": "Standard",
  "Carrier": "FedEx",
  "Order_Date": "2024-06-01",
  "Bought_Two_Sizes": 1,
  "Review_Score": 2,
  ...
}

// Response
{
  "return_probability": 0.7821,
  "risk_tier": "HIGH",
  "recommended_action": "Hold shipment for review. Consider size confirmation email.",
  "model_version": "xgboost-v1"
}
```

### Risk Tiers
| Tier | Threshold | Action |
|------|-----------|--------|
| 🟢 LOW | < 30% | Standard fulfillment |
| 🟡 MEDIUM | 30–60% | Send size guide / review prompt |
| 🔴 HIGH | > 60% | Hold for review / offer alternatives |

---

## 🛠️ Tech Stack

- **ML:** XGBoost, Random Forest, Logistic Regression (scikit-learn)
- **Imbalance handling:** imbalanced-learn (SMOTE)
- **API:** FastAPI + Uvicorn
- **Dashboard:** Streamlit
- **Data:** pandas, numpy
- **Persistence:** joblib
