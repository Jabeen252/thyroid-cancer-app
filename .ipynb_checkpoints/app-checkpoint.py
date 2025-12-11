import streamlit as st

st.title("My First Streamlit App")

name = st.text_input("Enter your name")
if st.button("Submit"):
    st.success(f"Hello, {name}! Welcome to your Streamlit app.")

import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model.pkl")

st.title("Thyroid Cancer Recurrence Prediction")

# --------------------
# Numeric input
# --------------------
age = st.number_input("Age", min_value=0, max_value=120, value=50)

# --------------------
# Helper function for one-hot encoding
# --------------------
def encode_one_hot(group_name, categories):
    """Create dropdown and return one-hot encoded dict for model input"""
    choice = st.selectbox(group_name, categories)
    encoded = {}
    for cat in categories:
        col_name = f"{group_name}_{cat}"
        encoded[col_name] = 1 if choice == cat else 0
    return encoded

# --------------------
# Categorical groups
# --------------------
gender_encoded = encode_one_hot("Gender", ["F","M"])

thyroid_encoded = encode_one_hot("Thyroid Function", [
    "Clinical Hyperthyroidism", "Clinical Hypothyroidism", "Euthyroid",
    "Subclinical Hyperthyroidism", "Subclinical Hypothyroidism"
])

physical_encoded = encode_one_hot("Physical Examination", [
    "Diffuse goiter", "Multinodular goiter", "Normal",
    "Single nodular goiter-left", "Single nodular goiter-right"
])

adenopathy_encoded = encode_one_hot("Adenopathy", ["Bilateral", "Extensive", "Left", "No", "Posterior", "Right"])

pathology_encoded = encode_one_hot("Pathology", ["Follicular", "Hurthel cell", "Micropapillary", "Papillary"])

focality_encoded = encode_one_hot("Focality", ["Multi-Focal", "Uni-Focal"])

risk_encoded = encode_one_hot("Risk", ["High", "Intermediate", "Low"])

t_encoded = encode_one_hot("T", ["T1a","T1b","T2","T3a","T3b","T4a","T4b"])

n_encoded = encode_one_hot("N", ["N0","N1a","N1b"])

m_encoded = encode_one_hot("M", ["M0","M1"])

stage_encoded = encode_one_hot("Stage", ["I","II","III","IVA","IVB"])

response_encoded = encode_one_hot("Response", ["Biochemical Incomplete","Excellent","Indeterminate","Structural Incomplete"])

# Smoking / History
smoking_encoded = {}
smoking_encoded["Smoking"] = 1 if st.selectbox("Smoking", ["No","Yes"])=="Yes" else 0
smoking_encoded["Hx Smoking"] = 1 if st.selectbox("History Smoking", ["No","Yes"])=="Yes" else 0
smoking_encoded["Hx Radiothreapy"] = 1 if st.selectbox("History Radiotherapy", ["No","Yes"])=="Yes" else 0

# --------------------
# Combine all inputs
# --------------------
input_dict = {"Age": age}
for d in [gender_encoded, thyroid_encoded, physical_encoded, adenopathy_encoded,
          pathology_encoded, focality_encoded, risk_encoded, t_encoded,
          n_encoded, m_encoded, stage_encoded, response_encoded, smoking_encoded]:
    input_dict.update(d)

X = pd.DataFrame([input_dict])

# --------------------
# Predict button
# --------------------
if st.button("Predict"):
    pred = model.predict(X)[0]
    if pred == 1:
        st.error("⚠️ High chance of cancer recurrence")
    else:
        st.success("✔ Low chance of cancer recurrence")

