import streamlit as st
import pandas as pd
import joblib

# Load models and scaler
xgb_diabetes = joblib.load('xgb_diabetes_model.pkl')
xgb_obesity = joblib.load('xgb_obesity_model.pkl')
xgb_heart = joblib.load('xgb_heart_model.pkl')
xgb_htn = joblib.load('xgb_htn_model.pkl')  # Replaced RF with XGBoost HTN
scaler = joblib.load('shared_scaler.pkl')

def predict_risks(user_input_df):
    # Scale numeric features
    user_input_df[['age_group_code', 'income_level']] = scaler.transform(
        user_input_df[['age_group_code', 'income_level']]
    )

    # Predict probabilities
    diabetes_prob = xgb_diabetes.predict_proba(user_input_df)[:, 1][0]
    obesity_prob = xgb_obesity.predict_proba(user_input_df)[:, 1][0]
    heart_prob = xgb_heart.predict_proba(user_input_df)[:, 1][0]
    htn_prob = xgb_htn.predict_proba(user_input_df)[:, 1][0]

    return {
        'Diabetes Risk': round(diabetes_prob * 100, 2),
        'Obesity Risk': round(obesity_prob * 100, 2),
        'Heart Disease Risk': round(heart_prob * 100, 2),
        'Hypertension Risk': round(htn_prob * 100, 2)
    }

# Streamlit UI
st.title("🧬 Chronic Disease Risk Predictor")
st.write("Enter your details below to estimate your health risks.")

# Input form
with st.form("risk_form"):
    # Label → Code maps
    age_group_map = {
        "18–24": 1, "25–29": 2, "30–34": 3, "35–39": 4,
        "40–44": 5, "45–49": 6, "50–54": 7, "55–59": 8,
        "60–64": 9, "65–69": 10, "70–74": 11, "75–79": 12, "80+": 13
    }

    sex_map = {"Male": 1, "Female": 2}

    race_map = {
        "White": 1, "Black or African American": 2, "Asian": 3,
        "American Indian/Alaska Native": 4, "Hispanic": 5,
        "Other / Multiracial": 6, "Unknown / Not Reported": 7
    }

    education_map = {
        "Never attended school": 1, "Elementary school": 2, "High school": 3,
        "Some college": 4, "College graduate": 5, "Post-graduate": 6
    }

    income_map = {
        "Less than $10,000": 1, "$10,000–$15,000": 2, "$15,000–$20,000": 3,
        "$20,000–$25,000": 4, "$25,000–$35,000": 5, "$35,000–$50,000": 6,
        "$50,000–$75,000": 7, "More than $75,000": 8
    }

    marital_map = {
        "Married": 1, "Divorced": 2, "Widowed": 3,
        "Separated": 4, "Never married": 5,
        "Unmarried partner": 6, "Other": 7
    }

    # Inputs
    age_group = st.selectbox("Age Group", list(age_group_map.keys()))
    sex = st.selectbox("Sex", list(sex_map.keys()))
    race_group = st.selectbox("Race", list(race_map.keys()))
    education_level = st.selectbox("Education Level", list(education_map.keys()))
    income_level = st.selectbox("Income Level", list(income_map.keys()))
    marital_status = st.selectbox("Marital Status", list(marital_map.keys()))
    ever_smoked = st.selectbox("Ever Smoked 100+ Cigarettes?", ["Yes", "No"])
    phys_activity = st.selectbox("Any Physical Activity in Last 30 Days?", ["Yes", "No"])
    any_alcohol = st.selectbox("Do You Consume Alcohol?", ["Yes", "No"])

    submitted = st.form_submit_button("Predict My Risk")

    if submitted:
        input_df = pd.DataFrame([{
            'age_group_code': age_group_map[age_group],
            'sex': sex_map[sex],
            'race_group': race_map[race_group],
            'education_level': education_map[education_level],
            'income_level': income_map[income_level],
            'marital_status': marital_map[marital_status],
            'ever_smoked': 1 if ever_smoked == "Yes" else 2,
            'phys_activity': 1 if phys_activity == "Yes" else 2,
            'any_alcohol': 1 if any_alcohol == "Yes" else 2
        }])

        results = predict_risks(input_df)

        st.subheader("📊 Your Predicted Risk Scores:")
        for condition, score in results.items():
            st.write(f"{condition}: **{score}%**")

        st.markdown("""
        > 🧠 **Note:** This tool provides a personalized **risk estimate** based on your profile — not the exact timing of disease onset. 
        > It is not a diagnosis. For medical guidance, please consult a healthcare professional.
        """)
