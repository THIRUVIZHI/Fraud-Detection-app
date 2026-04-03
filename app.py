import streamlit as st
import joblib
import numpy as np

# Page settings
st.set_page_config(page_title="Fraud Detection", layout="centered")

st.title("💳 Online Payment Fraud Detection")
st.write("Enter 30 transaction values (comma-separated) to check fraud")

# Load model safely
try:
    model = joblib.load('fraud_model.pkl')
    st.success("Model loaded successfully ✅")
except Exception as e:
    st.error(f"Error loading model ❌: {e}")
    st.stop()

# Input box
input_data = st.text_area("Enter 30 values here:")

# Predict button
if st.button("Predict"):
    try:
        # Convert input string to list
        data = list(map(float, input_data.strip().split(',')))

        # Check input length
        if len(data) != 30:
            st.warning("⚠️ Please enter exactly 30 values")
        else:
            data = np.array(data).reshape(1, -1)

            # Prediction
            prediction = model.predict(data)
            probability = model.predict_proba(data)

            st.subheader("Result:")

            if prediction[0] == 1:
                st.error("⚠️ Fraud Transaction")
            else:
                st.success("✅ Normal Transaction")

            # Show probability
            st.write(f"Fraud Probability: {probability[0][1]:.4f}")

            # Chart (for better demo)
            st.bar_chart([probability[0][0], probability[0][1]])

    except ValueError:
        st.error("❌ Please enter valid numeric values (comma-separated)")
    except Exception as e:
        st.error(f"Unexpected error: {e}")