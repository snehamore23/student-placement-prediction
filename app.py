import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Placement Prediction", layout="wide")

st.title("🎓 Student Placement Prediction App")
st.write("Fill in the details below to predict the placement status.")

# 1. Load the combined file (which contains the Scaler and Model)
try:
    with open("model_and_preprocessors.pkl", "rb") as file:
        data = pickle.load(file)
        model = data['model']
        scaler = data['scaler']
except FileNotFoundError:
    st.error("Error: 'model_and_preprocessors.pkl' not found.")
    st.stop()

# 2. Organize inputs into a form to prevent unnecessary re-runs
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        student_id = st.number_input("Student ID", min_value=1, value=1, help="Enter a unique student identifier")
        age = st.number_input("Age", min_value=18, max_value=100, value=20)
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.01)
        internships = st.number_input("Internships", min_value=0, max_value=20, value=0)
        projects = st.number_input("Projects", min_value=0, max_value=50, value=0)
        coding_skills = st.number_input("Coding Skills Rating (1-10)", min_value=1, max_value=10, value=5)

    with col2:
        comm_skills = st.number_input("Communication Skills Rating (1-10)", min_value=1, max_value=10, value=5)
        aptitude_score = st.number_input("Aptitude Test Score (0-100)", min_value=0, max_value=100, value=50)
        soft_skills = st.number_input("Soft Skills Rating (1-10)", min_value=1, max_value=10, value=5)
        certifications = st.number_input("Certifications", min_value=0, max_value=20, value=0)
        backlogs = st.number_input("Backlogs", min_value=0, max_value=20, value=0)

    st.divider()
    st.subheader("Categorical Details")
    gen_col, deg_col, br_col = st.columns(3)

    with gen_col:
        gender = st.selectbox("Gender", ["Female", "Male"])
    with deg_col:
        degree = st.selectbox("Degree", ["B.Sc", "B.Tech", "BCA", "MCA"])
    with br_col:
        branch = st.selectbox("Branch", ["CSE", "Civil", "ECE", "IT", "ME"])

    # 3. Handle Prediction on Submit
    submit_button = st.form_submit_button("Predict Placement Status", type="primary")

if submit_button:
    # Handle One-Hot Encoding
    gender_male = 1 if gender == "Male" else 0
    
    degree_btech = 1 if degree == "B.Tech" else 0
    degree_bca = 1 if degree == "BCA" else 0
    degree_mca = 1 if degree == "MCA" else 0
    
    branch_civil = 1 if branch == "Civil" else 0
    branch_ece = 1 if branch == "ECE" else 0
    branch_it = 1 if branch == "IT" else 0
    branch_me = 1 if branch == "ME" else 0

    # Combine all 19 features in the EXACT order the scaler expects
    features = [
        student_id, age, cgpa, internships, projects, 
        coding_skills, comm_skills, aptitude_score, 
        soft_skills, certifications, backlogs,
        gender_male, degree_btech, degree_bca, degree_mca,
        branch_civil, branch_ece, branch_it, branch_me
    ]

    # Convert to array and scale
    input_array = np.array([features])
    
    try:
        scaled_features = scaler.transform(input_array)
        prediction = model.predict(scaled_features)

        if prediction[0] == 1:
            st.success("🎉 **Status: PLACED**")
            st.balloons()
        else:
            st.warning("⚠️ **Status: NOT PLACED**")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
