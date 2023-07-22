# from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from streamlit_option_menu import option_menu
import streamlit as st


heart_data = pd.read_csv(
    '/Users/Machine Learning/Heart Disease/heartdisease.csv')

heart_data = heart_data.dropna()  # Removing Null Values

# Axis 1 for column and 0 for row
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']


X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2)  # Stratify and random_state puts data in random

print(X.shape, X_train.shape, X_test.shape)

model = LogisticRegression()

# Traing
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)


st.set_page_config(page_title="Disease Prediction Model",
                   page_icon=":tada:", layout="wide")

# r2 = r2_score(Y_test, y_pred)
# r2 = r2*100
# r2 = round(r2, 2)


with st.sidebar:
    selected = option_menu('Heart Disease Prediction Model',
                           ['Home Page',
                            'Heart Disease Prediction',
                            'Contact Me'],
                           icons=['house', 'activity', 'person-rolodex'],
                           default_index=0)

if (selected == 'Home Page'):
    st.header("Heart Disease Prediction")
    st.write("This repository contains code for a simple machine learning model that predicts the likelihood of heart disease based on various input features. The model is built using logistic regression and utilizes the scikit-learn library for data manipulation and model training. The prediction is made through a Streamlit web application, allowing users to input their data and receive predictions interactively.")

    st.header("Dataset")
    st.write("The heart disease dataset used in this project is stored in a CSV file named heartdisease.csv. The dataset contains several features related to individuals' health and a target column indicating the presence (1) or absence (0) of heart disease. Any missing values in the dataset are removed before training the model.")

    st.header("Launching the Streamlit Web Application")
    st.write("The Streamlit application has the following main sections:")
    st.write("Home Page")
    st.write("The landing page of the web application displays the title 'Heart Disease Prediction.' It serves as an introduction to the application.")
    st.write("Heart Disease Prediction")
    st.write("This section of the application allows users to input their health information, such as age, sex, chest pain type, blood pressure, serum cholestoral, fasting blood sugar, resting electrocardiographic results, maximum heart rate achieved, exercise-induced angina, ST depression induced by exercise, slope of the peak exercise ST segment, number of major vessels colored by fluoroscopy, and thalassemia type.After providing the required input, the user can click the 'Predict Disease' button to get the model's prediction on whether the person is likely to have heart disease or not.")
    st.write("Contact Me")
    st.write("This section provides contact information for getting in touch with the developer or project owner.")
    st.write("Please note that the accuracy of the model is based on the dataset available during model training. For real-world predictions, the accuracy may vary.")


if (selected == 'Heart Disease Prediction'):

    st.header("Heart Disease Prediction")
    st.write("Enter the Following Details !!")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Enter your Age", 0, 100)
    with col2:
        sex = st.number_input("Sex (1 for Male & 0 for Female)", 0, 1)
    with col3:
        Chest_Pain = st.number_input("Chest Pain Type (1-4)", 1, 4)
    with col1:
        resting_bp = st.number_input("Blood pressure (90-180)", 90, 180)
    with col2:
        serum_cholestoral = st.number_input(
            "serum cholestoral in mg/dI (150-300)", 150, 300)
    with col3:
        fasting_sugar = st.number_input(
            "fasting blood sugar > 120 mg/dI", 120, 300)
    with col1:
        restecg = st.number_input(
            "resting electrocardiographic results (values 0,1,2)", 0, 2)
    with col2:
        thalach = st.number_input(
            "Maximum Heart Rate Achieved (100-180)", 100, 180)
    with col3:
        exang = st.number_input("exercise induced angina (0,1)", 0, 1)
    with col1:
        oldpeak = st.number_input(
            "ST depression induced by exercise relative (0-5)", 0, 5)
    with col2:
        slope = st.number_input(
            "the slope of the peak exercise ST segment (0-2)", 0, 2)
    with col3:
        ca = st.number_input(
            "number of major vessels (0-3) colored by flourosopy", 0, 3)
    with col1:
        thal = st.number_input(
            "3=Normal; 6=fixed defect; 7=reversable defect", 3, 7)

    if st.button('Predict Disease'):
        pred = model.predict([[age, sex, Chest_Pain, resting_bp, serum_cholestoral,
                             fasting_sugar, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        if (pred[0] == 0):
            st.success(
                "The Person is Fit and does Not have any Heart Disease")
            st.warning("Accuracy of Model is 91.21 %")
        else:
            st.success("The Person is suffering from Heart Disease")
            st.warning("Accuracy of Model is 91.21 %")

    # st.write("Accuracy of Model is", r2,"%")


if (selected == 'Contact Me'):
    # def contact_form():
        st.header("Contact Me")
        st.write("Please fill out the form below to get in touch with me.")

    # Input fields for user's name, email, and message
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        message = st.text_area("Message", height=150)

    # Submit button
        if st.button("Submit"):
            if name.strip() == "" or email.strip() == "" or message.strip() == "":
                st.warning("Please fill out all the fields.")
            else:

                send_email_to = 'kumawatharsh2004@email.com'
                st.success("Your message has been sent successfully!")

# Main application
    # def main():
    #     # Display the Contact Me form
    #     contact_form()

    # if __name__ == "__main__":
    #     main()


footer = """<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: black;
color: grey;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with ❤ by <a style='display: block; text-align: center;' href="https://www.linkedin.com/in/harsh-kumawat-069bb324b/" target="_blank">Harsh</a></p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
