# Heart-disease-Prediction ❤️

This repository contains code for a machine learning model that predicts the likelihood of heart disease based on various health-related features. The model is built using logistic regression and is implemented as an interactive web application using Streamlit. The app allows users to input their health information and receive a prediction regarding the presence or absence of heart disease.

## 📌Sneak Peek of Page :

## Dataset 📊:
The dataset used in this project is stored in a CSV file named `heartdisease.csv`. It contains several features related to individuals' health, such as age, sex, chest pain type, blood pressure, serum cholestoral, fasting blood sugar, resting electrocardiographic results, maximum heart rate achieved, exercise-induced angina, ST depression induced by exercise, slope of the peak exercise ST segment, number of major vessels colored by fluoroscopy, and thalassemia type. Any missing values in the dataset are removed before training the model.

## Dependencies 🔧:
To run this code, you need the following dependencies:

Python 3.x

NumPy

pandas

scikit-learn

Streamlit

You can install the required packages using the following command:
`pip install numpy pandas scikit-learn streamlit`

## Model Training and Evaluation 📈:
The model is trained using logistic regression, and the dataset is split into training and testing sets using the `train_test_split` function from scikit-learn. The model's accuracy is evaluated using the R-squared metric and displayed on the web application.

## Streamlit Web Application  💻:
The Streamlit application consists of three main sections accessible from the sidebar:

### Home Page 🏠:
The landing page of the web application welcomes users to the "Heart Disease Prediction" section.

### Heart Disease Prediction 💓:
In this section, users can input their health details such as age, sex, chest pain type, blood pressure, serum cholestoral, fasting blood sugar, resting electrocardiographic results, maximum heart rate achieved, exercise-induced angina, ST depression induced by exercise, slope of the peak exercise ST segment, number of major vessels colored by fluoroscopy, and thalassemia type.

After providing the necessary inputs, users can click the "Predict Disease" button to get the model's prediction. If the prediction indicates that the person is fit and does not have any heart disease, a success message with the accuracy of the model (91.21%) is displayed. Otherwise, if the person is predicted to be suffering from heart disease, a relevant message is shown along with the model's accuracy.

## License
This project is licensed under the MIT License. You are free to use, modify, and distribute the code for personal and commercial purposes. 📜🆓

## Acknowledgements
I would like to express my gratitude to the open-source community for providing invaluable resources and inspiration for this project.🌟

## 📬 Contact Me
If you want to contact me, you can reach me through the below handles.

&nbsp;&nbsp;<a href="https://www.linkedin.com/in/harsh-kumawat-069bb324b/"><img src="https://www.felberpr.com/wp-content/uploads/linkedin-logo.png" width="30"></img></a>

© 2023 Harsh
