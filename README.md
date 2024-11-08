

# Heart Disease Prediction using Ensemble Learning

# Overview
The Heart Disease Prediction system is a machine learning-based web application designed to predict the likelihood of a person developing heart disease based on various health parameters. The prediction model leverages ensemble learning techniques such as Random Forest and XGBoost to provide accurate predictions.

The frontend of the application is built using Flask, which allows users to input health-related data and receive predictions regarding the likelihood of heart disease in real-time.

# Features
Predict Heart Disease: Users can input various health parameters (such as age, cholesterol level, blood pressure, etc.) to predict the likelihood of heart disease.

Random Forest & XGBoost Models: Ensemble learning algorithms (Random Forest and XGBoost) are used to build the prediction models for high accuracy and reliability.

Flask Web Interface: User-friendly web interface for inputting health data and displaying predictions.

Model Evaluation: Evaluation of models using metrics such as accuracy, precision, recall, and F1-score.

# Technology Stack
# Backend:

Python (for machine learning models and Flask backend)

Flask (for serving the web application)

Scikit-learn (for Random Forest model)

XGBoost (for the XGBoost model)

Pandas and NumPy (for data manipulation)

Matplotlib (for visualizations)

# Frontend:

HTML/CSS (for basic page structure and styling)

Bootstrap (for responsive web design)

# Installation

Python 3.x: Ensure you have Python installed.

Virtual Environment: 

To create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
Step 1: Clone the repository

Clone the repository to your local machine:

Step 2: Install dependencies

Install the required Python libraries listed in the requirements.txt file:


Step 3: Train the model 

If the model is not already pre-trained, you can train the Random Forest and XGBoost models using the dataset.

Run the following script to train the models and save them:

```bash
python train_model.py
```
This will train both the Random Forest and XGBoost models and save them as .pkl files, which will be used for making predictions in the Flask app.

Step 4: Run the Flask application

Once the dependencies are installed and the models are trained, you can run the Flask web application with:

```bash
python app.py
```
This will start a local development server, and you can access the application by navigating to http://127.0.0.1:5000 in your web browser.

# Usage
Open the web interface at http://127.0.0.1:5000.

Fill in the health parameters such as age, blood pressure, cholesterol levels, heart rate, etc.

Submit the form to get the prediction result: either No Heart Disease or Heart Disease.

View the prediction result and associated probabilities.

# Model Evaluation
The performance of both models is evaluated using various metrics such as accuracy, precision, recall, and F1-score.

Example Model Evaluation:
python
```bash
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Example of calculating metrics after predicting
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
```
