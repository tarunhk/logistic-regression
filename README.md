# 🤖 Logistic Regression – Binary Classification

## 📌 Project Description
This project demonstrates how to build a **binary classification model** using Logistic Regression.  
The model is trained on a breast cancer dataset to classify whether a tumor is malignant or benign.

Logistic Regression is commonly used for classification problems where the output has two classes.

---

## 🎯 Objective
The objective of this task is to:

- Load and preprocess dataset  
- Split data into training and testing sets  
- Train Logistic Regression model  
- Evaluate model performance  
- Visualize ROC curve  

---

## 🛠 Tools & Libraries Used
- Python 3  
- Pandas  
- NumPy  
- Matplotlib  
- Scikit-learn  

---

## ▶️ How to Run the Program
### Step 1: Install required libraries
```bash
pip3 install pandas numpy matplotlib scikit-learn
```
### Step 2: Run the program
Open terminal in the project folder and run:
```bash
python3 logistic_regression.py
```

---

## ⚙️ What the Program Does

The script performs the following:

- Loads the dataset
- Cleans and preprocesses data
- Splits into training and testing sets
- Trains Logistic Regression model
- Predicts class labels
- Evaluates model using:
 - 1. Confusion Matrix
 2. Precision
 3. Recall
 4. ROC-AUC Score
- Plots ROC curve

## 📊 Output

The program displays:

- Confusion Matrix
- Precision score
- Recall score
- ROC-AUC score
- ROC curve graph

These metrics help evaluate classification performance.
