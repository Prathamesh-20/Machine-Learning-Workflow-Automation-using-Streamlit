import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, silhouette_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import time

# Set the page layout
st.set_page_config(page_title="Enhanced ML Web App", layout="wide")

# Title and Description
st.title("Enhanced Machine Learning Web App with AutoML and Advanced Features")
st.write("""
This application supports data exploration, preprocessing, model selection, and evaluation.
It also provides AutoML capabilities and interactive visualizations!
""")

# Step 1: Upload CSV File
st.header("1. Upload Your Dataset")
uploaded_file = st.file_uploader("Upload your CSV file here", type=["csv"])

# Sample Dataset Option
example_dataset = st.selectbox("Or Use a Sample Dataset", ["None", "Iris", "Titanic"])
if example_dataset == "Iris":
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    df = iris.frame
    st.write("Iris Dataset Preview:")
    st.dataframe(df)
elif example_dataset == "Titanic":
    df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
    st.write("Titanic Dataset Preview:")
    st.dataframe(df)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of the uploaded dataset:")
    st.dataframe(df)

# Step 2: Data Exploration
st.header("2. Data Exploration")
if 'df' in locals():
    st.subheader("Dataset Summary")
    st.write(df.describe())
    
    st.subheader("Data Types")
    st.write(df.dtypes)

    st.subheader("Correlation Heatmap")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Select Features for Training")
    selected_features = st.multiselect("Select features to include in the model", df.columns.tolist(), default=df.columns.tolist())

    # Reduce dataset to selected features
    df = df[selected_features]

# Step 3: Data Preprocessing
st.header("3. Data Preprocessing")
if 'df' in locals():
    st.write("Choose preprocessing options:")
    # Handle Missing Values
    missing_value_option = st.selectbox("Handle Missing Values", ["None", "Fill with Mean", "Fill with Median", "Drop Rows"])
    if missing_value_option != "None":
        if missing_value_option == "Fill with Mean":
            df = df.fillna(df.mean())
        elif missing_value_option == "Fill with Median":
            df = df.fillna(df.median())
        elif missing_value_option == "Drop Rows":
            df = df.dropna()
        st.write("Updated Dataset Preview:")
        st.dataframe(df)

    # Encode Categorical Variables
    if st.checkbox("Encode Categorical Variables"):
        label_encoders = {}
        for col in df.select_dtypes(include=["object"]).columns:
            label_encoders[col] = LabelEncoder()
            df[col] = label_encoders[col].fit_transform(df[col])
        st.write("Encoded Dataset Preview:")
        st.dataframe(df)

    # Normalize Data
    if st.checkbox("Normalize/Standardize Data"):
        scaler = StandardScaler()
        numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        st.write("Scaled Dataset Preview:")
        st.dataframe(df)

# Step 4: Select Algorithm
st.header("4. Select Algorithm")
task_type = st.radio("What type of task are you performing?", ["Classification", "Regression", "Clustering"])
algorithm = st.selectbox("Choose an algorithm", ["AutoML (Best Model)", "Random Forest", "SVM", "Linear Regression", "K-Means"])

# Step 5: Train and Evaluate
st.header("5. Train and Evaluate")
if 'df' in locals():
    target_column = st.selectbox("Select the target column", df.columns)
    features = df.drop(columns=[target_column])
    target = df[target_column]

    # Split data into train-test
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    st.write("Data split into 80% training and 20% testing.")

    if st.button("Start Training"):
        with st.spinner("Training the model..."):
            time.sleep(1)

        # AutoML Logic
        if algorithm == "AutoML (Best Model)":
            st.write("Running AutoML to find the best algorithm...")
            with st.spinner("AutoML in progress..."):
                time.sleep(1)
                # Define candidate models
                models = {
                    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
                    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
                }
                best_model = None
                best_accuracy = 0
                for name, candidate_model in models.items():
                    # Perform cross-validation to evaluate each model
                    scores = cross_val_score(candidate_model, X_train, y_train, cv=5)
                    accuracy = scores.mean()
                    st.write(f"{name} Cross-Validation Accuracy: {accuracy:.2f}")
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = candidate_model
                        best_model_name = name
                model = best_model
                st.success(f"The best model selected by AutoML is **{best_model_name}** with a cross-validation accuracy of **{best_accuracy:.2f}**.")
        else:
            # Define the model manually based on selected algorithm
            if algorithm == "Random Forest":
                model = RandomForestClassifier(n_estimators=50, random_state=42)
            elif algorithm == "SVM":
                model = SVC(kernel="rbf", probability=True, random_state=42)
            elif algorithm == "Linear Regression":
                model = LinearRegression()
            elif algorithm == "K-Means":
                model = KMeans(n_clusters=3, random_state=42)
            else:
                st.error("No valid algorithm selected.")
                model = None

        # Train the model
        if model is not None:
            model.fit(X_train, y_train)
            st.success("Training Complete!")

            # Evaluate Results
            st.header("6. Results")
            if algorithm in ["Random Forest", "SVM", "Linear Regression"]:
                predictions = model.predict(X_test)
                if task_type == "Classification":
                    accuracy = accuracy_score(y_test, predictions)
                    st.write(f"Model Accuracy: {accuracy:.2f}")
                    cm = confusion_matrix(y_test, predictions)
                    st.write("Confusion Matrix:")
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                    st.pyplot(fig)
                    st.write("Classification Report:")
                    st.text(classification_report(y_test, predictions))
                elif task_type == "Regression":
                    st.write("Regression Metrics are not yet implemented.")
            elif algorithm == "K-Means":
                labels = model.predict(features)
                silhouette_avg = silhouette_score(features, labels)
                st.write(f"Silhouette Score: {silhouette_avg:.2f}")
