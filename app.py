import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, silhouette_score, mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
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

# Dynamically display algorithms based on task type
if task_type == "Classification":
    algorithm = st.selectbox("Choose an algorithm", ["AutoML (Best Model)", "Random Forest", "SVM"])
elif task_type == "Regression":
    algorithm = st.selectbox("Choose an algorithm", ["AutoML (Best Model)", "Random Forest Regressor", "Linear Regression"])
elif task_type == "Clustering":
    algorithm = st.selectbox("Choose an algorithm", ["K-Means"])

# Add hyperparameters based on the selected algorithm
if algorithm in ["Random Forest", "Random Forest Regressor"]:
    n_estimators = st.slider("Number of Trees", 10, 200, 50)
    max_depth = st.slider("Max Depth of Tree", 1, 20, 5)
elif algorithm == "SVM":
    C = st.slider("Regularization (C)", 0.01, 10.0, 1.0)
    kernel = st.selectbox("Kernel Type", ["linear", "rbf", "poly"])
elif algorithm == "K-Means":
    n_clusters = st.slider("Number of Clusters", 2, 10, 3)

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
                models = {
                    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
                    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
                }
                best_model = None
                best_accuracy = 0
                for name, candidate_model in models.items():
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
            if algorithm == "Random Forest":
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            elif algorithm == "SVM":
                model = SVC(C=C, kernel=kernel, probability=True, random_state=42)
            elif algorithm == "Linear Regression":
                model = LinearRegression()
            elif algorithm == "K-Means":
                model = KMeans(n_clusters=n_clusters, random_state=42)

        # Train the model
        if model is not None:
            model.fit(X_train, y_train)
            st.success("Training Complete!")

            # Evaluate Results
            st.header("6. Results")
            if task_type == "Classification":
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                st.write(f"Model Accuracy: {accuracy:.2f}")
                cm = confusion_matrix(y_test, predictions)
                st.write("Confusion Matrix:")
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)

                st.write("Classification Report:")
                st.text(classification_report(y_test, predictions))

                st.subheader("Understanding the Classification Report")
                st.markdown("""
                - **Precision**: Proportion of correctly predicted positive observations to total predicted positives.
                - **Recall**: Proportion of correctly predicted positive observations to all actual positives.
                - **F1-Score**: Harmonic mean of Precision and Recall.
                - **Support**: Number of actual occurrences for each class in the dataset.
                """)

            elif task_type == "Regression":
                predictions = model.predict(X_test)
                mse = mean_squared_error(y_test, predictions)
                mae = mean_absolute_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                st.write(f"Mean Squared Error (MSE): {mse:.2f}")
                st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
                st.write(f"R-squared (R2): {r2:.2f}")

            elif task_type == "Clustering":
                labels = model.predict(features)
                silhouette_avg = silhouette_score(features, labels)
                st.write(f"Silhouette Score: {silhouette_avg:.2f}")

                # Perform PCA for 2D visualization
                if features.shape[1] > 1:  # Ensure there are enough features for PCA
                    pca = PCA(2)
                    cluster_data = pca.fit_transform(features)

                    # Plot Clusters
                    fig, ax = plt.subplots(figsize=(8, 6))
                    scatter = ax.scatter(cluster_data[:, 0], cluster_data[:, 1], c=labels, cmap="viridis", alpha=0.8)
                    centroids = model.cluster_centers_
                    centroids_2d = pca.transform(centroids)  # Transform centroids to 2D
                    ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c="red", marker="X", s=200, label="Centroids")
                    ax.legend(loc="best")
                    plt.title("Cluster Visualization")
                    plt.xlabel("PCA Component 1")
                    plt.ylabel("PCA Component 2")
                    st.pyplot(fig)
                else:
                    st.warning("PCA visualization skipped because the dataset does not have enough features.")

                # Insights
                st.subheader("Clustering Insights")
                st.markdown("""
                - **Silhouette Score** measures the quality of clustering. It ranges from -1 to 1:
                  - A score close to 1 indicates well-defined clusters.
                  - A score near 0 indicates overlapping clusters.
                  - Negative values suggest clusters assigned to the wrong data points.
                - **Visualization**: Each cluster is displayed in a 2D plot with its centroid marked.
                - Consider increasing or decreasing the number of clusters to optimize the score.
                """)

# Footer
st.write("---")
st.write("Developed by Prathamesh Sawle. All rights reserved.")
