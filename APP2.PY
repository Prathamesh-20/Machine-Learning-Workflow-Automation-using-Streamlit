import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    silhouette_score, mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state.page = 1
if "df" not in st.session_state:
    st.session_state.df = None
if "features" not in st.session_state:
    st.session_state.features = None
if "target_column" not in st.session_state:
    st.session_state.target_column = None
if "algorithm" not in st.session_state:
    st.session_state.algorithm = None
if "task_type" not in st.session_state:
    st.session_state.task_type = None

# Page navigation buttons
def next_page():
    st.session_state.page += 1

def previous_page():
    st.session_state.page -= 1

# Step 1: Upload Dataset
if st.session_state.page == 1:
    st.title("Step 1: Upload Your Dataset 📂")
    uploaded_file = st.file_uploader("Upload your CSV file here", type=["csv"])
    example_dataset = st.selectbox("Or Use a Sample Dataset", ["None", "Iris", "Titanic"])

    if example_dataset == "Iris":
        from sklearn.datasets import load_iris
        iris = load_iris(as_frame=True)
        st.session_state.df = iris.frame
        st.write("Iris Dataset Preview:")
        st.dataframe(st.session_state.df)
    elif example_dataset == "Titanic":
        st.session_state.df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
        st.write("Titanic Dataset Preview:")
        st.dataframe(st.session_state.df)
    elif uploaded_file:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.write("Preview of the uploaded dataset:")
        st.dataframe(st.session_state.df)

    if st.session_state.df is not None and st.button("Next"):
        next_page()

# Step 2: Data Exploration
elif st.session_state.page == 2:
    st.title("Step 2: Data Exploration 🔍")
    df = st.session_state.df

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

    if selected_features:
        st.session_state.features = selected_features

    if st.button("Previous"):
        previous_page()
    if st.session_state.features and st.button("Next"):
        next_page()

# Step 3: Data Preprocessing
elif st.session_state.page == 3:
    st.title("Step 3: Data Preprocessing 🛠️")
    df = st.session_state.df[st.session_state.features]

    st.write("Choose preprocessing options:")
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

    if st.checkbox("Encode Categorical Variables"):
        label_encoders = {}
        for col in df.select_dtypes(include=["object"]).columns:
            label_encoders[col] = LabelEncoder()
            df[col] = label_encoders[col].fit_transform(df[col])
        st.write("Encoded Dataset Preview:")
        st.dataframe(df)

    if st.checkbox("Normalize/Standardize Data"):
        scaler = StandardScaler()
        numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        st.write("Scaled Dataset Preview:")
        st.dataframe(df)

    st.session_state.df = df

    if st.button("Previous"):
        previous_page()
    if st.button("Next"):
        next_page()

# Step 4: Select Algorithm
elif st.session_state.page == 4:
    st.title("Step 4: Select an Algorithm ⚙️")
    st.session_state.task_type = st.radio("What type of task are you performing?", ["Classification", "Regression", "Clustering"])

    if st.session_state.task_type == "Classification":
        st.session_state.algorithm = st.selectbox("Choose an algorithm", ["AutoML (Best Model)", "Random Forest", "SVM"])
    elif st.session_state.task_type == "Regression":
        st.session_state.algorithm = st.selectbox("Choose an algorithm", ["AutoML (Best Model)", "Random Forest Regressor", "Linear Regression"])
    elif st.session_state.task_type == "Clustering":
        st.session_state.algorithm = st.selectbox("Choose an algorithm", ["K-Means"])

    if st.button("Previous"):
        previous_page()
    if st.session_state.algorithm and st.button("Next"):
        next_page()

# Step 5: Train and Evaluate
elif st.session_state.page == 5:
    st.title("Step 5: Train and Evaluate 🚀")
    df = st.session_state.df
    st.session_state.target_column = st.selectbox("Select the target column", df.columns)
    features = df.drop(columns=[st.session_state.target_column])
    target = df[st.session_state.target_column]

    # Split data into train-test
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    st.write("Data split into 80% training and 20% testing.")

    algorithm = st.session_state.algorithm
    task_type = st.session_state.task_type

    model = None

    # Assign model based on algorithm
    if algorithm == "Random Forest":
        model = RandomForestClassifier(n_estimators=50, random_state=42)
    elif algorithm == "Random Forest Regressor":
        model = RandomForestRegressor(n_estimators=50, random_state=42)
    elif algorithm == "SVM":
        model = SVC(kernel="rbf", probability=True, random_state=42)
    elif algorithm == "Linear Regression":
        model = LinearRegression()
    elif algorithm == "K-Means":
        model = KMeans(n_clusters=3, random_state=42)

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

    else:
        st.error("No valid model selected. Please ensure all inputs are correct.")

    # Navigation buttons
    if st.button("Previous"):
        previous_page()
