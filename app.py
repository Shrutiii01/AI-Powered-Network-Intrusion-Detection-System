
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="AI-Powered NIDS",
    layout="wide"
)

st.title("AI-Based Network Intrusion Detection System")
st.markdown("""
This system uses **Machine Learning (Random Forest Algorithm)**  
to detect **malicious network traffic** in real time.

**Classes**
- 0 → Benign (Normal Traffic)
- 1 → Malicious (Attack Traffic)
""")


@st.cache_data
def load_data():
    """
    Generates synthetic network traffic data.
    This simulates CIC-IDS2017-like behavior.
    """

    np.random.seed(42)
    n_samples = 5000

    data = {
        "Destination_Port": np.random.randint(1, 65535, n_samples),
        "Flow_Duration": np.random.randint(10, 100000, n_samples),
        "Total_Fwd_Packets": np.random.randint(1, 100, n_samples),
        "Packet_Length_Mean": np.random.uniform(40, 1500, n_samples),
        "Active_Mean": np.random.uniform(1, 1000, n_samples),
        "Label": np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    }

    df = pd.DataFrame(data)

    
    attack_idx = df["Label"] == 1
    df.loc[attack_idx, "Total_Fwd_Packets"] += np.random.randint(50, 200, attack_idx.sum())
    df.loc[attack_idx, "Flow_Duration"] = np.random.randint(1, 1000, attack_idx.sum())

    return df


df = load_data()


st.sidebar.header("Model Controls")

train_size = st.sidebar.slider(
    "Training Data Percentage",
    min_value=50,
    max_value=90,
    value=80
)

n_estimators = st.sidebar.slider(
    "Number of Trees (Random Forest)",
    min_value=10,
    max_value=200,
    value=100
)


X = df.drop("Label", axis=1)
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=(100 - train_size) / 100,
    random_state=42
)


st.divider()
st.subheader("1. Model Training")

if st.button("Train Model"):
    with st.spinner("Training Random Forest Model..."):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42
        )
        model.fit(X_train, y_train)
        st.session_state["model"] = model
        st.success("Model trained successfully.")


st.divider()
st.subheader("2. Model Evaluation")

if "model" in st.session_state:
    model = st.session_state["model"]
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{acc * 100:.2f}%")
    c2.metric("Total Records", len(df))
    c3.metric("Detected Attacks", int(y_pred.sum()))

    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

else:
    st.warning("Train the model to see evaluation results.")


st.divider()
st.subheader("3. Live Traffic Analyzer")

st.markdown("Enter packet details to test whether traffic is malicious.")

col1, col2, col3, col4 = st.columns(4)

flow_duration = col1.number_input("Flow Duration", 0, 100000, 500)
total_packets = col2.number_input("Total Packets", 0, 500, 120)
packet_length = col3.number_input("Packet Length Mean", 0, 1500, 600)
active_mean = col4.number_input("Active Mean Time", 0, 1000, 100)

if st.button("Analyze Traffic"):
    if "model" not in st.session_state:
        st.error("Train the model first.")
    else:
        model = st.session_state["model"]

        input_data = np.array([[
            80,  # Destination Port (constant for demo)
            flow_duration,
            total_packets,
            packet_length,
            active_mean
        ]])

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error("🚨 MALICIOUS TRAFFIC DETECTED")
            st.write("Reason: High packet activity with abnormal duration.")
        else:
            st.success("✅ BENIGN TRAFFIC (Safe)")
