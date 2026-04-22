import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ---------- Page setup ----------
st.set_page_config(page_title="Breast Cancer Regression Dashboard", layout="wide")
st.title("🩺 Breast Cancer Survival Predictor")
st.caption("SLR & MLR dashboard — adjust inputs, watch predictions update live.")

# ---------- Load data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("Breast_Cancer.csv")
    df.columns = df.columns.str.strip()
    # handle the typo in the original dataset
    if "Reginol Node Positive" in df.columns:
        df = df.rename(columns={"Reginol Node Positive": "Regional Node Positive"})
    # encode categorical columns we need
    df["Estrogen_Pos"] = (df["Estrogen Status"] == "Positive").astype(int)
    df["Progesterone_Pos"] = (df["Progesterone Status"] == "Positive").astype(int)
    df["Grade_num"] = pd.to_numeric(df["Grade"], errors="coerce").fillna(2)
    return df

df = load_data()

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Controls")
view = st.sidebar.radio("View", ["📊 Data Analysis", "📈 SLR", "🧠 MLR"])
st.sidebar.markdown("---")
st.sidebar.write(f"**Patients loaded:** {len(df)}")

# ============================================================
# 1. DATA ANALYSIS VIEW
# ============================================================
if view == "📊 Data Analysis":
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Patients", len(df))
        st.metric("Alive", (df["Status"] == "Alive").sum())
    with col2:
        st.metric("Dead", (df["Status"] == "Dead").sum())
        st.metric("Avg Survival (months)", round(df["Survival Months"].mean(), 1))

    st.subheader("Age Distribution")
    fig, ax = plt.subplots()
    ax.hist(df["Age"], bins=20, color="#2E86AB", edgecolor="white")
    ax.set_xlabel("Age")
    ax.set_ylabel("Patients")
    st.pyplot(fig)

# ============================================================
# 2. SIMPLE LINEAR REGRESSION (SLR)
# ============================================================
elif view == "📈 SLR":
    st.subheader("Simple Linear Regression")
    st.write("Pick **one** feature to predict survival months.")

    feature_options = {
        "Age": "Age",
        "Tumor Size": "Tumor Size",
        "Positive Nodes": "Regional Node Positive",
        "Grade": "Grade_num",
    }
    choice = st.selectbox("Choose predictor", list(feature_options.keys()))
    col_name = feature_options[choice]

    X = df[[col_name]]
    y = df["Survival Months"]

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    # Input slider
    x_min, x_max = float(X[col_name].min()), float(X[col_name].max())
    user_val = st.slider(f"Enter {choice}", x_min, x_max, float(X[col_name].mean()))
    prediction = model.predict([[user_val]])[0]

    # Show metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Survival", f"{prediction:.1f} months")
    c2.metric("R² Score", f"{r2:.3f}")
    c3.metric("Slope", f"{model.coef_[0]:.3f}")

    st.write(f"**Equation:** ŷ = {model.intercept_:.2f} + ({model.coef_[0]:.3f}) × {choice}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X, y, alpha=0.3, color="#A23B72", label="Patients")
    ax.plot(X, y_pred, color="#2E86AB", linewidth=2, label="Regression line")
    ax.scatter([user_val], [prediction], color="orange", s=200, zorder=5,
               edgecolor="black", label="Your prediction")
    ax.set_xlabel(choice)
    ax.set_ylabel("Survival Months")
    ax.legend()
    st.pyplot(fig)

# ============================================================
# 3. MULTIPLE LINEAR REGRESSION (MLR)
# ============================================================
elif view == "🧠 MLR":
    st.subheader("Multiple Linear Regression")
    st.write("All features combined to predict survival months.")

    features = ["Age", "Tumor Size", "Regional Node Positive",
                "Regional Node Examined", "Grade_num",
                "Estrogen_Pos", "Progesterone_Pos"]
    X = df[features]
    y = df["Survival Months"]

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    st.markdown("### 👤 Patient Profile")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 20, 90, 55)
        tumor = st.slider("Tumor Size (mm)", 1, 140, 25)
        nodes_pos = st.slider("Positive Nodes", 0, 40, 2)
        nodes_exam = st.slider("Nodes Examined", 1, 60, 10)
    with col2:
        grade = st.slider("Grade (1-4)", 1, 4, 2)
        er = st.radio("Estrogen Receptor +", ["Yes", "No"]) == "Yes"
        pr = st.radio("Progesterone Receptor +", ["Yes", "No"]) == "Yes"

    user_input = [[age, tumor, nodes_pos, nodes_exam, grade, int(er), int(pr)]]
    prediction = model.predict(user_input)[0]

    c1, c2 = st.columns(2)
    c1.metric("Predicted Survival", f"{prediction:.1f} months")
    c2.metric("R² Score", f"{r2:.3f}")

    # Predicted vs Actual plot
    st.subheader("Predicted vs Actual Survival")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y, y_pred, alpha=0.3, color="#A23B72", label="Patients")
    ax.plot([y.min(), y.max()], [y.min(), y.max()],
            color="#2E86AB", linestyle="--", label="Perfect fit")
    ax.scatter([prediction], [prediction], color="orange", s=200,
               zorder=5, edgecolor="black", label="Your patient")
    ax.set_xlabel("Actual Survival Months")
    ax.set_ylabel("Predicted Survival Months")
    ax.legend()
    st.pyplot(fig)

    # Feature importance bar chart
    st.subheader("Feature Coefficients")
    coef_df = pd.DataFrame({"Feature": features, "Coefficient": model.coef_})
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    colors = ["#2E86AB" if c >= 0 else "#E63946" for c in coef_df["Coefficient"]]
    ax2.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Effect on Survival Months")
    st.pyplot(fig2)

st.markdown("---")
st.caption("Built with Streamlit • scikit-learn • matplotlib")
