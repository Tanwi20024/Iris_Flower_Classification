
import streamlit as st
import numpy as np
import joblib

model  = joblib.load("iris_model.pkl")
scaler = joblib.load("iris_scaler.pkl")

species_names = ["Setosa", "Versicolor", "Virginica"]
species_emoji = ["🌼", "🌸", "🌺"]
species_info  = {
    "Setosa"    : "Small flowers with short, narrow petals.",
    "Versicolor": "Medium-sized flowers, moderate petal dimensions.",
    "Virginica" : "Largest flowers with long, wide petals."
}

st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸", layout="centered")

st.title("🌸 Iris Flower Classifier")
st.markdown("Adjust the measurements using the sliders, then click **Predict**.")
st.divider()

col1, col2 = st.columns(2)
with col1:
    st.subheader("🌿 Sepal")
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
    sepal_width  = st.slider("Sepal Width (cm)",  2.0, 4.5, 3.0, 0.1)
with col2:
    st.subheader("🌺 Petal")
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 3.7, 0.1)
    petal_width  = st.slider("Petal Width (cm)",  0.1, 2.5, 1.2, 0.1)

st.divider()

if st.button("🔍 Predict Species", use_container_width=True):
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    scaled = scaler.transform(sample)
    pred   = model.predict(scaled)[0]
    proba  = model.predict_proba(scaled)[0]

    st.success(f"## {species_emoji[pred]} Predicted: **{species_names[pred]}**")
    st.info(f"📖 {species_info[species_names[pred]]}")

    st.markdown("### 📊 Confidence Scores")
    for name, emoji, p in zip(species_names, species_emoji, proba):
        st.progress(float(p), text=f"{emoji} {name}: {p*100:.1f}%")

st.divider()
st.caption("Built with scikit-learn & Streamlit · Iris Dataset (UCI)")
