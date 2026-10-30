
import streamlit as st
import pandas as pd
import joblib
from io import BytesIO


st.set_page_config(
    page_title="Abstract Classifier",
    page_icon="📄",
    layout="wide"
)

st.title("Abstract Classification")
st.write(
    "Upload a CSV containing research abstracts to classify "
    "them into Business, Science, or Social Science."
)


@st.cache_resource
def load_models():
    tfidf = joblib.load("tfidf_vectorizer (1).joblib")
    top_7000_indices = joblib.load("top_7000_feature_indices.joblib")
    model = joblib.load("svm_top_7000.joblib")
    return tfidf, top_7000_indices, model


try:
    tfidf, top_7000_indices, model = load_models()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()


uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    st.subheader("Dataset Preview")
    st.write(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
    st.dataframe(df.head(10), use_container_width=True)

    # Select the abstract column
    text_column = st.selectbox(
        "Select the column containing abstracts",
        options=df.columns.tolist(),
        index=(
            df.columns.tolist().index("Abstract")
            if "Abstract" in df.columns
            else 0
        )
    )

    if st.button("Classify Abstracts", type="primary"):

        if len(df) == 0:
            st.warning("The uploaded CSV contains no rows.")

        else:
            with st.spinner("Classifying abstracts..."):
                try:
                    abstracts = (
                        df[text_column]
                        .fillna("")
                        .astype(str)
                        .tolist()
                    )

                    X_tfidf = tfidf.transform(abstracts)
                    X_selected = X_tfidf[:, top_7000_indices]

                    predictions = model.predict(X_selected)

                    results = df.copy()
                    results["Predicted_Category"] = predictions

                    st.session_state["classification_results"] = results

                    st.success(
                        f"Successfully classified {len(results):,} abstracts."
                    )

                except Exception as e:
                    st.error(f"Prediction failed: {e}")



if "classification_results" in st.session_state:

    results = st.session_state["classification_results"]

    st.subheader("Classification Results")

    # Category counts
    counts = results["Predicted_Category"].value_counts()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Abstracts", f"{len(results):,}")

    with col2:
        st.metric("Predicted Categories", len(counts))

    st.write("Category Distribution")
    st.bar_chart(counts)

    st.dataframe(results, use_container_width=True)

    
    csv_data = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Predictions CSV",
        data=csv_data,
        file_name="predictions.csv",
        mime="text/csv"
    )
