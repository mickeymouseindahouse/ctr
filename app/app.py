import os.path
import sys
from io import BytesIO

import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from data_loader.train_loader_session_splitter import TrainLoaderSessionSplitter
from preprocessor.fill_na_preprocessor import FillNaPreprocessor
from pipeline.base_model_pipeline import BaseModelPipeline
from constants import getroot


@st.cache_resource
def load_model():
    return BaseModelPipeline.load_pickle(os.path.join(getroot(), "results/roc_submission/rocauc.pkl"))

st.title("User clicks prediction provided by the Clickers' Clique team")

model = load_model()


uploaded_file = st.file_uploader("Upload a CSV file with the test data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=["DateTime"])

    try:
        print(uploaded_file)
        data_loader = TrainLoaderSessionSplitter(train_file=os.path.join(getroot(), "data/train_dataset_full.csv"),
                                                 test_data=df,
                                                 preprocessing=BaseModelPipeline(steps=[FillNaPreprocessor()]))
        data_loader.load_data()
        X_test = data_loader.test_data

        predictions = model.best_model.predict_proba(X_test)[:, 1]
        df["Click Probability Predictions"] = predictions
        df["Clicks"] = model.best_model.predict(X_test)

        csv_buffer = BytesIO()
        pd.DataFrame(predictions).to_csv(csv_buffer, index=None, header=None)
        csv_buffer.seek(0)

        csv_buffer.seek(0)

        st.download_button(
            label="Download Predictions (only) as CSV with no header and index",
            data=csv_buffer,
            file_name="predictions.csv",
            mime="text/csv"
        )



        st.subheader("🔍 Test Data with predicted click probabilities preview")
        st.dataframe(df)

        if "product" in df.columns and "Clicks" in df.columns:
            product_ctr = df.groupby("product")["Clicks"].agg(Impressions="count", Clicks="sum").reset_index()
            product_ctr["CTR (%)"] = (product_ctr["Clicks"] / product_ctr["Impressions"]) * 100  # CTR calculation

            session_ctr = df.groupby("webpage_id")["Clicks"].agg(Impressions="count", Clicks="sum").reset_index()
            session_ctr["CTR (%)"] = (session_ctr["Clicks"] / session_ctr["Impressions"]) * 100  # CTR calculation

            st.write("### 📊 Predicted Click-Through Rate (CTR) Per Product & Per Web Page")

            col1, col2 = st.columns(2)

            with col1:
                st.write("#### 📌 CTR by Product")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x="product", y="CTR (%)", data=product_ctr, palette="coolwarm", ax=ax)
                ax.set_title("CTR for Products", fontsize=12)
                ax.set_xlabel("Product", fontsize=10)
                ax.set_ylabel("CTR (%)", fontsize=10)
                st.pyplot(fig)

            with col2:
                st.write("#### 📌 CTR by WebPage")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x="webpage_id", y="CTR (%)", data=session_ctr, palette="viridis", ax=ax)
                ax.set_title("CTR for Web Pages", fontsize=12)
                ax.set_xlabel("WebPage Id", fontsize=10)
                ax.set_ylabel("CTR (%)", fontsize=10)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                st.pyplot(fig)

    except Exception as e:
            st.error(f"Error in making predictions: {e}")
