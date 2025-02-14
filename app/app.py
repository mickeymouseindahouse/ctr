import os.path
import pickle
import sys
from io import BytesIO

import pandas as pd
import streamlit as st

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
    st.write("Uploaded Data Preview:")
    st.dataframe(df.head())

    try:
        print(uploaded_file)
        data_loader = TrainLoaderSessionSplitter(train_file=os.path.join(getroot(), "data/train_dataset_full.csv"),
                                                 test_data=df,
                                                 preprocessing=BaseModelPipeline(steps=[FillNaPreprocessor()]))
        data_loader.load_data()
        X_test = data_loader.test_data

        predictions = model.best_model.predict(X_test)
        df["Prediction"] = predictions

        st.subheader("Test Data with predictions")
        st.dataframe(df.head())

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
    except Exception as e:
        st.error(f"Error in making predictions: {e}")
