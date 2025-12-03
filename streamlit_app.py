import os
from io import StringIO
from time import time
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st

from DeepTemporalClustering import DTC
from datasets import all_ucr_datasets, load_data


@st.cache_data(show_spinner=False)
def load_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and cache a UCR/UEA dataset by name."""
    return load_data(dataset_name)


@st.cache_data(show_spinner=False)
def load_uploaded_dataframe(file_bytes: bytes) -> pd.DataFrame:
    """Read an uploaded CSV file into a DataFrame and cache the result."""
    return pd.read_csv(StringIO(file_bytes.decode("utf-8")))


def prepare_uploaded_dataset(
    df: pd.DataFrame, feature_columns: Tuple[str, ...], label_column: str | None
) -> Tuple[np.ndarray, np.ndarray | None]:
    """Convert a DataFrame into model-ready arrays."""

    feature_values = df.loc[:, feature_columns].to_numpy(dtype=np.float32)
    X = np.expand_dims(feature_values, -1)

    if label_column:
        y = df[label_column].to_numpy()
    else:
        y = None

    return X, y


def run_training(X_train: np.ndarray,
                y_train: np.ndarray | None,
                n_clusters: int,
                n_filters: int,
                kernel_size: int,
                strides: int,
                pool_size: int,
                n_units: Tuple[int, int],
                gamma: float,
                alpha: float,
                dist_metric: str,
                cluster_init: str,
                heatmap: bool,
                pretrain_epochs: int,
                epochs: int,
                eval_epochs: int,
                save_epochs: int,
                batch_size: int,
                tol: float,
                patience: int,
                finetune_heatmap_at_epoch: int,
                initial_heatmap_loss_weight: float,
                final_heatmap_loss_weight: float,
                save_dir: str):
    """Train a DTC model with the provided hyperparameters."""
    os.makedirs(save_dir, exist_ok=True)

    if n_clusters is None:
        n_clusters = len(np.unique(y_train))

    dtc = DTC(
        n_clusters=n_clusters,
        input_dim=X_train.shape[-1],
        timesteps=X_train.shape[1],
        n_filters=n_filters,
        kernel_size=kernel_size,
        strides=strides,
        pool_size=pool_size,
        n_units=n_units,
        alpha=alpha,
        dist_metric=dist_metric,
        cluster_init=cluster_init,
        heatmap=heatmap,
    )

    dtc.initialize()
    dtc.compile(
        gamma=gamma,
        optimizer="adam",
        initial_heatmap_loss_weight=initial_heatmap_loss_weight,
        final_heatmap_loss_weight=final_heatmap_loss_weight,
    )

    if pretrain_epochs > 0:
        with st.spinner("Pretraining autoencoder…"):
            dtc.pretrain(X=X_train, optimizer="adam", epochs=pretrain_epochs, batch_size=batch_size, save_dir=save_dir)

    dtc.init_cluster_weights(X_train)

    with st.spinner("Training clustering model…"):
        t0 = time()
        dtc.fit(
            X_train,
            y_train,
            None,
            None,
            epochs,
            eval_epochs,
            save_epochs,
            batch_size,
            tol,
            patience,
            finetune_heatmap_at_epoch,
            save_dir,
        )
        training_time = time() - t0

    st.success(f"Training finished in {training_time:.2f} seconds.")

    q = dtc.model.predict(X_train)[1]
    y_pred = q.argmax(axis=1)
    results = {"n_clusters": n_clusters}

    if y_train is not None:
        from metrics import cluster_acc, cluster_purity
        from sklearn import metrics

        results.update(
            acc=cluster_acc(y_train, y_pred),
            pur=cluster_purity(y_train, y_pred),
            nmi=metrics.normalized_mutual_info_score(y_train, y_pred),
            ari=metrics.adjusted_rand_score(y_train, y_pred),
        )

    return results


def main():
    st.set_page_config(page_title="Deep Temporal Clustering", layout="wide")
    st.title("Deep Temporal Clustering")
    st.write(
        "Train and evaluate the DTC model on datasets from the UCR/UEA archive or your own time-series CSV uploads. "
        "Tune hyperparameters in the sidebar, then start a run when you are ready."
    )

    with st.sidebar:
        st.header("Configuration")
        dataset_source = st.radio("Dataset source", ["UCR/UEA archive", "Upload CSV"], index=0)

        dataset_name: str | None = None
        uploaded_data: Tuple[np.ndarray, np.ndarray | None] | None = None
        uploaded_filename: str | None = None
        uploaded_file = None

        if dataset_source == "UCR/UEA archive":
            dataset_name = st.selectbox("Dataset", sorted(all_ucr_datasets), index=sorted(all_ucr_datasets).index("CBF"))
            X_sample, y_sample = load_dataset(dataset_name)
        else:
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"], help="Rows = samples, columns = time steps/features.")
            if uploaded_file is not None:
                uploaded_filename = uploaded_file.name
                df = load_uploaded_dataframe(uploaded_file.getvalue())

                st.caption("Select which columns to use as time steps/features and optionally choose a label column.")
                numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

                label_choice = st.selectbox("Label column (optional)", ["<none>"] + numeric_columns)
                label_column = None if label_choice == "<none>" else label_choice

                feature_choices = [col for col in numeric_columns if col != label_column]
                feature_columns = st.multiselect(
                    "Feature/time-step columns", feature_choices, default=feature_choices, help="At least one column required."
                )

                if feature_columns:
                    uploaded_data = prepare_uploaded_dataset(df, tuple(feature_columns), label_column)
                    X_sample, y_sample = uploaded_data
                else:
                    st.warning("Select at least one numeric column to use as the time series input.")

        if dataset_source == "UCR/UEA archive" or uploaded_data is not None:
            timesteps = X_sample.shape[1]
            valid_pool_sizes = [size for size in range(1, timesteps + 1) if timesteps % size == 0]
            default_pool_size = 8 if 8 in valid_pool_sizes else max([size for size in valid_pool_sizes if size <= 8], default=min(valid_pool_sizes))
        else:
            timesteps = 1
            valid_pool_sizes = [1]
            default_pool_size = 1

        n_clusters = st.number_input("Clusters (0 = infer from labels)", min_value=0, value=0, step=1)
        n_filters = st.number_input("Conv filters", min_value=1, value=50, step=1)
        kernel_size = st.number_input("Kernel size", min_value=1, value=10, step=1)
        strides = st.number_input("Strides", min_value=1, value=1, step=1)
        pool_size = st.selectbox(
            "Pool size",
            valid_pool_sizes,
            index=valid_pool_sizes.index(default_pool_size),
            help=f"Must divide the sequence length ({timesteps} timesteps).",
        )
        n_units_first = st.number_input("BiLSTM units (first layer)", min_value=1, value=50, step=1)
        n_units_second = st.number_input("BiLSTM units (second layer)", min_value=1, value=1, step=1)
        gamma = st.number_input("Gamma (clustering loss weight)", min_value=0.0, value=1.0, step=0.1)
        alpha = st.number_input("Alpha (Student kernel)", min_value=0.1, value=1.0, step=0.1)
        dist_metric = st.selectbox("Distance metric", ["eucl", "cid", "cor", "acf"], index=0)
        cluster_init = st.selectbox("Cluster initialization", ["kmeans", "hierarchical"], index=0)
        heatmap = st.checkbox("Train heatmap network", value=False)
        pretrain_epochs = st.number_input("Pretrain epochs", min_value=0, value=10, step=1)
        epochs = st.number_input("Training epochs", min_value=1, value=50, step=1)
        eval_epochs = st.number_input("Eval every N epochs", min_value=1, value=1, step=1)
        save_epochs = st.number_input("Save weights every N epochs", min_value=1, value=10, step=1)
        batch_size = st.number_input("Batch size", min_value=1, value=64, step=1)
        tol = st.number_input("Tolerance", min_value=0.0, value=0.001, step=0.0001, format="%f")
        patience = st.number_input("Patience", min_value=1, value=5, step=1)
        finetune_heatmap_at_epoch = st.number_input("Heatmap finetune epoch", min_value=1, value=8, step=1)
        initial_heatmap_loss_weight = st.number_input(
            "Initial heatmap loss weight", min_value=0.0, value=0.1, step=0.1
        )
        final_heatmap_loss_weight = st.number_input(
            "Final heatmap loss weight", min_value=0.0, value=0.9, step=0.1
        )
        save_dir = st.text_input("Save directory", value="results/tmp")

        start_training = st.button("Start training", type="primary", disabled=(dataset_source == "Upload CSV" and uploaded_data is None))

    if start_training:
        if dataset_source == "UCR/UEA archive":
            X_train, y_train = load_dataset(dataset_name)
        else:
            if uploaded_data is None:
                st.error("Upload a dataset and select feature columns before starting training.")
                return
            X_train, y_train = uploaded_data

        if n_clusters == 0:
            if y_train is None:
                st.error("Please specify the number of clusters when labels are not provided.")
                return
            n_clusters_value = len(np.unique(y_train))
        else:
            n_clusters_value = n_clusters

        results = run_training(
            X_train,
            y_train,
            n_clusters_value,
            n_filters,
            kernel_size,
            strides,
            pool_size,
            (n_units_first, n_units_second),
            gamma,
            alpha,
            dist_metric,
            cluster_init,
            heatmap,
            pretrain_epochs,
            epochs,
            eval_epochs,
            save_epochs,
            batch_size,
            tol,
            patience,
            finetune_heatmap_at_epoch,
            initial_heatmap_loss_weight,
            final_heatmap_loss_weight,
            save_dir,
        )

        if dataset_source == "UCR/UEA archive":
            results["dataset"] = dataset_name
        else:
            results["dataset"] = uploaded_filename or "uploaded"

        st.subheader("Training results")
        st.json(results)
    else:
        st.info("Configure the training parameters in the sidebar and click **Start training** to begin.")


if __name__ == "__main__":
    main()
