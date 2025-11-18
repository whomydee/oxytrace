# flake8: noqa
"""
OxyTrace Streamlit Application

Interactive web interface for oxygen anomaly detection and forecasting.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from oxytrace.core.features.oxygen import OxygenFeatureEngineer
from oxytrace.core.models.detector import AnomalyDetector
from oxytrace.core.models.forecaster import Forecaster
from oxytrace.core.utils.logger import LOGGER


def get_available_models(model_type="detector"):
    """Get list of available trained models."""
    if model_type == "detector":
        base_path = Path("artifacts/anomaly_detector")
    else:
        base_path = Path("artifacts/forecaster")

    if not base_path.exists():
        return []

    # Look for model files
    if model_type == "detector":
        model_files = list(base_path.glob("anomaly_detector*.pkl"))
    else:
        model_files = list(base_path.glob("forecaster*.pkl"))

    return [f.stem for f in model_files]


def load_detector_model(model_name="anomaly_detector"):
    """Load anomaly detector and feature engineer."""
    base_path = Path("artifacts/anomaly_detector")

    fe_path = base_path / "feature_engineer.pkl"
    model_path = base_path / f"{model_name}.pkl"

    if not fe_path.exists() or not model_path.exists():
        return None, None

    feature_engineer = OxygenFeatureEngineer.load(str(fe_path))
    detector = AnomalyDetector.load(str(model_path))

    return feature_engineer, detector


def load_forecaster_model(model_name="forecaster"):
    """Load forecaster model."""
    model_path = Path("artifacts/forecaster") / f"{model_name}.pkl"

    if not model_path.exists():
        return None

    forecaster = Forecaster.load(str(model_path))
    return forecaster


def train_detector_page():
    """Anomaly detector training page."""
    st.header("🔧 Train Anomaly Detector")

    st.markdown(
        """
    Train an Isolation Forest model to detect anomalies in oxygen sensor data.
    The model learns patterns from normal data and identifies unusual readings.
    """
    )

    col1, col2 = st.columns(2)

    with col1:
        data_percent = st.slider(
            "Data Percentage",
            min_value=1,
            max_value=100,
            value=10,
            help="Percentage of data to use for training (lower = faster)",
        )

    with col2:
        contamination = st.slider(
            "Contamination",
            min_value=0.01,
            max_value=0.2,
            value=0.05,
            step=0.01,
            help="Expected proportion of anomalies in the data",
        )

    if st.button("Start Training", type="primary"):
        with st.spinner("Training anomaly detector..."):
            try:
                # Load data
                df = pd.read_csv("oxytrace/data/dataset.csv")
                df["time"] = pd.to_datetime(df["time"], format="mixed")
                df = df[df["Oxygen[%sat]"].notna()].copy()
                df = df.sort_values("time").reset_index(drop=True)

                # Sample data
                n_samples = int(len(df) * (data_percent / 100))
                df_sample = df.iloc[:n_samples].copy()

                st.info(f"Training on {len(df_sample):,} samples ({data_percent}% of data)")

                # Train feature engineer
                feature_engineer = OxygenFeatureEngineer()
                feature_engineer.fit(df_sample)

                # Engineer features
                features = feature_engineer.transform(df_sample)

                # Train detector
                detector = AnomalyDetector(contamination=contamination)
                detector.fit(features)

                # Save models
                output_dir = Path("artifacts/anomaly_detector")
                output_dir.mkdir(parents=True, exist_ok=True)

                feature_engineer.save(str(output_dir / "feature_engineer.pkl"))
                detector.save(str(output_dir / "anomaly_detector.pkl"))

                st.success("✅ Training complete!")
                st.success(f"Models saved to: {output_dir}")

                # Show training info
                st.metric("Samples Trained", f"{len(df_sample):,}")
                st.metric("Features Engineered", len(features.columns))
                st.metric("Contamination", f"{contamination:.2%}")

            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                LOGGER.error("Training failed", error=str(e))


def train_forecaster_page():
    """Forecaster training page."""
    st.header("📈 Train Forecaster")

    st.markdown(
        """
    Train a seasonal forecasting model with daily and weekly patterns.
    The model captures trends and seasonality in oxygen readings.
    """
    )

    col1, col2 = st.columns(2)

    with col1:
        alpha = st.slider(
            "Smoothing Factor (α)",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.1,
            help="Higher values respond faster to changes",
        )

    with col2:
        data_percent = st.slider(
            "Data Percentage", min_value=10, max_value=100, value=100, help="Percentage of data to use for training"
        )

    if st.button("Start Training", type="primary"):
        with st.spinner("Training forecaster..."):
            try:
                # Load data
                df = pd.read_csv("oxytrace/data/dataset.csv")
                df["time"] = pd.to_datetime(df["time"], format="mixed")
                df = df[df["Oxygen[%sat]"].notna()].copy()
                df = df.sort_values("time").reset_index(drop=True)

                # Sample data
                n_samples = int(len(df) * (data_percent / 100))
                df_sample = df.iloc[:n_samples].copy()

                st.info(f"Training on {len(df_sample):,} samples ({data_percent}% of data)")

                # Train forecaster
                forecaster = Forecaster(alpha=alpha)
                forecaster.fit(df_sample)

                # Save model
                output_dir = Path("artifacts/forecaster")
                output_dir.mkdir(parents=True, exist_ok=True)

                forecaster.save(str(output_dir / "forecaster.pkl"))

                st.success("✅ Training complete!")
                st.success(f"Model saved to: {output_dir}")

                # Show training info
                st.metric("Samples Trained", f"{len(df_sample):,}")
                st.metric("Smoothing Factor (α)", f"{alpha:.1f}")
                st.metric("Has Seasonality", "Yes" if forecaster.has_seasonality_ else "No")

            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                LOGGER.error("Training failed", error=str(e))


def predict_anomaly_page():
    """Anomaly detection prediction page."""
    st.header("🔍 Detect Anomalies")

    st.markdown(
        """
    Upload oxygen sensor data or enter readings manually to detect anomalies.
    The model will identify unusual patterns and assign severity levels.
    """
    )

    # Model selection
    available_models = get_available_models("detector")

    if not available_models:
        st.warning("⚠️ No trained models found. Please train a model first.")
        return

    selected_model = st.selectbox(
        "Select Model", available_models, help="Choose which trained model to use for prediction"
    )

    # Load model
    feature_engineer, detector = load_detector_model(selected_model)

    if feature_engineer is None or detector is None:
        st.error("❌ Failed to load model")
        return

    st.success(f"✅ Model loaded: {selected_model}")

    # Input method selection
    input_method = st.radio("Input Method", ["Upload CSV", "Manual Entry"], horizontal=True)

    # Initialize session state for input data
    if "anomaly_input_data" not in st.session_state:
        st.session_state.anomaly_input_data = None

    if input_method == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload CSV file", type=["csv"], help="CSV must have 'time' and 'Oxygen[%sat]' columns"
        )

        if uploaded_file is not None:
            df_input = pd.read_csv(uploaded_file)
            df_input["time"] = pd.to_datetime(df_input["time"])
            st.session_state.anomaly_input_data = df_input
            st.success(f"✅ Loaded {len(df_input)} readings")

    else:  # Manual Entry
        st.markdown("**Enter readings (one per line): `time, oxygen`**")
        st.markdown("Example: `2024-01-01 10:00:00, 88.5`")

        manual_input = st.text_area(
            "Oxygen Readings",
            height=150,
            placeholder="2024-01-01 10:00:00, 88.5\n2024-01-01 10:01:00, 87.2\n2024-01-01 10:02:00, 89.1",
        )

        if st.button("Parse Input"):
            if manual_input.strip():
                try:
                    lines = [line.strip() for line in manual_input.strip().split("\n") if line.strip()]
                    data = []
                    for line in lines:
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) == 2:
                            data.append({"time": parts[0], "Oxygen[%sat]": float(parts[1])})

                    df_input = pd.DataFrame(data)
                    df_input["time"] = pd.to_datetime(df_input["time"])
                    st.session_state.anomaly_input_data = df_input
                    st.success(f"✅ Parsed {len(df_input)} readings")

                except Exception as e:
                    st.error(f"❌ Failed to parse input: {str(e)}")

    # Get data from session state
    df_input = st.session_state.anomaly_input_data

    # Show data preview if available
    if df_input is not None:
        with st.expander("📊 Preview Data", expanded=False):
            st.dataframe(df_input.head(10), use_container_width=True)

    # Threshold setting
    threshold = st.slider(
        "Anomaly Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.25,
        step=0.05,
        help="Higher values = stricter detection",
    )

    # Predict button
    if df_input is not None and st.button("Detect Anomalies", type="primary"):
        with st.spinner("Detecting anomalies..."):
            try:
                # Engineer features
                features = feature_engineer.transform(df_input)

                # Predict
                scores, severity = detector.predict_with_severity(features)

                # Add results to dataframe
                df_results = df_input.copy()
                df_results["anomaly_score"] = scores
                df_results["severity"] = severity
                df_results["is_anomaly"] = scores > threshold

                severity_labels = {0: "normal", 1: "mild", 2: "moderate", 3: "severe"}
                df_results["severity_label"] = df_results["severity"].map(severity_labels)

                # Display metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Readings", len(df_results))

                with col2:
                    anomaly_count = df_results["is_anomaly"].sum()
                    st.metric("Anomalies Detected", anomaly_count)

                with col3:
                    anomaly_rate = (anomaly_count / len(df_results)) * 100
                    st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")

                with col4:
                    avg_score = df_results[df_results["is_anomaly"]]["anomaly_score"].mean() if anomaly_count > 0 else 0
                    st.metric("Avg Anomaly Score", f"{avg_score:.3f}")

                # Severity breakdown
                st.subheader("Severity Breakdown")
                severity_counts = df_results["severity_label"].value_counts()

                fig_pie = px.pie(
                    values=severity_counts.values,
                    names=severity_counts.index,
                    title="Severity Distribution",
                    color_discrete_map={
                        "normal": "#00CC96",
                        "mild": "#FFA15A",
                        "moderate": "#EF553B",
                        "severe": "#B6E880",
                    },
                )
                st.plotly_chart(fig_pie, use_container_width=True)

                # Time series plot
                st.subheader("Anomaly Detection Timeline")

                fig = go.Figure()

                # Normal points
                normal_data = df_results[~df_results["is_anomaly"]]
                fig.add_trace(
                    go.Scatter(
                        x=normal_data["time"],
                        y=normal_data["Oxygen[%sat]"],
                        mode="markers",
                        name="Normal",
                        marker=dict(color="blue", size=6),
                    )
                )

                # Anomaly points
                anomaly_data = df_results[df_results["is_anomaly"]]
                fig.add_trace(
                    go.Scatter(
                        x=anomaly_data["time"],
                        y=anomaly_data["Oxygen[%sat]"],
                        mode="markers",
                        name="Anomaly",
                        marker=dict(color="red", size=10, symbol="x"),
                    )
                )

                fig.update_layout(
                    title="Oxygen Levels with Anomalies Highlighted",
                    xaxis_title="Time",
                    yaxis_title="Oxygen [%sat]",
                    hovermode="x unified",
                )

                st.plotly_chart(fig, use_container_width=True)

                # Show anomalies table
                if anomaly_count > 0:
                    st.subheader("Detected Anomalies")
                    anomalies_df = df_results[df_results["is_anomaly"]][
                        ["time", "Oxygen[%sat]", "anomaly_score", "severity_label"]
                    ].sort_values("anomaly_score", ascending=False)
                    st.dataframe(anomalies_df, use_container_width=True)

                # Download results
                st.subheader("Download Results")
                csv = df_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"anomaly_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                LOGGER.error("Prediction failed", error=str(e))


def predict_forecast_page():
    """Forecasting prediction page."""
    st.header("📊 Generate Forecast")

    st.markdown(
        """
    Generate oxygen level forecasts using trained seasonal models.
    The model predicts future values with uncertainty intervals.
    """
    )

    # Model selection
    available_models = get_available_models("forecaster")

    if not available_models:
        st.warning("⚠️ No trained models found. Please train a model first.")
        return

    selected_model = st.selectbox(
        "Select Model", available_models, help="Choose which trained model to use for forecasting"
    )

    # Load model
    forecaster = load_forecaster_model(selected_model)

    if forecaster is None:
        st.error("❌ Failed to load model")
        return

    st.success(f"✅ Model loaded: {selected_model}")

    # Forecast parameters
    col1, col2 = st.columns(2)

    with col1:
        horizon = st.number_input(
            "Forecast Horizon (minutes)",
            min_value=60,
            max_value=10080,
            value=1440,
            step=60,
            help="How far into the future to predict (1440 = 1 day)",
        )

    with col2:
        horizon_days = horizon / 1440
        st.metric("Forecast Duration", f"{horizon_days:.1f} days")

    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Generating forecast..."):
            try:
                # Generate forecast
                forecast_df = forecaster.predict(horizon=horizon)

                # Rename columns for display
                forecast_df = forecast_df.rename(
                    columns={
                        "predicted_oxygen": "forecast",
                        "lower_bound_95": "lower_bound",
                        "upper_bound_95": "upper_bound",
                    }
                )

                st.success(f"✅ Forecast generated for {len(forecast_df)} time steps")

                # Display metrics
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Forecast Points", len(forecast_df))

                with col2:
                    avg_pred = forecast_df["forecast"].mean()
                    st.metric("Avg Predicted O₂", f"{avg_pred:.2f}%")

                with col3:
                    uncertainty = (forecast_df["upper_bound"] - forecast_df["lower_bound"]).mean()
                    st.metric("Avg Uncertainty", f"±{uncertainty/2:.2f}%")

                # Forecast plot
                st.subheader("Forecast Visualization")

                fig = go.Figure()

                # Predicted values
                fig.add_trace(
                    go.Scatter(
                        x=forecast_df["time"],
                        y=forecast_df["forecast"],
                        mode="lines",
                        name="Forecast",
                        line=dict(color="blue", width=2),
                    )
                )

                # Confidence interval
                fig.add_trace(
                    go.Scatter(
                        x=forecast_df["time"],
                        y=forecast_df["upper_bound"],
                        mode="lines",
                        name="Upper Bound (95%)",
                        line=dict(width=0),
                        showlegend=False,
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=forecast_df["time"],
                        y=forecast_df["lower_bound"],
                        mode="lines",
                        name="Lower Bound (95%)",
                        fill="tonexty",
                        fillcolor="rgba(0,100,255,0.2)",
                        line=dict(width=0),
                        showlegend=True,
                    )
                )

                fig.update_layout(
                    title=f"Oxygen Forecast ({horizon_days:.1f} days)",
                    xaxis_title="Time",
                    yaxis_title="Oxygen [%sat]",
                    hovermode="x unified",
                )

                st.plotly_chart(fig, use_container_width=True)

                # Show forecast table
                st.subheader("Forecast Data")
                st.dataframe(
                    forecast_df[["time", "forecast", "lower_bound", "upper_bound"]].head(20), use_container_width=True
                )

                # Download forecast
                st.subheader("Download Forecast")
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"❌ Forecast failed: {str(e)}")
                LOGGER.error("Forecast failed", error=str(e))


def main():
    """Main Streamlit application."""
    st.set_page_config(page_title="OxyTrace", page_icon="🫁", layout="wide", initial_sidebar_state="expanded")

    # Sidebar
    st.sidebar.title("🫁 OxyTrace")
    st.sidebar.markdown("Oxygen Anomaly Detection & Forecasting")

    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "🔧 Train Detector", "📈 Train Forecaster", "🔍 Detect Anomalies", "📊 Generate Forecast"],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        """
    **OxyTrace** uses machine learning to:
    - Detect anomalies in oxygen sensor data
    - Forecast future oxygen levels
    - Identify patterns and trends
    """
    )

    # Main content
    if page == "🏠 Home":
        st.title("Welcome to OxyTrace 🫁")

        st.markdown(
            """
        ## Oxygen Sensor Anomaly Detection & Forecasting
        
        OxyTrace is a comprehensive platform for analyzing oxygen sensor data using machine learning.
        
        ### Features
        
        - **🔧 Train Models**: Train custom anomaly detection and forecasting models on your data
        - **🔍 Detect Anomalies**: Identify unusual patterns in oxygen readings with severity levels
        - **📊 Forecast**: Predict future oxygen levels with confidence intervals
        - **📈 Visualize**: Interactive charts and insights into your data
        
        ### Quick Start
        
        1. **Train Models**: Use the training pages to build models on your data
        2. **Make Predictions**: Upload data or enter readings manually
        3. **Analyze Results**: View charts, metrics, and download results
        
        ### Model Performance
        
        - **Anomaly Detection**: 64.3% recall, optimal threshold 0.25
        - **Forecasting**: 10% MAE on 7-day horizon
        
        Use the sidebar to navigate between different features.
        """
        )

        # Show available models
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Available Detectors")
            detector_models = get_available_models("detector")
            if detector_models:
                for model in detector_models:
                    st.success(f"✅ {model}")
            else:
                st.warning("No trained detectors found")

        with col2:
            st.subheader("Available Forecasters")
            forecaster_models = get_available_models("forecaster")
            if forecaster_models:
                for model in forecaster_models:
                    st.success(f"✅ {model}")
            else:
                st.warning("No trained forecasters found")

    elif page == "🔧 Train Detector":
        train_detector_page()

    elif page == "📈 Train Forecaster":
        train_forecaster_page()

    elif page == "🔍 Detect Anomalies":
        predict_anomaly_page()

    elif page == "📊 Generate Forecast":
        predict_forecast_page()


if __name__ == "__main__":
    main()
