import pandas as pd
import streamlit as st
from prophet import Prophet
from prophet.plot import plot_components_plotly, plot_plotly
from datetime import datetime
import numpy as np
import math
import plotly.graph_objects as go


DATA_PATH = "data/03-final/daily_truck_data.csv"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    data["order_date"] = pd.to_datetime(data["order_date"])
    return data.sort_values("order_date").reset_index(drop=True)


def build_holidays() -> pd.DataFrame:
    holiday_dates = [
        "2023-01-01",
        "2023-02-06",
        "2023-03-20",
        "2023-05-01",
        "2023-09-16",
        "2023-11-20",
        "2023-12-25",
        "2024-01-01",
        "2024-02-05",
        "2024-03-18",
        "2024-05-01",
        "2024-09-16",
        "2024-11-18",
        "2024-12-25",
        "2025-01-01",
        "2025-02-03",
        "2025-03-17",
        "2025-05-01",
        "2025-09-16",
        "2025-11-17",
        "2025-12-25",
        "2026-01-01",
        "2026-02-02",
        "2026-03-16",
        "2026-05-01",
    ]
    return pd.DataFrame(
        {
            "holiday": "non_operating_day",
            "ds": pd.to_datetime(holiday_dates),
            "lower_window": 0,
            "upper_window": 0,
        }
    )


def run_prophet_forecast(
    data: pd.DataFrame,
    target_col: str,
    forecast_days: int,
    interval_width: float,
    use_holidays: bool,
    truck_daily_capacity: float = 15.0,
) -> tuple[Prophet, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prophet_df = data[["order_date", target_col]].rename(
        columns={"order_date": "ds", target_col: "y"}
    )

    model = Prophet(
        growth="linear",
        changepoint_prior_scale=0.05,
        n_changepoints=25,
        seasonality_mode="multiplicative" if target_col == "volume" else "additive",
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        holidays=build_holidays() if use_holidays else None,
        interval_width=interval_width,
    )

    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=forecast_days, freq="D")
    forecast = model.predict(future)

    forecast[["yhat", "yhat_lower", "yhat_upper"]] = forecast[
        ["yhat", "yhat_lower", "yhat_upper"]
    ].clip(lower=0)

    cutoff_date = prophet_df["ds"].max()
    forecast_future = forecast[forecast["ds"] > cutoff_date].copy()

    weekly_summary = forecast_future.groupby(forecast_future['ds'].dt.isocalendar().week).agg({
        'yhat': 'mean',
        'yhat_lower': 'mean',
        'yhat_upper': 'mean'
    }).reset_index()

    current_week = pd.Timestamp(datetime.now()).isocalendar().week

    weekly_summary['week'] = weekly_summary['week'] - current_week + 1

    weekly_summary['trucks_needed'] = np.ceil(weekly_summary['yhat'] / truck_daily_capacity)
    weekly_summary['trucks_needed_lower'] = np.ceil(weekly_summary['yhat_lower'] / truck_daily_capacity)
    weekly_summary['trucks_needed_upper'] = np.ceil(weekly_summary['yhat_upper'] / truck_daily_capacity)

    return model, prophet_df, forecast, forecast_future, weekly_summary


def main() -> None:
    st.set_page_config(page_title="Truck Demand Dashboard", layout="wide")
    st.title("Truck Demand Forecast Dashboard")
    st.caption("Historical trends + Prophet forecast from your cleaned daily data")

    data = load_data(DATA_PATH)

    st.sidebar.header("Forecast Controls")
    metric = "volume"  # For now, we only forecast volume. Can add more metrics later.
    forecast_days = st.sidebar.slider("Forecast days", min_value=30, max_value=60, value=45)
    interval_width = st.sidebar.slider(
        "Confidence interval",
        min_value=0.80,
        max_value=0.99,
        value=0.90,
        step=0.01,
    )
    truck_daily_capacity = st.sidebar.slider(
            "Daily volume per truck",
            min_value=0.5,
            value=15.0,
            max_value=30.0,
            step=0.1,
        )
    # use_holidays = st.sidebar.checkbox("Use holiday effects", value=True)
    use_holidays = True

    with st.spinner("Training Prophet and generating forecast..."):
        model, prophet_df, forecast, forecast_future, weekly_summary = run_prophet_forecast(
            data=data,
            target_col=metric,
            forecast_days=forecast_days,
            interval_width=interval_width,
            use_holidays=use_holidays,
            truck_daily_capacity=truck_daily_capacity,
        )

    st.subheader(f"Truck Requirement Estimate ({forecast_days} days - {forecast_days // 7} weeks ahead)")

    total_forecast_volume = weekly_summary[weekly_summary['week'] == forecast_days // 7]['yhat'].sum()
    trucks_needed = weekly_summary[weekly_summary['week'] == forecast_days // 7]['trucks_needed'].sum()
    trucks_needed_lower = weekly_summary[weekly_summary['week'] == forecast_days // 7]['trucks_needed_lower'].sum()
    trucks_needed_upper = weekly_summary[weekly_summary['week'] == forecast_days // 7]['trucks_needed_upper'].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric(f"Forecast week volume", f"{total_forecast_volume:,.1f}")
    col2.metric(f"Estimated trucks needed", f"{trucks_needed:,}")
    col3.metric("Range (low-high)", f"{trucks_needed_lower:,} - {trucks_needed_upper:,}")

    forecast_fig = plot_plotly(
        model,
        forecast,
        xlabel="Date",
        ylabel=f"Predicted {metric}",
    )

    # Add secondary y-axis for trucks needed based on daily volume per truck
    daily_trucks_needed = np.ceil(forecast["yhat"] / truck_daily_capacity)
    forecast_fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=daily_trucks_needed,
            name="Trucks Needed",
            mode="lines",
            yaxis="y2",
            line=dict(color="red"),
        )
    )
    forecast_fig.update_layout(
        yaxis2=dict(
            title="Trucks Needed",
            overlaying="y",
            side="right",
        )
    )
    st.plotly_chart(forecast_fig, width='stretch')

    st.subheader("Forecast Components")
    components_fig = plot_components_plotly(model, forecast)
    st.plotly_chart(components_fig, width='stretch')

    st.subheader("Forward Forecast Table")
    st.dataframe(
        weekly_summary[["week", "yhat", "yhat_lower", "yhat_upper", "trucks_needed", "trucks_needed_lower", "trucks_needed_upper"]]
        .rename(columns={
            "week": "Week #",
            "yhat": "Predicted Volume",
            "yhat_lower": "Volume Lower Bound",
            "yhat_upper": "Volume Upper Bound",
            "trucks_needed": "Trucks Needed",
            "trucks_needed_lower": "Trucks Needed Lower",
            "trucks_needed_upper": "Trucks Needed Upper",
        })
        .round(2),
        width='stretch',
    )


if __name__ == "__main__":
    main()