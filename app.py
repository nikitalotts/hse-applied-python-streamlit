import asyncio

import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from src.data_manager import DataManager
from src.open_weather_client import OpenWeatherClient
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime

st.set_page_config(
    page_title="Weather Analysis App",
    layout="wide"
)


async def main():
    uploaded_file = await upload_file_block()

    if uploaded_file is not None:
        data_manager = await upload_file(uploaded_file)

        city = await city_block(data_manager)
        valid_key, open_weather_client = await api_key_block()

        if not valid_key:
            st.error("API KEY IS NOT VALID")
        else:
            if st.button("Analyze and Monitor Temperature"):
                st.subheader("Statistics")
                stats = await describe_statistics(city, data_manager)
                st.dataframe(stats)
                decomposition = data_manager.get_decomposition(city)
                st.subheader("Time series components")
                await time_series_components_block(decomposition)
                st.subheader("Anomalies and Seasonal Profiles")
                await render_anomalies(data_manager, city)
                await render_seasonal_profile(data_manager, city)
                st.subheader("Current temperature analysis")
                await render_current_weather(city, data_manager, open_weather_client)
                st.subheader("Prediction of mean temperature for next month (ARIMA model)")
                pred = await predict(data_manager, city)
                await draw_predictions(decomposition, pred)


async def draw_predictions(decomposition, pred):
    observed_df = decomposition.observed.reset_index()
    observed_df.columns = ['Date', 'Value']

    predicted_df = pred.reset_index()
    predicted_df.columns = ['Date', 'Value']

    fig = px.line(
        observed_df[-330:],
        x='Date',
        y="Value",
        labels={'Value': ''},
        line_shape='linear',
        color_discrete_sequence=["lightblue"]
    )

    fig.add_scatter(x=predicted_df['Date'], y=predicted_df['Value'], mode='lines', name='Predicted', line=dict(color="green"))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Value",
        height=400,
        template="plotly_white",
        legend_title=""
    )

    st.plotly_chart(fig, use_container_width=True, key="predictions")


async def predict(data_manager: DataManager, city: str):
    return pd.DataFrame(data_manager.predict_arima(city))


async def render_anomalies(data_manager: DataManager, city: str):
    anomalies = data_manager.detect_anomalies(city)
    if anomalies is not None:
        data = data_manager.get_data_by_city(city).to_pandas() if data_manager.parallel else data_manager.get_data_by_city(city)

        fig = px.scatter(
            data,
            x="timestamp",
            y="temperature",
            color="is_anomaly",
            color_discrete_map={True: "red", False: "lightblue"},
            labels={"is_anomaly": "Anomaly"},
            title="Temperature analysis",
            size_max=10
        )
        fig.update_traces(marker=dict(size=6), selector=dict(mode="markers"))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Temperature (°C)",
            legend_title="Anomalies",
            template="plotly_white",
        )

        st.plotly_chart(fig, use_container_width=True)


async def render_seasonal_profile(data_manager: DataManager, city: str):
    data = data_manager.get_data_by_city(city)

    fig = px.scatter(
        data,
        x="timestamp",
        y="temperature",
        color="is_anomaly_season",
        color_discrete_map={True: "red", False: "lightgreen"},
        labels={"is_anomaly": "Anomaly"},
        title="Temperature analysis",
        size_max=10
    )
    fig.update_traces(marker=dict(size=6), selector=dict(mode="markers"))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        legend_title="Anomalies",
        template="plotly_white",
    )

    fig.add_scatter(
        x=data["timestamp"],
        y=data["season_lower_bound"],
        mode="lines",
        line=dict(dash="dot", color="green"),
        name="Lower Bound (Mean - 2*Std)",
    )

    fig.add_scatter(
        x=data["timestamp"],
        y=data["season_upper_bound"],
        mode="lines",
        line=dict(dash="dot", color="yellow"),
        name="Upper Bound (Mean + 2*Std)",
    )

    fig.add_scatter(
        x=data["timestamp"],
        y=data["season_mean"],
        mode="lines",
        line=dict(dash="dot", color="pink"),
        name="Season Mean",
    )

    fig.update_traces(marker=dict(size=6), selector=dict(mode="markers"))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        legend_title="Legend",
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)

async def render_trend(data_manager: DataManager, city: str):
    trend = data_manager.get_trend(city)
    if trend is not None:
        st.subheader("Temperature Trend")
        fig = px.line(
            trend,
            x="timestamp",
            y=["trend"],
            labels={"value": "Temperature (°C)", "variable": "Type"},
            title="Temperature Trend",
            hover_data=["timestamp"]
        )
        st.plotly_chart(fig, use_container_width=True)


async def upload_file(uploaded_file: UploadedFile) -> DataManager:
    data_manager = DataManager()
    data_manager.upload_streamlit_file(uploaded_file.getvalue())
    return data_manager


async def city_block(data_manager: DataManager):
    st.subheader("Select a City")
    cities = data_manager.get_cities()
    city = st.selectbox("City", cities)
    return city


async def render_anomalies(data_manager: DataManager, city: str):
    data = data_manager.get_data_by_city(city)

    fig = px.scatter(
        data,
        x="timestamp",
        y="temperature",
        color="is_anomaly",
        color_discrete_map={True: "red", False: "lightblue"},
        labels={"is_anomaly": "Anomaly"},
        title=f"Anomalies And Bounds For {city}",
    )

    fig.add_scatter(
        x=data["timestamp"],
        y=data["lower_bound"],
        mode="lines",
        line=dict(dash="dot", color="green"),
        name="Lower Bound (MA - 2*Std)",
    )

    fig.add_scatter(
        x=data["timestamp"],
        y=data["upper_bound"],
        mode="lines",
        line=dict(dash="dot", color="yellow"),
        name="Upper Bound (MA + 2*Std)",
    )

    fig.add_scatter(
        x=data["timestamp"],
        y=data["rolling_mean"],
        mode="lines",
        line=dict(dash="dot", color="pink"),
        name="Moving Average (30 days)",
    )

    fig.update_traces(marker=dict(size=6), selector=dict(mode="markers"))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        legend_title="Anomalies",
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)


async def api_key_block():
    st.subheader("Enter OpenWeatherMap API Key")
    api_key = st.text_input("API Key", placeholder="Enter key")
    open_weather_client = OpenWeatherClient(api_key)
    valid_key = open_weather_client.check_key()

    return valid_key, open_weather_client


async def upload_file_block():
    st.title("Weather Analysis App")
    st.subheader("Upload Historical Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    return uploaded_file


async def describe_statistics(city: str, data_manager: DataManager):
    return data_manager.get_describe_statisic(city)


async def render_current_weather(
        city: str,
        data_manager: DataManager,
        open_weather_client: OpenWeatherClient):
    current_weather = await open_weather_client.get_weather_async(city)
    if "error" not in current_weather:
        current_temp = current_weather["main"]["temp"]
        st.write(f"Current temperature in {city}: {current_temp:.2f}°C")
        anomaly_status = "normal" if data_manager.is_temperature_normal(city, season=str(get_season()), current_temp=current_temp) \
            else "abnormal"
        st.write(f"The current temperature is **{anomaly_status}** for the season.")
    else:
        st.error(f"Failed to fetch current weather: {current_weather['error']}")


async def time_series_components_block(decomposition):

    components = {
        "Observed": decomposition.observed,
        "Trend": decomposition.trend,
        "Seasonal": decomposition.seasonal,
        "Residuals": decomposition.resid
    }

    colors = {
        "Observed": "white",
        "Trend": "orange",
        "Seasonal": "green",
        "Residuals": "red"
    }

    for component_name, component_data in components.items():
        df = component_data.reset_index()
        df.columns = ['Date', 'Value']

        fig = px.line(
            df,
            x='Date',
            y='Value',
            title=f"{component_name} Component",
            labels={'Value': component_name},
            line_shape='linear'
        )

        fig.update_traces(line=dict(color=colors[component_name]))

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=component_name,
            height=400,
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)


def get_season():
    month = datetime.now().month

    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    elif month in [9, 10, 11]:
        return "autumn"


if __name__ == "__main__":
    asyncio.run(main())