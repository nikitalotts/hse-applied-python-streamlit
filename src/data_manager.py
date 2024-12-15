import time
import uuid
import plotly.express as px
import polars as pl
import pandas as pd
from typing import Union, Optional, List
import os
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly
import statsmodels.api as sm

from methodtools import lru_cache

COLUMNS = ['temperature', 'season', 'city', 'timestamp']
FILES_DIR = 'files'


class DataManager:

    def __init__(self, file_path: str = None, parallel: bool = True) -> None:
        self.file_path = None
        self.parallel = parallel
        self.files_dir = os.path.join(os.path.dirname(__file__), './files_folder')
        self.check_files_dir()
        self.data = None
        self.table = None
        if file_path is not None:
            self.file_path = file_path
            self.data = self.check_data(self.load_data())
            self.table = self.process_data()


    def check_files_dir(self):
        if not os.path.exists(self.files_dir):
            os.makedirs(self.files_dir)

    def load_file(self, file_data):
        file_name = str(uuid.uuid4())
        file_path = f"{os.path.join(self.files_dir, file_name)}.csv"
        with open(file_path, "wb") as f:
            f.write(file_data)

        return file_path

    def upload_streamlit_file(self, file: bytes):
        file_path = self.load_file(file)
        self.data = self.check_data(self.load_data(file_path))
        self.table = self.process_data()
        self.file_path = file_path
        return self.load_data(file_path)

    def load_data(self, file_path: str = None) -> Optional[Union[pl.DataFrame, pd.DataFrame]]:
        if file_path is None:
            file_path = self.file_path
        try:
            if self.parallel:
                return pl.read_csv(file_path)
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"Error during data loading: {e}")
            return None

    def check_data(self, data) -> None:
        try:
            for col in COLUMNS:
                _ = data[col]
            return data
        except Exception as e:
            print(f"Error during data check: {e}")
            return None

    @lru_cache()
    def get_rolling_mean(self, city: str, days: int = 30) -> Optional[Union[pl.Series, pd.Series]]:
        if self.data is None:
            raise ValueError("Data is None")
        try:
            if self.parallel:
                return self.data.filter(pl.col("city") == city) \
                    .select(pl.col("temperature") \
                            .rolling_mean(window_size=days).alias("rolling_mean"))["rolling_mean"]
            else:
                return self.data[self.data["city"] == city]["temperature"] \
                    .rolling(window=days).mean().reset_index(drop=True)
        except Exception as e:
            print(f"Error calculating rolling mean: {e}")
            return None

    @lru_cache()
    def get_rolling_std(self, city: str, days: int = 30) -> Optional[Union[pl.Series, pd.Series]]:
        try:
            if self.parallel:
                return self.get_data_by_city(city) \
                    .select(pl.col("temperature") \
                            .rolling_std(window_size=days).alias("rolling_std"))["rolling_std"]
            return self.get_data_by_city(city) \
                .apply(lambda g: g[g['city'] == city])["temperature"] \
                .rolling(days).std()
        except Exception as e:
            print(f"Error calculating rolling std: {e}")
            return None

    @lru_cache()
    def get_data_by_city(self, city: str):
        return self.table.filter(pl.col("city") == city) if self.parallel\
            else self.table[self.table["city"] == city]

    @lru_cache()
    def get_temperature_by_city(self, city: str):
        return self.get_data_by_city(city)["temperature"]


    @lru_cache()
    def detect_anomalies(self, city: str) -> Optional[Union[pl.Series, pd.Series]]:
        if self.data is None:
            raise ValueError("Data is None")
        if city is None or city not in self.get_cities():
            raise ValueError("City value is incorrect")

        try:
            city_data = self.get_data_by_city(city)

            return city_data["is_anomaly"]

        except Exception as e:
            raise ValueError(f"Error during detecting anomalies: {str(e)}")

    @lru_cache()
    def get_trend(self, city: str) -> Optional[Union[pl.Series, pd.Series]]:
        if self.data is None:
            raise ValueError("Data is None")
        try:
            city_data = self.get_data_by_city(city)
            trend = self.get_rolling_mean(city, days=30)

            if self.parallel:
                city_data = city_data.with_columns(trend.alias("trend"))
            else:
                city_data['trend'] = trend

            return city_data
        except Exception as e:
            print(f"Error calculating trend: {e}")
            return None

    def get_describe_statisic(self, city: str):
        return self.get_data_by_city(city)["temperature"].describe()


    def get_data(self) -> Union[pl.DataFrame, pd.DataFrame]:
        return self.data

    def get_cities(self) -> Union[pl.Series, pd.Series]:
        return sorted(self.data["city"].unique())


    @lru_cache()
    def is_temperature_normal(self, city: str, timestamp: str = None, season: str = None,
                              current_temp: float = None) -> bool:
        print(city, timestamp, season, current_temp)
        if city is None or city not in self.get_cities():
            raise ValueError("City value is incorrect")

        if season is not None and timestamp is not None:
            raise ValueError("Season and timestamp are both not None")

        data = self.get_data_by_city(city)
        lower_bound, upper_bound = None, None
        if self.parallel:
            if season is not None:
                data = data.filter(pl.col("season") == season)
                lower_bound, upper_bound = data[0]["season_lower_bound"], data[0]["season_upper_bound"]

            if timestamp is not None:
                data = data.filter(pl.col("timestamp") == timestamp)
                lower_bound, upper_bound = data[0]["lower_bound"], data[0]["upper_bound"]
        else:
            if season is not None:
                data = data[data["season"] == season]

                lower_bound, upper_bound = data.iloc[0]["season_lower_bound"], data.iloc[0]["season_upper_bound"]

            if timestamp is not None:
                data = data[data["timestamp"] == timestamp]

                lower_bound, upper_bound = data.iloc[0]["lower_bound"], data.iloc[0]["upper_bound"]

        print(lower_bound.item(), current_temp, upper_bound.item())
        return lower_bound.item() <= current_temp <= upper_bound.item()

    def get_decomposition(self, city: str, period=365):
        if self.parallel:
            data = self.get_data_by_city(city).to_pandas().set_index("timestamp")
        else:
            data = self.get_data_by_city(city)
            data.set_index('timestamp', inplace=True)

        decomposition = seasonal_decompose(data['temperature'], model='additive', period=period)

        return decomposition

    def train_arima(self, city: str, order: tuple = (7, 1, 0)):
        data = self.get_data_by_city(city)
        temperatures = data['rolling_mean'].to_numpy()

        arima_model = sm.tsa.arima.ARIMA(
            temperatures,
            order=order
        )
        arima_model_fit = arima_model.fit()

        return arima_model_fit

    def predict_arima(self, city: str, steps: int = 30, order: tuple = (7, 1, 0)):
        arima_model_fit = self.train_arima(city, order=order)
        forecast = arima_model_fit.forecast(steps=steps)
        forecast_index = pd.date_range(start=self.get_data_by_city(city)['timestamp'].max(),
                                       periods=steps + 1,
                                       freq='D')[1:]
        forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['Predicted Temperature'])

        return forecast_df

    def process_data(self) -> Union[pd.DataFrame, pl.DataFrame]:
        if self.data is None:
            raise ValueError("Данные не загружены")

        try:
            if self.parallel:
                # скользящее среднее и std
                smoothed_data = self.data.with_columns(
                    pl.col("temperature").rolling_std(window_size=30).alias("rolling_std"),
                    pl.col("temperature").rolling_mean(window_size=30).alias("rolling_mean")
                )

                processed_data = smoothed_data.with_columns([
                    (pl.col("rolling_mean") - 2 * pl.col("rolling_std")).alias("lower_bound"),
                    (pl.col("rolling_mean") + 2 * pl.col("rolling_std")).alias("upper_bound"),
                ])

                processed_data = processed_data.with_columns([
                    (pl.col("temperature") < pl.col("lower_bound")).or_(
                        pl.col("temperature") > pl.col("upper_bound")
                    ).alias("is_anomaly")
                ])

                # вычисление сезонных значений
                seasonal_stats = processed_data.group_by(["city", "season"]).agg([
                    pl.col("temperature").mean().alias("season_mean"),
                    pl.col("temperature").std().alias("season_std")
                ])

                # добавление сезонных границ
                seasonal_stats = seasonal_stats.with_columns([
                    (pl.col("season_mean") - 2 * pl.col("season_std")).alias("season_lower_bound"),
                    (pl.col("season_mean") + 2 * pl.col("season_std")).alias("season_upper_bound")
                ])

                processed_data = processed_data.join(seasonal_stats, on=["city", "season"])

                processed_data = processed_data.with_columns([
                    (pl.col("temperature") < pl.col("season_lower_bound")).or_(
                        pl.col("temperature") > pl.col("season_upper_bound")
                    ).alias("is_anomaly_season")
                ])

                return processed_data

            else:
                self.data["rolling_mean"] = (
                    self.data.groupby("city")["temperature"]
                    .rolling(window=30, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )

                self.data["rolling_std"] = (
                    self.data.groupby("city")["temperature"]
                    .rolling(window=30, min_periods=1)
                    .std()
                    .reset_index(level=0, drop=True)
                )

                self.data["lower_bound"] = self.data["rolling_mean"] - 2 * self.data["rolling_std"]
                self.data["upper_bound"] = self.data["rolling_mean"] + 2 * self.data["rolling_std"]

                self.data["is_anomaly"] = (
                        (self.data["temperature"] < self.data["lower_bound"]) |
                        (self.data["temperature"] > self.data["upper_bound"])
                )

                seasonal_stats = (
                    self.data.groupby(["city", "season"])["temperature"]
                    .agg(season_mean="mean", season_std="std")
                    .reset_index()
                )

                seasonal_stats["season_lower_bound"] = seasonal_stats["season_mean"] - 2 * seasonal_stats["season_std"]
                seasonal_stats["season_upper_bound"] = seasonal_stats["season_mean"] + 2 * seasonal_stats["season_std"]

                # Объединение данных
                processed_data = pd.merge(
                    self.data,
                    seasonal_stats,
                    on=["city", "season"],
                    how="left"
                )

                processed_data["is_anomaly_season"] = (
                        (processed_data["temperature"] < processed_data["season_lower_bound"]) |
                        (processed_data["temperature"] > processed_data["season_upper_bound"])
                )

                return processed_data

        except Exception as e:
            print(f"Ошибка обработки данных: {e}")
            return None


def ensure_file():
    file_path = os.path.join(os.path.dirname(__file__), 'files_folder', 'temperature_data.csv')
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'files_folder')):
        os.mkdir('files_folder')
    if not os.path.exists(file_path):
        import pandas as pd
        import numpy as np

        # Реальные средние температуры (примерные данные) для городов по сезонам
        seasonal_temperatures = {
            "New York": {"winter": 0, "spring": 10, "summer": 25, "autumn": 15},
            "London": {"winter": 5, "spring": 11, "summer": 18, "autumn": 12},
            "Paris": {"winter": 4, "spring": 12, "summer": 20, "autumn": 13},
            "Tokyo": {"winter": 6, "spring": 15, "summer": 27, "autumn": 18},
            "Moscow": {"winter": -10, "spring": 5, "summer": 18, "autumn": 8},
            "Sydney": {"winter": 12, "spring": 18, "summer": 25, "autumn": 20},
            "Berlin": {"winter": 0, "spring": 10, "summer": 20, "autumn": 11},
            "Beijing": {"winter": -2, "spring": 13, "summer": 27, "autumn": 16},
            "Rio de Janeiro": {"winter": 20, "spring": 25, "summer": 30, "autumn": 25},
            "Dubai": {"winter": 20, "spring": 30, "summer": 40, "autumn": 30},
            "Los Angeles": {"winter": 15, "spring": 18, "summer": 25, "autumn": 20},
            "Singapore": {"winter": 27, "spring": 28, "summer": 28, "autumn": 27},
            "Mumbai": {"winter": 25, "spring": 30, "summer": 35, "autumn": 30},
            "Cairo": {"winter": 15, "spring": 25, "summer": 35, "autumn": 25},
            "Mexico City": {"winter": 12, "spring": 18, "summer": 20, "autumn": 15},
        }

        # Сопоставление месяцев с сезонами
        month_to_season = {12: "winter", 1: "winter", 2: "winter",
                           3: "spring", 4: "spring", 5: "spring",
                           6: "summer", 7: "summer", 8: "summer",
                           9: "autumn", 10: "autumn", 11: "autumn"}

        # Генерация данных о температуре
        def generate_realistic_temperature_data(cities, num_years=10):
            dates = pd.date_range(start="2010-01-01", periods=365 * num_years, freq="D")
            data = []

            for city in cities:
                for date in dates:
                    season = month_to_season[date.month]
                    mean_temp = seasonal_temperatures[city][season]
                    # Добавляем случайное отклонение
                    temperature = np.random.normal(loc=mean_temp, scale=5)
                    data.append({"city": city, "timestamp": date, "temperature": temperature})

            df = pd.DataFrame(data)
            df['season'] = df['timestamp'].dt.month.map(lambda x: month_to_season[x])
            return df

        # Генерация данных
        data = generate_realistic_temperature_data(list(seasonal_temperatures.keys()))
        data.to_csv(file_path, index=False)


def test(file_path: str, parallel: bool = True):
    file_path = os.path.join(os.path.dirname(__file__), 'files_folder', file_path)
    start_time = time.time()
    data_manager = DataManager(file_path, parallel)
    data_manager.detect_anomalies("Moscow")
    print("Время выполнения (parallel={}): {}".format(parallel, time.time() - start_time))


if __name__ == "__main__":
    ensure_file()
    test('temperature_data.csv', parallel=True)
    test('temperature_data.csv', parallel=False)

###################### ИССЛЕДОВАНИЕ РАСПАРАЛЛЕЛИВАНИЯ АНАЛИЗ ДАННЫХ #########################

# Примечание: для проведения анализа, автор проекта написал метод test, который в зависимости
# от аргументов будет либо последовательно, либо параллельно проводить анализиз данных.

# В экспериментах получились следующие результаты:
# Время выполнения (parallel=True): 0.028001070022583008 (параллельный подход)
# Время выполнения (parallel=False): 0.1869981288909912 (последовательный подход)

# Таким образом, делаем вывод, что распаралелливание анализа данных действительно
# дает прирост к производительности в ~6 раз,поэтому в далее в проекте будут использован подход,
# использующий распаралелливание.

# P.S. читатель может сам провести тесты, запустив этот файл (data_manager.py) в питоне.
# P.P.S. автор оставил возможность изменить режим работы с данными в проекте с параллельного на последовательный,
# для это нужно в файле app.py изменить 177 строку на "data_manager = DataManager(parallel=False)".