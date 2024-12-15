import requests
import aiohttp
import asyncio
import time
from typing import List, Dict, Any
from cachetools import cached, TTLCache


class OpenWeatherClient:
    def __init__(self, api_key: str) -> None:
        self.api_key: str = api_key
        self.base_url: str = "http://api.openweathermap.org/data/2.5/weather"

    def check_key(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}?q=London&appid={self.api_key}")
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException:
            return False

    @cached(cache=TTLCache(maxsize=100, ttl=60*60))
    def get_weather_sync(self, city: str) -> Dict[str, Any]:
        try:
            response: requests.Response = requests.get(f"{self.base_url}?q={city}&appid={self.api_key}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    @cached(cache=TTLCache(maxsize=100, ttl=60 * 60))
    async def get_weather_async(self, city: str) -> Dict[str, Any]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}?q={city}&appid={self.api_key}&units=metric&lang=ru") as response:
                    response.raise_for_status()
                    return await response.json()
        except aiohttp.ClientError as e:
            return {"error": str(e)}

    async def get_weather_async_multiple(self, cities: List[str]) -> List[Dict[str, Any]]:
        tasks = [self.get_weather_async(city) for city in cities]
        return await asyncio.gather(*tasks)

    @staticmethod
    def test_request_speed(api_key: str, cities: List[str]) -> Dict[str, Any]:
        client = OpenWeatherClient(api_key)

        start_time_sync = time.time()
        sync_results = [client.get_weather_sync(city) for city in cities]
        sync_duration = time.time() - start_time_sync

        start_time_async = time.time()
        async_results = asyncio.run(client.get_weather_async_multiple(cities))
        async_duration = time.time() - start_time_async

        print(f"Синхронные запросы: {sync_duration:.2f} сек")
        print(f"Асинхронные запросы: {async_duration:.2f} сек")

        return {
            "sync_duration_sec": sync_duration,
            "sync_results": sync_results,
            "async_duration_sec": async_duration,
            "async_results": async_results,
        }


if __name__ == "__main__":
    key = ""
    cities = [
        "London",
        "Paris", "Berlin", "Madrid", "Rome",
        "Moscow", "New York", "Tokyo", "Beijing", "Delhi",
        "São Paulo", "Cairo", "Sydney", "Toronto", "Dubai",
        "Mexico City", "Buenos Aires", "Cape Town", "Seoul", "Bangkok",
        "Lagos", "Istanbul", "Rio de Janeiro", "Jakarta", "Tehran"
    ]
    _ = OpenWeatherClient.test_request_speed(key, cities)

# Синхронные запросы: 7.79 сек
# Асинхронные запросы: 0.35 сек

###################### ИССЛЕДОВАНИЕ СПОСОБОВ ЗАПРОСОВ #########################

# Примечание: для проведения анализа, автор проекта написал метод OpenWeatherClient.test_request_speed,
# который совершает синхронные и асинхронные запросы к одному и тому же списку городов и замеряет время.

# В экспериментах получились следующие результаты:
# Синхронные запросы: 8.18 сек (синхронный подход)
# Асинхронные запросы: 0.52 сек (асинхронный подход)

# Таким образом, делаем вывод, что использование асинхронных запросов действтильно дает прирост
# к скорости работы.

# P.S. читатель может сам провести тесты, запустив этот файл (open_weather_api.py) в питоне,
# указав API KEY в переменную key, 69 строка.