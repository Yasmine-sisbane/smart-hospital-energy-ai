import requests
from config import LATITUDE, LONGITUDE, TIMEZONE

def safe_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def get_weather_data():
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LATITUDE}"
        f"&longitude={LONGITUDE}"
        f"&timezone={TIMEZONE}"
        "&current=temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m"
        "&wind_speed_unit=ms"
    )

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        data = response.json()

        current = data.get("current", {})

        return {
            "outside_temp_c": safe_float(current.get("temperature_2m")),
            "outside_humidity_pct": safe_float(current.get("relative_humidity_2m")),
            "outside_pressure_hpa": safe_float(current.get("pressure_msl")),
            "wind_speed_mps": safe_float(current.get("wind_speed_10m")),
        }

    except Exception as e:
        print(f"Erreur API météo: {e}")

        return {
            "outside_temp_c": None,
            "outside_humidity_pct": None,
            "outside_pressure_hpa": None,
            "wind_speed_mps": None,
        }