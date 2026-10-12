import os

import httpx
from fastapi import HTTPException


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": (
            "Get current weather or a forecast for a named location. Use this "
            "for questions about current or upcoming weather."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City or place name; include region/country if useful.",
                },
                "forecast_days": {
                    "type": "integer",
                    "description": "Forecast length from 1 to 3 days; use 1 for current conditions.",
                },
            },
            "required": ["location", "forecast_days"],
            "additionalProperties": False,
        },
    },
}

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": (
            "Search the internet for recent, changing, or otherwise "
            "up-to-date information. Cite source URLs in your answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A concise search query.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

GENERAL_CHAT_TOOLS = [WEATHER_TOOL, SEARCH_TOOL]

_CURRENT_WEATHER_FIELDS = (
    "last_updated",
    "temp_c",
    "temp_f",
    "is_day",
    "condition",
    "wind_kph",
    "humidity",
    "feelslike_c",
    "precip_mm",
)
_DAILY_WEATHER_FIELDS = (
    "maxtemp_c",
    "mintemp_c",
    "avgtemp_c",
    "daily_chance_of_rain",
    "daily_chance_of_snow",
    "condition",
)


def _get_weather(location: str, forecast_days: int) -> dict:
    location = location.strip()
    if not location or len(location) > 200:
        raise HTTPException(status_code=422, detail="A valid weather location is required")
    if not 1 <= forecast_days <= 3:
        raise HTTPException(status_code=422, detail="forecast_days must be between 1 and 3")
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="WEATHER_API_KEY is not configured")

    try:
        response = httpx.get(
            "https://api.weatherapi.com/v1/forecast.json",
            params={
                "key": api_key,
                "q": location,
                "days": forecast_days,
                "aqi": "no",
                "alerts": "no",
            },
            timeout=15,
        )
        response.raise_for_status()
        weather_data = response.json()
        if not isinstance(weather_data, dict) or not isinstance(
            weather_data.get("location"), dict
        ):
            raise ValueError("Invalid weather response")
        place = weather_data["location"]
        current = weather_data.get("current")
        forecast = weather_data.get("forecast")
        forecast_days_data = (
            forecast.get("forecastday", []) if isinstance(forecast, dict) else []
        )
        if not isinstance(forecast_days_data, list):
            raise ValueError("Invalid weather forecast")
        return {
            "location": ", ".join(
                part
                for part in (place.get("name"), place.get("region"), place.get("country"))
                if part
            ),
            "current": {
                key: current[key]
                for key in _CURRENT_WEATHER_FIELDS
                if isinstance(current, dict) and key in current
            },
            "forecast": [
                {
                    "date": day.get("date"),
                    "day": {
                        key: day_data[key]
                        for key in _DAILY_WEATHER_FIELDS
                        if isinstance(day_data, dict) and key in day_data
                    },
                }
                for day in forecast_days_data[:3]
                if isinstance(day, dict)
                for day_data in [day.get("day")]
            ],
        }
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
        raise HTTPException(
            status_code=502,
            detail="Weather data could not be retrieved",
        ) from error


def _search_web(query: str) -> dict:
    query = query.strip()
    if not query or len(query) > 500:
        raise HTTPException(status_code=422, detail="A valid web search query is required")

    api_key = os.getenv("TAVILY_SEARCH_API_KEY") or os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="TAVILY_SEARCH_API_KEY is not configured",
        )

    try:
        response = httpx.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": 3,
            },
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Invalid web-search response")
        results = data.get("results", [])
        if not isinstance(results, list) or any(
            not isinstance(result, dict) for result in results[:5]
        ):
            raise ValueError("Invalid web-search results")
        return {
            "query": query,
            "results": [
                {
                    "title": str(result.get("title", ""))[:200],
                    "url": str(result.get("url", ""))[:500],
                    "content": str(result.get("content", ""))[:350],
                }
                for result in results[:3]
            ],
        }
    except (httpx.HTTPError, ValueError, TypeError) as error:
        raise HTTPException(
            status_code=502,
            detail="Web search could not be completed",
        ) from error


def execute_general_chat_tool(name: str, arguments: dict) -> dict:
    if name == "get_weather":
        location = arguments.get("location")
        forecast_days = arguments.get("forecast_days")
        if not isinstance(location, str) or isinstance(forecast_days, bool) or not isinstance(
            forecast_days, int
        ):
            raise HTTPException(status_code=502, detail="The assistant requested invalid weather inputs")
        return _get_weather(location, forecast_days)
    if name == "search_web":
        query = arguments.get("query")
        if not isinstance(query, str):
            raise HTTPException(status_code=502, detail="The assistant requested invalid search inputs")
        return _search_web(query)
    raise HTTPException(status_code=502, detail="The assistant requested an unsupported tool")
