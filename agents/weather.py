import random

known_weather_data = {
    'berlin': 20.0
}

def get_weather(city: str) -> float:
    city = city.strip().lower()

    if city in known_weather_data:
        return known_weather_data[city]
    
    return round(random.uniform(-5, 35), 1)

get_weather_tool = {
    "type": "function",
    "name": "get_weather",
    "description": "a function to look for weather data by city",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "the name of the city to look for"
            }
        },
        "required": "city",
        "additionalProperties": False
    }
}

def set_weather(city: str, temp: float) -> None:
    city = city.strip().lower()
    known_weather_data[city] = temp
    return 'OK'


set_weather_tool = {
    "type": "function",
    "name": "set_weather",
    "description": "a function to append temperature for a new city into the weather_data dictionary",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "the name of the city to add"
            },
            "temp": {
                "type": "float",
                "description": "the temperature for the city"
            }
        },
        "required": ["city", "temp"],
        "additionalProperties": False
    }
}