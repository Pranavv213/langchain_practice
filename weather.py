import requests
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent

# ----------------------------
# TOOL 1: COORDINATES
# ----------------------------
@tool
def find_coordinates(city: str) -> dict:
    """Get latitude and longitude for a city."""
    headers = {"User-Agent": "WeatherAgent/1.0 (myemail@example.com)"}
    response = requests.get(
        f"https://nominatim.openstreetmap.org/search?q={city}&format=json",
        headers=headers
    )

    if response.status_code == 200 and response.json():
        data = response.json()[0]
        return {
            "city": city,
            # CRITICAL FIX: Convert strings to floats so the next API doesn't crash
            "lat": float(data["lat"]), 
            "lon": float(data["lon"])
        }

    return {"error": f"Could not find coordinates for {city}"}


# ----------------------------
# TOOL 2: WEATHER
# ----------------------------
@tool
def get_weather_info(lat: float, lon: float) -> dict:  # Updated type hints to float
    """Get current weather using latitude and longitude."""
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}&current_weather=true"
    )

    # CRITICAL FIX: Safely check if the response is actually JSON before decoding
    if response.status_code == 200:
        try:
            return response.json()
        except ValueError:
            return {"error": "API response was not valid JSON."}

    return {"error": f"Could not fetch weather data. Status code: {response.status_code}"}


# ----------------------------
# LLM
# ----------------------------
llm = ChatOllama(model="llama3.2:3b", temperature=0)


# ----------------------------
# IMPORTANT SYSTEM PROMPT
# ----------------------------
system_prompt = """
You are a weather assistant.

RULES:
1. For any city, ALWAYS call find_coordinates first.
2. Then ALWAYS call get_weather_info using lat and lon.
3. NEVER answer without using tools.
4. Final answer must be simple human-readable weather.
"""


# ----------------------------
# AGENT
# ----------------------------
agent = create_agent(
    model=llm,
    tools=[find_coordinates, get_weather_info],
    system_prompt=system_prompt
)


# ----------------------------
# LOOP
# ----------------------------
while True:
    query = input("\nAsk (or type exit): ")

    if query.lower() == "exit":
        break

    result = agent.invoke({
        "messages": [
            {"role": "user", "content": query}
        ]
    })

    print("\nResponse:", result["messages"][-1].content)
