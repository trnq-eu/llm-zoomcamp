import asyncio
from fastmcp import FastMCP, Client
from weather_server import get_weather

mcp = FastMCP("My MCP Server")

# @mcp.tool
# def greet(name: str) -> str:
#     return f"Hello, {name}!"

known_weather_data = {
    'berlin': 20.0
}

@mcp.tool
def get_weather(city: str) -> float:
    """
    Retrieves the temperature for a specified city.

    Parameters:
        city (str): The name of the city for which to retrieve weather data.

    Returns:
        float: The temperature associated with the city.
    """
    city = city.strip().lower()

    if city in known_weather_data:
        return known_weather_data[city]

    # return round(random.uniform(-5, 35), 1)
    return 35.0

# client = Client(mcp)


async def call_tool(city: str):
    async with client:
        result = await client.call_tool("get_weather", {"city": city})
        print(result)

# asyncio.run(call_tool("Berlin"))


# tools = client.list_tools()
#     # tools -> list[mcp.types.Tool]
# for tool in tools:
#         print(f"Tool: {tool.name}")
#         print(f"Description: {tool.description}")
#         if tool.inputSchema:
#             print(f"Parameters: {tool.inputSchema}")

async def main():
    async with Client(mcp) as mcp_client:
        tools = await mcp_client.list_tools()
        print(tools)

if __name__ == "__main__":
    asyncio.run(main())