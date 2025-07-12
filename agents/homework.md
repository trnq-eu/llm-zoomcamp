# Homework for agents workshop

## Question 1

**Question**: What did you put in TODO3?
**Answer**: city

## Question 2

**Answer**:
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

## Question 3
**Question**: What's the version of FastMCP you installed?
**Answer**: 2.10.5

## Question 4
**Answer**: server.py:1371

## Question 5
{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {"roots": {"listChanged": true}, "sampling": {}}, "clientInfo": {"name": "test-client", "version": "1.0.0"}}}

{"jsonrpc": "2.0", "method": "notifications/initialized"}

{"jsonrpc": "2.0", "id": 2, "method": "tools/list"}

{"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "get_weather", "arguments": {city: berlin}}}

{"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "get_weather", "arguments": {city: berlin}}}


**Answer**: CallToolResult(content=[TextContent(type='text', text='20.0', annotations=None, meta=None)], structured_content={'result': 20.0}, data=20.0, is_error=False)