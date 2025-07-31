from mcp import MCPServer, Tool
from mcp.types import TextContent

class CalculatorMCPServer(MCPServer):
    def __init__(self):
        super().__init__("calculator-server")
        
        # Register a tool
        self.add_tool(Tool(
            name="add_numbers",
            description="Add two numbers together",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["a", "b"]
            }
        ))
    
    async def handle_tool_call(self, tool_name: str, arguments: dict):
        if tool_name == "add_numbers":
            result = arguments["a"] + arguments["b"]
            return TextContent(text=f"The sum is: {result}")
        
        raise ValueError(f"Unknown tool: {tool_name}")

# Start the server
server = CalculatorMCPServer()
server.run()