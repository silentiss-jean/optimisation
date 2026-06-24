from typing import Dict, Any, List, Type
from .base_tool import Tool

class ToolManager:
    """
    Manages the registration and execution of all available tools.
    Acts as a central hub for the Agent to interact with various capabilities.
    """
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register_tool(self, tool: Tool) -> None:
        """Registers a tool instance into the manager."""
        self._tools[tool.name] = tool

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Returns a list of all registered tools with their definitions."""
        return [tool.definition for tool in self._tools.values()]

    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """
        Finds and executes a tool by name.
        
        Args:
            tool_name: The name of the tool to call (e.g., 'filesystem_operation').
            **kwargs: Arguments passed to the tool's execute method.
            
        Returns:
            The result of the tool execution as a string.
        """
        if tool_name not in self._tools:
            return f"Error: Tool '{tool_name}' is not registered."
        
        tool = self._tools[tool_name]
        # The implementation logic for specific tools might differ slightly, 
        # but the standard interface should be followed.
        return tool.execute(**kwargs)

    def list_tools(self) -> List[str]:
        """Returns a list of registered tool names."""
        return list(self._tools.keys())

# Initialize and register tools automatically (can be expanded as more are added)
manager = ToolManager()
from .filesystem_tool import FileSystemTool
from .command_line_tool import CommandLineTool
from .web_browser_tool import WebBrowserTool
from .web_scraper_tool import WebScraperTool

# Register the tools
manager.register_tool(FileSystemTool())
manager.register_tool(CommandLineTool())
manager.register_tool(WebBrowserTool())
manager.register_tool(WebScraperTool())

# Export instance for use in other modules
tool_manager = manager
