"""Base class for all tools."""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union
from datetime import datetime

from fancy_llm_router.schemas.tools import ToolDefinition, ToolCall, ToolResult

logger = logging.getLogger(__name__)


class ToolError(Exception):
    """Error in tool execution."""
    pass


class ToolTimeoutError(ToolError):
    """Tool execution timed out."""
    pass


class BaseTool(ABC):
    """
    Abstract base class for all tools.
    
    Tools are external resources that LLMs can use to perform tasks.
    They provide a standardized interface for the LLM to interact with
    external systems, APIs, databases, files, etc.
    
    All tools must implement:
    - definition: Tool metadata and configuration
    - execute: The actual tool execution logic
    """
    
    def __init__(
        self,
        tool_id: str,
        name: str,
        description: str,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ):
        self.tool_id = tool_id
        self.name = name
        self.description = description
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.extra_config = kwargs
        
        # Tool definition
        self._definition: Optional[ToolDefinition] = None
        
        # Request tracking
        self._request_counter = 0
    
    @property
    def definition(self) -> ToolDefinition:
        """Get the tool definition."""
        if self._definition is None:
            self._definition = self._create_definition()
        return self._definition
    
    def _create_definition(self) -> ToolDefinition:
        """Create the tool definition. Override in subclasses."""
        return ToolDefinition(
            tool_id=self.tool_id,
            name=self.name,
            tool_type="custom",
            description=self.description,
            parameters=self._get_parameters_schema(),
            required_parameters=self._get_required_parameters(),
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            metadata=self.extra_config,
        )
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for tool parameters. Override in subclasses."""
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
    
    def _get_required_parameters(self) -> List[str]:
        """Get list of required parameter names. Override in subclasses."""
        return []
    
    def _validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize tool parameters."""
        # Check required parameters
        required = self._get_required_parameters()
        for param in required:
            if param not in parameters:
                raise ToolError(f"Missing required parameter: {param}")
        
        # Check for extra parameters
        schema = self._get_parameters_schema()
        properties = schema.get("properties", {})
        for param in parameters:
            if param not in properties:
                logger.warning(f"Unknown parameter '{param}' for tool {self.tool_id}")
        
        return parameters
    
    def _generate_call_id(self) -> str:
        """Generate a unique call ID."""
        self._request_counter += 1
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        return f"{self.tool_id}-{timestamp}-{self._request_counter}"
    
    @abstractmethod
    async def execute(
        self,
        parameters: Dict[str, Any],
        call_id: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Execute the tool with the given parameters.
        
        Args:
            parameters: Dictionary of parameter values
            call_id: Optional call ID for tracking
            **kwargs: Additional arguments
            
        Returns:
            The tool result (type depends on the tool)
        """
        pass
    
    async def call(
        self,
        parameters: Dict[str, Any],
        call_id: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """
        Call the tool and return a standardized ToolResult.
        
        This method handles:
        - Parameter validation
        - Retry logic
        - Timeout handling
        - Error handling
        - Metrics tracking
        """
        call_id = call_id or self._generate_call_id()
        start_time = time.time()
        
        # Validate parameters
        try:
            parameters = self._validate_parameters(parameters)
        except ToolError as e:
            return ToolResult(
                call_id=call_id,
                tool_id=self.tool_id,
                tool_name=self.name,
                output=None,
                is_error=True,
                error_message=str(e),
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                duration_ms=0,
            )
        
        # Execute with retries
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                # Execute the tool
                result = await asyncio.wait_for(
                    self.execute(parameters, call_id, **kwargs),
                    timeout=self.timeout_seconds
                )
                
                end_time = time.time()
                
                return ToolResult(
                    call_id=call_id,
                    tool_id=self.tool_id,
                    tool_name=self.name,
                    output=result,
                    is_error=False,
                    started_at=datetime.fromtimestamp(start_time),
                    completed_at=datetime.utcnow(),
                    duration_ms=(end_time - start_time) * 1000,
                )
                
            except asyncio.TimeoutError:
                last_error = ToolTimeoutError(
                    f"Tool {self.tool_id} timed out after {self.timeout_seconds}s"
                )
            except Exception as e:
                last_error = ToolError(f"Tool {self.tool_id} failed: {e}")
            
            # Retry if not last attempt
            if attempt < self.max_retries:
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        # All retries failed
        end_time = time.time()
        return ToolResult(
            call_id=call_id,
            tool_id=self.tool_id,
            tool_name=self.name,
            output=None,
            is_error=True,
            error_message=str(last_error),
            started_at=datetime.fromtimestamp(start_time),
            completed_at=datetime.utcnow(),
            duration_ms=(end_time - start_time) * 1000,
        )
    
    async def health_check(self) -> bool:
        """Check if the tool is available and healthy."""
        return True
    
    async def close(self):
        """Clean up resources."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.tool_id})"


class ToolRegistry:
    """
    Registry for managing tools.
    
    Features:
    - Register and unregister tools
    - Look up tools by ID or name
    - Group tools by category
    - Health checking
    """
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._tools_by_name: Dict[str, BaseTool] = {}
        self._tool_groups: Dict[str, List[BaseTool]] = {}
    
    def register(
        self,
        tool: BaseTool,
        group: Optional[str] = None
    ):
        """Register a tool."""
        if tool.tool_id in self._tools:
            logger.warning(f"Tool {tool.tool_id} already registered, replacing")
        
        self._tools[tool.tool_id] = tool
        self._tools_by_name[tool.name] = tool
        
        if group:
            self._tool_groups.setdefault(group, []).append(tool)
        
        logger.info(f"Registered tool: {tool.tool_id} ({tool.name})")
    
    def unregister(self, tool_id: str):
        """Unregister a tool."""
        if tool_id in self._tools:
            tool = self._tools[tool_id]
            del self._tools[tool_id]
            self._tools_by_name.pop(tool.name, None)
            
            # Remove from groups
            for group, tools in self._tool_groups.items():
                if tool in tools:
                    tools.remove(tool)
                    if not tools:
                        del self._tool_groups[group]
            
            logger.info(f"Unregistered tool: {tool_id}")
    
    def get(self, tool_id: str) -> Optional[BaseTool]:
        """Get a tool by ID."""
        return self._tools.get(tool_id)
    
    def get_by_name(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools_by_name.get(name)
    
    def list_tools(
        self,
        group: Optional[str] = None
    ) -> List[BaseTool]:
        """List all tools, optionally filtered by group."""
        if group:
            return self._tool_groups.get(group, [])
        return list(self._tools.values())
    
    def list_tool_definitions(self) -> List[ToolDefinition]:
        """List definitions for all registered tools."""
        return [tool.definition for tool in self._tools.values()]
    
    async def check_health(self) -> Dict[str, bool]:
        """Check health of all registered tools."""
        results = {}
        tasks = []
        
        for tool_id, tool in self._tools.items():
            tasks.append((tool_id, tool.health_check()))
        
        for tool_id, task in tasks:
            try:
                results[tool_id] = await task
            except Exception:
                results[tool_id] = False
        
        return results
    
    async def close_all(self):
        """Close all registered tools."""
        tasks = []
        for tool in self._tools.values():
            tasks.append(tool.close())
        
        await asyncio.gather(*tasks)
        self._tools.clear()
        self._tools_by_name.clear()
        self._tool_groups.clear()


# Global tool registry instance
tool_registry = ToolRegistry()
