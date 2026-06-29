"""Mock tools for testing and development."""

import asyncio
import random
import time
from typing import Optional, Dict, Any, List
from datetime import datetime

from fancy_llm_router.tools.base import BaseTool


class MockTool(BaseTool):
    """
    A mock tool that simulates various behaviors for testing.
    
    This tool can be configured to:
    - Return static responses
    - Return dynamic responses based on input
    - Simulate delays
    - Simulate errors
    - Generate random data
    """
    
    def __init__(
        self,
        tool_id: str = "mock_tool",
        name: str = "Mock Tool",
        description: str = "A mock tool for testing",
        response: Any = None,
        response_function: Optional[callable] = None,
        delay_seconds: float = 0.0,
        error_rate: float = 0.0,
        error_message: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            tool_id=tool_id,
            name=name,
            description=description,
            **kwargs
        )
        
        self.response = response
        self.response_function = response_function
        self.delay_seconds = delay_seconds
        self.error_rate = error_rate
        self.error_message = error_message or "Mock tool error"
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Input to the mock tool",
                },
                "value": {
                    "type": ["string", "number", "boolean", "object", "array"],
                    "description": "A value to process",
                },
            },
            "additionalProperties": True,
        }
    
    async def execute(
        self,
        parameters: Dict[str, Any],
        call_id: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Execute the mock tool."""
        # Simulate delay
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)
        
        # Simulate error
        if self.error_rate > 0 and random.random() < self.error_rate:
            raise Exception(self.error_message)
        
        # Return configured response
        if self.response_function:
            return self.response_function(parameters, call_id, **kwargs)
        
        if self.response is not None:
            return self.response
        
        # Default: echo the input
        return {
            "success": True,
            "input": parameters.get("input"),
            "value": parameters.get("value"),
            "timestamp": datetime.utcnow().isoformat(),
            "call_id": call_id,
        }


class EchoTool(MockTool):
    """A mock tool that echoes its input."""
    
    def __init__(self, **kwargs):
        super().__init__(
            tool_id="echo_tool",
            name="Echo Tool",
            description="Echoes the input back to the caller",
            **kwargs
        )
    
    async def execute(
        self,
        parameters: Dict[str, Any],
        call_id: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Echo the input."""
        return {
            "echo": parameters.get("input", ""),
            "call_id": call_id,
        }


class RandomTool(MockTool):
    """A mock tool that generates random data."""
    
    def __init__(self, **kwargs):
        super().__init__(
            tool_id="random_tool",
            name="Random Tool",
            description="Generates random data",
            **kwargs
        )
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["integer", "float", "string", "boolean", "choice"],
                    "description": "Type of random value to generate",
                },
                "min": {
                    "type": "number",
                    "description": "Minimum value (for numbers)",
                },
                "max": {
                    "type": "number",
                    "description": "Maximum value (for numbers)",
                },
                "length": {
                    "type": "integer",
                    "description": "Length of string",
                },
                "choices": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Choices for random selection",
                },
            },
        }
    
    async def execute(
        self,
        parameters: Dict[str, Any],
        call_id: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Generate random data."""
        type_ = parameters.get("type", "integer")
        
        if type_ == "integer":
            min_val = parameters.get("min", 0)
            max_val = parameters.get("max", 100)
            return random.randint(min_val, max_val)
        
        elif type_ == "float":
            min_val = parameters.get("min", 0.0)
            max_val = parameters.get("max", 1.0)
            return random.uniform(min_val, max_val)
        
        elif type_ == "string":
            length = parameters.get("length", 10)
            chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            return ''.join(random.choice(chars) for _ in range(length))
        
        elif type_ == "boolean":
            return random.choice([True, False])
        
        elif type_ == "choice":
            choices = parameters.get("choices", ["a", "b", "c"])
            return random.choice(choices)
        
        return None


class DelayTool(MockTool):
    """A mock tool that simulates delays."""
    
    def __init__(self, **kwargs):
        super().__init__(
            tool_id="delay_tool",
            name="Delay Tool",
            description="Simulates processing delays",
            **kwargs
        )
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "number",
                    "description": "Number of seconds to delay",
                    "minimum": 0,
                },
            },
            "required": ["seconds"],
        }
    
    async def execute(
        self,
        parameters: Dict[str, Any],
        call_id: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Simulate a delay."""
        seconds = parameters.get("seconds", 1.0)
        await asyncio.sleep(seconds)
        
        return {
            "delayed_for": seconds,
            "message": f"Delayed for {seconds} seconds",
            "call_id": call_id,
        }


class ErrorTool(MockTool):
    """A mock tool that always fails."""
    
    def __init__(self, **kwargs):
        super().__init__(
            tool_id="error_tool",
            name="Error Tool",
            description="Always fails with an error",
            **kwargs
        )
    
    async def execute(
        self,
        parameters: Dict[str, Any],
        call_id: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Always raise an error."""
        error_type = parameters.get("error_type", "Exception")
        error_message = parameters.get("error_message", "Intentional error for testing")
        
        if error_type == "ValueError":
            raise ValueError(error_message)
        elif error_type == "TypeError":
            raise TypeError(error_message)
        elif error_type == "KeyError":
            raise KeyError(error_message)
        else:
            raise Exception(error_message)
