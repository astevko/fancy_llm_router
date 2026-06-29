"""Tools and external resources for LLMs."""

from fancy_llm_router.tools.base import BaseTool, ToolRegistry
from fancy_llm_router.tools.mock import MockTool
from fancy_llm_router.tools.file import FileTool
from fancy_llm_router.tools.web import WebTool
from fancy_llm_router.tools.database import DatabaseTool

__all__ = [
    "BaseTool",
    "ToolRegistry",
    "MockTool",
    "FileTool",
    "WebTool",
    "DatabaseTool",
]
