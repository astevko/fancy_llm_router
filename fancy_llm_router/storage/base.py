"""Base class for storage backends."""

import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from datetime import datetime

from fancy_llm_router.schemas.metrics import (
    RequestMetrics,
    SessionMetrics,
    BenchmarkMetrics,
)


class StorageError(Exception):
    """Error in storage operations."""
    pass


class BaseStorage(ABC):
    """
    Abstract base class for storage backends.
    
    Storage backends are responsible for persisting metrics and other data.
    Implementations can use SQLite, Postgres, MySQL, etc.
    """
    
    def __init__(
        self,
        backend: str,
        **kwargs
    ):
        self.backend = backend
        self.config = kwargs
        self._is_initialized = False
    
    @abstractmethod
    async def initialize(self):
        """Initialize the storage backend (create tables, etc.)."""
        pass
    
    @abstractmethod
    async def store_request_metrics(self, metrics: RequestMetrics):
        """Store request metrics."""
        pass
    
    @abstractmethod
    async def store_session_metrics(self, metrics: SessionMetrics):
        """Store session metrics."""
        pass
    
    @abstractmethod
    async def store_benchmark_metrics(self, metrics: BenchmarkMetrics):
        """Store benchmark metrics."""
        pass
    
    @abstractmethod
    async def get_request_metrics(
        self,
        request_id: str
    ) -> Optional[RequestMetrics]:
        """Get metrics for a specific request."""
        pass
    
    @abstractmethod
    async def get_session_metrics(
        self,
        session_id: str
    ) -> Optional[SessionMetrics]:
        """Get metrics for a specific session."""
        pass
    
    @abstractmethod
    async def get_benchmark_metrics(
        self,
        benchmark_id: str
    ) -> Optional[BenchmarkMetrics]:
        """Get metrics for a specific benchmark."""
        pass
    
    @abstractmethod
    async def query_request_metrics(
        self,
        filters: Dict[str, Any],
        limit: int = 100,
        offset: int = 0,
        order_by: Optional[str] = None,
        descending: bool = True
    ) -> List[RequestMetrics]:
        """Query request metrics with filters."""
        pass
    
    @abstractmethod
    async def query_session_metrics(
        self,
        filters: Dict[str, Any],
        limit: int = 100,
        offset: int = 0,
    ) -> List[SessionMetrics]:
        """Query session metrics with filters."""
        pass
    
    @abstractmethod
    async def get_model_metrics(
        self,
        model_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get aggregated metrics for a model."""
        pass
    
    @abstractmethod
    async def cleanup(
        self,
        older_than: Optional[datetime] = None
    ):
        """Clean up old data."""
        pass
    
    async def close(self):
        """Clean up resources."""
        pass
    
    async def __aenter__(self):
        """Async context manager entry."""
        if not self._is_initialized:
            await self.initialize()
            self._is_initialized = True
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


class StorageFactory:
    """Factory for creating storage backends."""
    
    _backends: Dict[str, BaseStorage] = {}
    
    @classmethod
    def create(
        cls,
        backend: str,
        **kwargs
    ) -> BaseStorage:
        """Create a storage backend."""
        if backend == "sqlite":
            from fancy_llm_router.storage.sqlite import SQLiteStorage
            return SQLiteStorage(backend=backend, **kwargs)
        elif backend == "postgres":
            from fancy_llm_router.storage.postgres import PostgresStorage
            return PostgresStorage(backend=backend, **kwargs)
        else:
            raise StorageError(f"Unknown storage backend: {backend}")
    
    @classmethod
    def register(cls, backend: str, storage: BaseStorage):
        """Register a storage backend."""
        cls._backends[backend] = storage
    
    @classmethod
    def get(cls, backend: str) -> Optional[BaseStorage]:
        """Get a registered storage backend."""
        return cls._backends.get(backend)
