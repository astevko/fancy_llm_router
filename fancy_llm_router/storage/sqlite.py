"""SQLite storage backend for metrics."""

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import aiosqlite
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from fancy_llm_router.storage.base import BaseStorage, StorageError
from fancy_llm_router.schemas.metrics import (
    RequestMetrics,
    SessionMetrics,
    BenchmarkMetrics,
    TokenUsage,
    CostMetrics,
    LatencyMetrics,
    QualityMetrics,
    ModelMetrics,
)

logger = logging.getLogger(__name__)

Base = declarative_base()


class RequestMetricsDB(Base):
    """SQLAlchemy model for request metrics."""
    __tablename__ = "request_metrics"
    
    id = Column(Integer, primary_key=True)
    request_id = Column(String, unique=True, index=True)
    session_id = Column(String, index=True)
    prompt_hash = Column(String, index=True)
    
    # Timestamps
    created_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Request details
    request_type = Column(String)
    
    # Model info
    model_id = Column(String)
    model_provider = Column(String)
    model_version = Column(String)
    model_parameters = Column(Integer)
    context_window = Column(Integer)
    
    # Token usage
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer)
    total_tokens = Column(Integer)
    
    # Cost
    input_token_cost = Column(Float)
    output_token_cost = Column(Float)
    total_cost = Column(Float)
    currency = Column(String, default="USD")
    
    # Latency
    time_to_first_token_ms = Column(Float)
    time_to_complete_ms = Column(Float)
    
    # Quality
    relevance_score = Column(Float)
    accuracy_score = Column(Float)
    coherence_score = Column(Float)
    helpfulness_score = Column(Float)
    human_rating = Column(Float)
    is_error = Column(Integer, default=0)
    error_type = Column(String)
    error_message = Column(Text)
    
    # Metadata
    git_commit = Column(String)
    environment = Column(String)
    user_id = Column(String)
    metadata = Column(JSON)
    tags = Column(JSON)


class SessionMetricsDB(Base):
    """SQLAlchemy model for session metrics."""
    __tablename__ = "session_metrics"
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String, unique=True, index=True)
    
    # Timestamps
    created_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Aggregated metrics
    total_requests = Column(Integer)
    total_prompt_tokens = Column(Integer)
    total_completion_tokens = Column(Integer)
    total_tokens = Column(Integer)
    total_input_cost = Column(Float)
    total_output_cost = Column(Float)
    total_cost = Column(Float)
    avg_time_to_complete_ms = Column(Float)
    
    # Session info
    chain_length = Column(Integer)
    is_complete = Column(Integer, default=0)
    final_output = Column(Text)
    
    # Metadata
    git_commit = Column(String)
    metadata = Column(JSON)


class BenchmarkMetricsDB(Base):
    """SQLAlchemy model for benchmark metrics."""
    __tablename__ = "benchmark_metrics"
    
    id = Column(Integer, primary_key=True)
    benchmark_id = Column(String, unique=True, index=True)
    benchmark_name = Column(String)
    
    # Model info
    model_id = Column(String)
    model_provider = Column(String)
    model_version = Column(String)
    
    # Aggregated metrics
    avg_prompt_tokens = Column(Float)
    avg_completion_tokens = Column(Float)
    avg_total_tokens = Column(Float)
    avg_input_cost = Column(Float)
    avg_output_cost = Column(Float)
    avg_total_cost = Column(Float)
    avg_time_to_complete_ms = Column(Float)
    avg_time_to_first_token_ms = Column(Float)
    avg_relevance_score = Column(Float)
    avg_accuracy_score = Column(Float)
    avg_coherence_score = Column(Float)
    
    # Statistical measures
    std_cost = Column(Float)
    std_latency = Column(Float)
    std_quality = Column(Float)
    
    # Comparison
    baseline_model_id = Column(String)
    cost_savings_pct = Column(Float)
    quality_improvement_pct = Column(Float)
    latency_improvement_pct = Column(Float)
    
    # Info
    num_runs = Column(Integer)
    
    # Timestamps
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    
    # Metadata
    metadata = Column(JSON)


class SQLiteStorage(BaseStorage):
    """
    SQLite storage backend for metrics.
    
    This backend stores all metrics in a SQLite database for easy
    querying and analysis.
    """
    
    def __init__(
        self,
        backend: str = "sqlite",
        db_path: str = "data/metrics.db",
        **kwargs
    ):
        super().__init__(backend=backend, **kwargs)
        self.db_path = db_path
        self._engine = None
        self._async_engine = None
    
    async def initialize(self):
        """Initialize the SQLite database."""
        # Ensure directory exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        
        # Create sync engine for table creation
        self._engine = create_engine(f"sqlite:///{self.db_path}")
        
        # Create tables
        Base.metadata.create_all(self._engine)
        
        # Create async engine for operations
        self._async_engine = create_engine(f"sqlite+aiosqlite:///{self.db_path}")
        
        logger.info(f"Initialized SQLite storage at {self.db_path}")
    
    async def store_request_metrics(self, metrics: RequestMetrics):
        """Store request metrics."""
        if not self._async_engine:
            await self.initialize()
        
        async with self._async_engine.begin() as conn:
            # Convert metrics to DB model
            db_metrics = RequestMetricsDB(
                request_id=metrics.request_id,
                session_id=metrics.session_id,
                prompt_hash=metrics.prompt_hash,
                created_at=metrics.created_at,
                completed_at=metrics.completed_at,
                request_type=metrics.request_type,
                model_id=metrics.model_info.model_id,
                model_provider=metrics.model_info.model_provider,
                model_version=metrics.model_info.model_version,
                model_parameters=metrics.model_info.model_parameters,
                context_window=metrics.model_info.context_window,
                prompt_tokens=metrics.token_usage.prompt_tokens,
                completion_tokens=metrics.token_usage.completion_tokens,
                total_tokens=metrics.token_usage.total_tokens,
                input_token_cost=metrics.cost.input_token_cost,
                output_token_cost=metrics.cost.output_token_cost,
                total_cost=metrics.cost.total_cost,
                currency=metrics.cost.currency,
                time_to_first_token_ms=metrics.latency.time_to_first_token_ms,
                time_to_complete_ms=metrics.latency.time_to_complete_ms,
                relevance_score=metrics.quality.relevance_score,
                accuracy_score=metrics.quality.accuracy_score,
                coherence_score=metrics.quality.coherence_score,
                helpfulness_score=metrics.quality.helpfulness_score,
                human_rating=metrics.quality.human_rating,
                is_error=1 if metrics.quality.is_error else 0,
                error_type=metrics.quality.error_type,
                error_message=metrics.quality.error_message,
                git_commit=metrics.git_commit,
                environment=metrics.environment,
                user_id=metrics.user_id,
                metadata=metrics.metadata,
                tags=metrics.tags,
            )
            
            conn.execute(
                RequestMetricsDB.__table__.insert(),
                db_metrics.__dict__
            )
        
        logger.debug(f"Stored request metrics: {metrics.request_id}")
    
    async def store_session_metrics(self, metrics: SessionMetrics):
        """Store session metrics."""
        if not self._async_engine:
            await self.initialize()
        
        async with self._async_engine.begin() as conn:
            # Convert metrics to DB model
            db_metrics = SessionMetricsDB(
                session_id=metrics.session_id,
                created_at=metrics.created_at,
                completed_at=metrics.completed_at,
                total_requests=metrics.total_requests,
                total_prompt_tokens=metrics.total_token_usage.prompt_tokens,
                total_completion_tokens=metrics.total_token_usage.completion_tokens,
                total_tokens=metrics.total_token_usage.total_tokens,
                total_input_cost=metrics.total_cost.input_token_cost,
                total_output_cost=metrics.total_cost.output_token_cost,
                total_cost=metrics.total_cost.total_cost,
                avg_time_to_complete_ms=metrics.total_latency.time_to_complete_ms,
                chain_length=metrics.chain_length,
                is_complete=1 if metrics.is_complete else 0,
                final_output=metrics.final_output,
                git_commit=metrics.git_commit,
                metadata=metrics.metadata,
            )
            
            conn.execute(
                SessionMetricsDB.__table__.insert(),
                db_metrics.__dict__
            )
        
        logger.debug(f"Stored session metrics: {metrics.session_id}")
    
    async def store_benchmark_metrics(self, metrics: BenchmarkMetrics):
        """Store benchmark metrics."""
        if not self._async_engine:
            await self.initialize()
        
        async with self._async_engine.begin() as conn:
            # Convert metrics to DB model
            db_metrics = BenchmarkMetricsDB(
                benchmark_id=metrics.benchmark_id,
                benchmark_name=metrics.benchmark_name,
                model_id=metrics.model_info.model_id,
                model_provider=metrics.model_info.model_provider,
                model_version=metrics.model_info.model_version,
                avg_prompt_tokens=metrics.avg_token_usage.prompt_tokens,
                avg_completion_tokens=metrics.avg_token_usage.completion_tokens,
                avg_total_tokens=metrics.avg_token_usage.total_tokens,
                avg_input_cost=metrics.avg_cost.input_token_cost,
                avg_output_cost=metrics.avg_cost.output_token_cost,
                avg_total_cost=metrics.avg_cost.total_cost,
                avg_time_to_complete_ms=metrics.avg_latency.time_to_complete_ms,
                avg_time_to_first_token_ms=metrics.avg_latency.time_to_first_token_ms,
                avg_relevance_score=metrics.avg_quality.relevance_score,
                avg_accuracy_score=metrics.avg_quality.accuracy_score,
                avg_coherence_score=metrics.avg_quality.coherence_score,
                std_cost=metrics.std_cost,
                std_latency=metrics.std_latency,
                std_quality=metrics.std_quality,
                baseline_model_id=metrics.baseline_model_id,
                cost_savings_pct=metrics.cost_savings_pct,
                quality_improvement_pct=metrics.quality_improvement_pct,
                latency_improvement_pct=metrics.latency_improvement_pct,
                num_runs=metrics.num_runs,
                created_at=metrics.created_at,
                updated_at=metrics.updated_at,
                metadata=metrics.metadata,
            )
            
            conn.execute(
                BenchmarkMetricsDB.__table__.insert(),
                db_metrics.__dict__
            )
        
        logger.debug(f"Stored benchmark metrics: {metrics.benchmark_id}")
    
    async def get_request_metrics(
        self,
        request_id: str
    ) -> Optional[RequestMetrics]:
        """Get metrics for a specific request."""
        if not self._async_engine:
            await self.initialize()
        
        async with self._async_engine.begin() as conn:
            result = await conn.execute(
                RequestMetricsDB.__table__.select()
                .where(RequestMetricsDB.request_id == request_id)
            )
            row = result.fetchone()
            
            if row:
                return self._row_to_request_metrics(row)
        
        return None
    
    async def get_session_metrics(
        self,
        session_id: str
    ) -> Optional[SessionMetrics]:
        """Get metrics for a specific session."""
        if not self._async_engine:
            await self.initialize()
        
        async with self._async_engine.begin() as conn:
            result = await conn.execute(
                SessionMetricsDB.__table__.select()
                .where(SessionMetricsDB.session_id == session_id)
            )
            row = result.fetchone()
            
            if row:
                return self._row_to_session_metrics(row)
        
        return None
    
    async def get_benchmark_metrics(
        self,
        benchmark_id: str
    ) -> Optional[BenchmarkMetrics]:
        """Get metrics for a specific benchmark."""
        if not self._async_engine:
            await self.initialize()
        
        async with self._async_engine.begin() as conn:
            result = await conn.execute(
                BenchmarkMetricsDB.__table__.select()
                .where(BenchmarkMetricsDB.benchmark_id == benchmark_id)
            )
            row = result.fetchone()
            
            if row:
                return self._row_to_benchmark_metrics(row)
        
        return None
    
    async def query_request_metrics(
        self,
        filters: Dict[str, Any],
        limit: int = 100,
        offset: int = 0,
        order_by: Optional[str] = None,
        descending: bool = True
    ) -> List[RequestMetrics]:
        """Query request metrics with filters."""
        if not self._async_engine:
            await self.initialize()
        
        # Build query
        from sqlalchemy import select, and_, or_, desc, asc
        
        conditions = []
        for field, value in filters.items():
            if field == "model_id":
                conditions.append(RequestMetricsDB.model_id == value)
            elif field == "session_id":
                conditions.append(RequestMetricsDB.session_id == value)
            elif field == "prompt_hash":
                conditions.append(RequestMetricsDB.prompt_hash == value)
            elif field == "start_date":
                conditions.append(RequestMetricsDB.created_at >= value)
            elif field == "end_date":
                conditions.append(RequestMetricsDB.created_at <= value)
            elif field == "is_error":
                conditions.append(RequestMetricsDB.is_error == (1 if value else 0))
        
        query = select(RequestMetricsDB)
        if conditions:
            query = query.where(and_(*conditions))
        
        if order_by:
            if descending:
                query = query.order_by(desc(getattr(RequestMetricsDB, order_by)))
            else:
                query = query.order_by(asc(getattr(RequestMetricsDB, order_by)))
        
        query = query.limit(limit).offset(offset)
        
        async with self._async_engine.begin() as conn:
            result = await conn.execute(query)
            rows = result.fetchall()
            
            return [self._row_to_request_metrics(row) for row in rows]
    
    async def query_session_metrics(
        self,
        filters: Dict[str, Any],
        limit: int = 100,
        offset: int = 0,
    ) -> List[SessionMetrics]:
        """Query session metrics with filters."""
        if not self._async_engine:
            await self.initialize()
        
        # Build query
        from sqlalchemy import select, and_, desc
        
        conditions = []
        for field, value in filters.items():
            if field == "session_id":
                conditions.append(SessionMetricsDB.session_id == value)
            elif field == "start_date":
                conditions.append(SessionMetricsDB.created_at >= value)
            elif field == "end_date":
                conditions.append(SessionMetricsDB.created_at <= value)
            elif field == "is_complete":
                conditions.append(SessionMetricsDB.is_complete == (1 if value else 0))
        
        query = select(SessionMetricsDB)
        if conditions:
            query = query.where(and_(*conditions))
        
        query = query.order_by(desc(SessionMetricsDB.created_at))
        query = query.limit(limit).offset(offset)
        
        async with self._async_engine.begin() as conn:
            result = await conn.execute(query)
            rows = result.fetchall()
            
            return [self._row_to_session_metrics(row) for row in rows]
    
    async def get_model_metrics(
        self,
        model_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get aggregated metrics for a model."""
        if not self._async_engine:
            await self.initialize()
        
        from sqlalchemy import select, func, and_
        
        conditions = [RequestMetricsDB.model_id == model_id]
        if start_date:
            conditions.append(RequestMetricsDB.created_at >= start_date)
        if end_date:
            conditions.append(RequestMetricsDB.created_at <= end_date)
        
        query = select(
            func.count(RequestMetricsDB.id),
            func.sum(RequestMetricsDB.prompt_tokens),
            func.sum(RequestMetricsDB.completion_tokens),
            func.sum(RequestMetricsDB.total_cost),
            func.avg(RequestMetricsDB.time_to_complete_ms),
            func.avg(RequestMetricsDB.relevance_score),
        ).where(and_(*conditions))
        
        async with self._async_engine.begin() as conn:
            result = await conn.execute(query)
            row = result.fetchone()
            
            if row:
                return {
                    "total_requests": row[0],
                    "total_prompt_tokens": row[1] or 0,
                    "total_completion_tokens": row[2] or 0,
                    "total_cost": row[3] or 0.0,
                    "avg_latency_ms": row[4] or 0.0,
                    "avg_quality_score": row[5] or 0.0,
                }
        
        return {}
    
    async def cleanup(
        self,
        older_than: Optional[datetime] = None
    ):
        """Clean up old data."""
        if not self._async_engine:
            await self.initialize()
        
        from sqlalchemy import delete, and_, or_
        
        async with self._async_engine.begin() as conn:
            if older_than:
                # Delete old request metrics
                await conn.execute(
                    delete(RequestMetricsDB)
                    .where(RequestMetricsDB.created_at < older_than)
                )
                
                # Delete old session metrics
                await conn.execute(
                    delete(SessionMetricsDB)
                    .where(SessionMetricsDB.created_at < older_than)
                )
                
                # Delete old benchmark metrics
                await conn.execute(
                    delete(BenchmarkMetricsDB)
                    .where(BenchmarkMetricsDB.created_at < older_than)
                )
        
        logger.info(f"Cleaned up data older than {older_than}")
    
    async def close(self):
        """Clean up resources."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
        if self._async_engine:
            self._async_engine.dispose()
            self._async_engine = None
    
    def _row_to_request_metrics(self, row: Any) -> RequestMetrics:
        """Convert a database row to RequestMetrics."""
        return RequestMetrics(
            request_id=row.request_id,
            session_id=row.session_id,
            prompt_hash=row.prompt_hash,
            created_at=row.created_at,
            completed_at=row.completed_at,
            request_type=row.request_type,
            model_info=ModelMetrics(
                model_id=row.model_id,
                model_provider=row.model_provider,
                model_version=row.model_version,
                model_parameters=row.model_parameters,
                context_window=row.context_window,
            ),
            token_usage=TokenUsage(
                prompt_tokens=row.prompt_tokens or 0,
                completion_tokens=row.completion_tokens or 0,
                total_tokens=row.total_tokens or 0,
            ),
            cost=CostMetrics(
                input_token_cost=row.input_token_cost or 0.0,
                output_token_cost=row.output_token_cost or 0.0,
                total_cost=row.total_cost or 0.0,
                currency=row.currency or "USD",
            ),
            latency=LatencyMetrics(
                time_to_first_token_ms=row.time_to_first_token_ms,
                time_to_complete_ms=row.time_to_complete_ms,
            ),
            quality=QualityMetrics(
                relevance_score=row.relevance_score,
                accuracy_score=row.accuracy_score,
                coherence_score=row.coherence_score,
                helpfulness_score=row.helpfulness_score,
                human_rating=row.human_rating,
                is_error=row.is_error == 1,
                error_type=row.error_type,
                error_message=row.error_message,
            ),
            git_commit=row.git_commit,
            environment=row.environment,
            user_id=row.user_id,
            metadata=row.metadata or {},
            tags=row.tags or [],
        )
    
    def _row_to_session_metrics(self, row: Any) -> SessionMetrics:
        """Convert a database row to SessionMetrics."""
        return SessionMetrics(
            session_id=row.session_id,
            created_at=row.created_at,
            completed_at=row.completed_at,
            total_requests=row.total_requests or 0,
            total_token_usage=TokenUsage(
                prompt_tokens=row.total_prompt_tokens or 0,
                completion_tokens=row.total_completion_tokens or 0,
                total_tokens=row.total_tokens or 0,
            ),
            total_cost=CostMetrics(
                input_token_cost=row.total_input_cost or 0.0,
                output_token_cost=row.total_output_cost or 0.0,
                total_cost=row.total_cost or 0.0,
            ),
            total_latency=LatencyMetrics(
                time_to_complete_ms=row.avg_time_to_complete_ms,
            ),
            chain_length=row.chain_length or 0,
            is_complete=row.is_complete == 1,
            final_output=row.final_output,
            git_commit=row.git_commit,
            metadata=row.metadata or {},
        )
    
    def _row_to_benchmark_metrics(self, row: Any) -> BenchmarkMetrics:
        """Convert a database row to BenchmarkMetrics."""
        return BenchmarkMetrics(
            benchmark_id=row.benchmark_id,
            benchmark_name=row.benchmark_name,
            model_info=ModelMetrics(
                model_id=row.model_id,
                model_provider=row.model_provider,
                model_version=row.model_version,
            ),
            avg_token_usage=TokenUsage(
                prompt_tokens=row.avg_prompt_tokens or 0,
                completion_tokens=row.avg_completion_tokens or 0,
                total_tokens=row.avg_total_tokens or 0,
            ),
            avg_cost=CostMetrics(
                input_token_cost=row.avg_input_cost or 0.0,
                output_token_cost=row.avg_output_cost or 0.0,
                total_cost=row.avg_total_cost or 0.0,
            ),
            avg_latency=LatencyMetrics(
                time_to_first_token_ms=row.avg_time_to_first_token_ms,
                time_to_complete_ms=row.avg_time_to_complete_ms,
            ),
            avg_quality=QualityMetrics(
                relevance_score=row.avg_relevance_score,
                accuracy_score=row.avg_accuracy_score,
                coherence_score=row.avg_coherence_score,
            ),
            std_cost=row.std_cost,
            std_latency=row.std_latency,
            std_quality=row.std_quality,
            baseline_model_id=row.baseline_model_id,
            cost_savings_pct=row.cost_savings_pct,
            quality_improvement_pct=row.quality_improvement_pct,
            latency_improvement_pct=row.latency_improvement_pct,
            num_runs=row.num_runs or 0,
            created_at=row.created_at,
            updated_at=row.updated_at,
            metadata=row.metadata or {},
        )
