"""Session management for chained prompts and multi-turn conversations."""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Union
from datetime import datetime
from contextlib import asynccontextmanager

from fancy_llm_router.schemas.sessions import (
    SessionConfig,
    SessionState,
    SessionStatus,
    PromptChain,
    ChainStep,
    ChainStepStatus,
)
from fancy_llm_router.schemas.requests import (
    CompletionRequest,
    ChatRequest,
    EmbeddingRequest,
    ChatMessage,
)
from fancy_llm_router.schemas.routing import RoutingDecision
from fancy_llm_router.core.router import LLMRouter
from fancy_llm_router.metrics.collector import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class SessionResult:
    """Result of a session execution."""
    session_id: str
    status: SessionStatus
    output: Optional[str] = None
    outputs: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    metrics: Optional[Any] = None
    request_ids: List[str] = field(default_factory=list)
    execution_time_seconds: float = 0.0


class SessionError(Exception):
    """Error in session execution."""
    pass


class SessionTimeoutError(SessionError):
    """Session execution timed out."""
    pass


class Session:
    """
    A single session that can execute a chain of prompts.
    
    Features:
    - Execute chains of prompts with dependencies
    - Maintain conversation context
    - Track session-level metrics
    - Support for conditional execution
    - Timeout handling
    """
    
    def __init__(
        self,
        config: SessionConfig,
        router: Optional[LLMRouter] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        **kwargs
    ):
        self.config = config
        self.router = router
        self.metrics_collector = metrics_collector
        self.extra_config = kwargs
        
        # Session state
        self.state = SessionState(
            session_id=config.session_id or str(uuid.uuid4()),
            config=config,
            status=SessionStatus.CREATED,
            created_at=datetime.utcnow(),
        )
        
        # Execution tracking
        self._current_step: Optional[ChainStep] = None
        self._completed_steps: List[str] = []
        self._failed_steps: List[str] = []
        self._step_results: Dict[str, Any] = {}
        
        # Timeout
        self._timeout_task: Optional[asyncio.Task] = None
        self._timeout_event = asyncio.Event()
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
    
    @property
    def session_id(self) -> str:
        return self.state.session_id
    
    @property
    def is_complete(self) -> bool:
        return self.state.is_complete()
    
    @property
    def is_running(self) -> bool:
        return self.state.status == SessionStatus.IN_PROGRESS
    
    def _get_chain(self) -> Optional[PromptChain]:
        """Get the prompt chain for this session."""
        if self.config.chain:
            return self.config.chain
        if self.config.chain_id:
            # In a real implementation, this would look up the chain by ID
            return None
        return None
    
    def _get_entry_step(self) -> Optional[ChainStep]:
        """Get the entry step of the chain."""
        chain = self._get_chain()
        if chain:
            return chain.get_entry_step()
        return None
    
    def _get_next_step(self, current_step: Optional[ChainStep] = None) -> Optional[ChainStep]:
        """Get the next step to execute."""
        chain = self._get_chain()
        if not chain:
            return None
        
        # If no current step, start with entry step
        if current_step is None:
            return self._get_entry_step()
        
        # Find the next step that depends on the current step
        for step in chain.steps:
            if current_step.step_id in step.depends_on:
                # Check if all dependencies are completed
                all_deps_completed = all(
                    dep in self._completed_steps 
                    for dep in step.depends_on
                )
                if all_deps_completed and step.step_id not in self._completed_steps:
                    return step
        
        # If no dependencies, find the next uncompleted step
        for step in chain.steps:
            if step.step_id not in self._completed_steps and step.step_id not in self._failed_steps:
                return step
        
        return None
    
    def _evaluate_condition(self, condition: Optional[str], variables: Dict[str, Any]) -> bool:
        """Evaluate a step condition."""
        if not condition:
            return True
        
        try:
            # Simple expression evaluation
            # In a real implementation, this would use a safe expression evaluator
            return eval(condition, {}, variables)
        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False
    
    async def _execute_step(self, step: ChainStep) -> bool:
        """Execute a single step in the chain."""
        start_time = time.time()
        
        try:
            # Update step status
            step.status = ChainStepStatus.IN_PROGRESS
            step.started_at = datetime.utcnow()
            
            # Check condition
            if not self._evaluate_condition(step.condition, self.state.variables):
                step.status = ChainStepStatus.SKIPPED
                step.completed_at = datetime.utcnow()
                logger.info(f"Skipped step {step.step_id} due to condition")
                return True
            
            # Prepare the request
            request = self._prepare_request(step)
            
            # Route and execute the request
            if self.router:
                decision = await self.router.route(request)
                response = await self.router.execute(
                    request,
                    track_metrics=self.metrics_collector is not None
                )
            else:
                # Direct execution without routing
                if step.model:
                    request.model = step.model
                if step.provider:
                    request.provider = step.provider
                
                # This would use a direct provider in a real implementation
                raise SessionError("No router configured for session execution")
            
            # Process the response
            step.response = self._process_response(response)
            step.status = ChainStepStatus.COMPLETED
            step.completed_at = datetime.utcnow()
            
            # Store the result
            self._step_results[step.step_id] = {
                "response": step.response,
                "request_id": request.request_id,
                "execution_time": time.time() - start_time,
            }
            
            # Update session state
            self.state.completed_steps.append(step.step_id)
            self.state.request_ids.append(request.request_id)
            self.state.total_tokens += request.max_tokens  # Approximate
            self.state.total_time_seconds += time.time() - start_time
            
            # Update variables from step output
            if step.output_schema:
                # Extract variables based on output schema
                self._update_variables_from_output(step)
            
            logger.info(f"Completed step {step.step_id} in {time.time() - start_time:.2f}s")
            return True
            
        except Exception as e:
            step.status = ChainStepStatus.FAILED
            step.completed_at = datetime.utcnow()
            step.response = str(e)
            
            self.state.failed_steps.append(step.step_id)
            self.state.errors[step.step_id] = str(e)
            
            logger.error(f"Failed step {step.step_id}: {e}")
            
            # Check if we should continue on failure
            chain = self._get_chain()
            if chain and chain.continue_on_failure:
                return True
            return False
    
    def _prepare_request(self, step: ChainStep) -> Union[CompletionRequest, ChatRequest]:
        """Prepare a request from a step configuration."""
        # Format the prompt with variables
        prompt = step.prompt_template
        for var_name, var_value in self.state.variables.items():
            prompt = prompt.replace(f"{{{{{var_name}}}}}", str(var_value))
        
        # Create the request
        request_id = f"{self.session_id}-{step.step_id}-{int(time.time())}"
        
        request = ChatRequest(
            request_id=request_id,
            session_id=self.session_id,
            model=step.model or self.config.default_model or "",
            messages=[
                ChatMessage(
                    role="user",
                    content=prompt,
                )
            ],
            max_tokens=step.prompt_variables.get("max_tokens", 256),
            temperature=step.prompt_variables.get("temperature", 0.7),
            metadata={
                "step_id": step.step_id,
                "step_name": step.step_name,
                "session_id": self.session_id,
            },
        )
        
        return request
    
    def _process_response(self, response: Any) -> str:
        """Process a response from the LLM."""
        if hasattr(response, 'choices') and response.choices:
            if hasattr(response.choices[0], 'message'):
                return response.choices[0].message.content
            elif hasattr(response.choices[0], 'text'):
                return response.choices[0].text
        return str(response)
    
    def _update_variables_from_output(self, step: ChainStep):
        """Update session variables from step output."""
        # Simple implementation: store the response in a variable with the step name
        if step.step_name:
            self.state.variables[step.step_name] = step.response
        
        # If there's an output schema, parse the response accordingly
        if step.output_schema:
            # In a real implementation, this would parse the response
            # based on the output schema (e.g., JSON, structured data)
            pass
    
    async def _check_timeout(self, timeout: Optional[float] = None):
        """Check if the session has timed out."""
        timeout = timeout or self.config.max_total_time
        if timeout is None:
            return
        
        start_time = self.state.created_at.timestamp()
        current_time = time.time()
        
        if current_time - start_time > timeout:
            self._timeout_event.set()
    
    async def start(self):
        """Start the session."""
        async with self._lock:
            if self.state.status != SessionStatus.CREATED:
                raise SessionError(f"Cannot start session in {self.state.status} state")
            
            self.state.status = SessionStatus.IN_PROGRESS
            self.state.started_at = datetime.utcnow()
            
            # Start timeout monitoring if configured
            if self.config.max_total_time:
                self._timeout_task = asyncio.create_task(
                    self._monitor_timeout()
                )
    
    async def _monitor_timeout(self):
        """Monitor session timeout."""
        timeout = self.config.max_total_time
        if timeout is None:
            return
        
        await asyncio.sleep(timeout)
        self._timeout_event.set()
    
    async def execute(self) -> SessionResult:
        """
        Execute the session (run the prompt chain).
        
        Returns:
            SessionResult with the execution results
        """
        start_time = time.time()
        
        try:
            # Start the session
            await self.start()
            
            # Get the chain
            chain = self._get_chain()
            if not chain:
                raise SessionError("No prompt chain configured for session")
            
            # Initialize session variables with input data
            self.state.variables.update(self.config.input_data)
            
            # Execute steps
            current_step = self._get_entry_step()
            while current_step and not self._timeout_event.is_set():
                # Execute the step
                success = await self._execute_step(current_step)
                
                if not success and not chain.continue_on_failure:
                    self.state.status = SessionStatus.FAILED
                    break
                
                # Mark step as completed
                self._completed_steps.append(current_step.step_id)
                
                # Get next step
                current_step = self._get_next_step(current_step)
            
            # Determine final status
            if self._timeout_event.is_set():
                self.state.status = SessionStatus.TIMEOUT
            elif self.state.status != SessionStatus.FAILED:
                if len(self._completed_steps) == len(chain.steps):
                    self.state.status = SessionStatus.COMPLETED
                else:
                    self.state.status = SessionStatus.FAILED
            
            # Finalize
            self.state.completed_at = datetime.utcnow()
            
            # Get the final output
            output = None
            if chain.exit_step_id:
                exit_step = next(
                    (s for s in chain.steps if s.step_id == chain.exit_step_id),
                    None
                )
                if exit_step and exit_step.step_id in self._step_results:
                    output = self._step_results[exit_step.step_id]["response"]
            
            # Build result
            result = SessionResult(
                session_id=self.session_id,
                status=self.state.status,
                output=output,
                outputs=self._step_results,
                errors=self.state.errors,
                request_ids=self.state.request_ids,
                execution_time_seconds=time.time() - start_time,
            )
            
            return result
            
        except asyncio.TimeoutError:
            self.state.status = SessionStatus.TIMEOUT
            self.state.completed_at = datetime.utcnow()
            raise SessionTimeoutError("Session execution timed out")
        except Exception as e:
            self.state.status = SessionStatus.FAILED
            self.state.completed_at = datetime.utcnow()
            self.state.errors["session"] = str(e)
            raise SessionError(f"Session execution failed: {e}")
        finally:
            # Cancel timeout task
            if self._timeout_task:
                self._timeout_task.cancel()
                try:
                    await self._timeout_task
                except asyncio.CancelledError:
                    pass
    
    async def execute_step(self, step_id: str) -> bool:
        """Execute a specific step by ID."""
        chain = self._get_chain()
        if not chain:
            raise SessionError("No prompt chain configured for session")
        
        # Find the step
        step = next((s for s in chain.steps if s.step_id == step_id), None)
        if not step:
            raise SessionError(f"Step {step_id} not found in chain")
        
        # Start the session if not already started
        if self.state.status == SessionStatus.CREATED:
            await self.start()
        
        # Execute the step
        return await self._execute_step(step)
    
    async def cancel(self):
        """Cancel the session."""
        async with self._lock:
            if self.state.status in [SessionStatus.COMPLETED, SessionStatus.FAILED, SessionStatus.TIMEOUT]:
                return
            
            self.state.status = SessionStatus.FAILED
            self.state.completed_at = datetime.utcnow()
            self.state.errors["session"] = "Cancelled by user"
            
            # Cancel timeout task
            if self._timeout_task:
                self._timeout_task.cancel()
                try:
                    await self._timeout_task
                except asyncio.CancelledError:
                    pass
            
            # Set timeout event to stop execution
            self._timeout_event.set()
    
    def get_state(self) -> SessionState:
        """Get the current session state."""
        return self.state
    
    def get_step_results(self) -> Dict[str, Any]:
        """Get results for all executed steps."""
        return self._step_results
    
    def get_step_status(self, step_id: str) -> Optional[ChainStepStatus]:
        """Get the status of a specific step."""
        chain = self._get_chain()
        if chain:
            for step in chain.steps:
                if step.step_id == step_id:
                    return step.status
        return None


class SessionManager:
    """
    Manages multiple sessions.
    
    Features:
    - Create and manage multiple sessions
    - Track active sessions
    - Clean up completed sessions
    - Session recovery
    """
    
    def __init__(
        self,
        router: Optional[LLMRouter] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        max_concurrent_sessions: int = 10,
        session_timeout: Optional[float] = None,
        **kwargs
    ):
        self.router = router
        self.metrics_collector = metrics_collector
        self.max_concurrent_sessions = max_concurrent_sessions
        self.session_timeout = session_timeout
        self.extra_config = kwargs
        
        # Session storage
        self._sessions: Dict[str, Session] = {}
        self._active_sessions: Dict[str, Session] = {}
        self._completed_sessions: Dict[str, Session] = {}
        
        # Semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent_sessions)
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
    
    async def create_session(
        self,
        config: Optional[SessionConfig] = None,
        **kwargs
    ) -> Session:
        """Create a new session."""
        if config is None:
            config = SessionConfig()
        
        session = Session(
            config=config,
            router=self.router,
            metrics_collector=self.metrics_collector,
            **kwargs
        )
        
        async with self._lock:
            self._sessions[session.session_id] = session
        
        return session
    
    async def execute_session(
        self,
        config: Optional[SessionConfig] = None,
        wait: bool = True,
        **kwargs
    ) -> SessionResult:
        """
        Create and execute a session.
        
        Args:
            config: Session configuration
            wait: Whether to wait for completion
            **kwargs: Additional arguments
            
        Returns:
            SessionResult if wait=True, otherwise None
        """
        session = await self.create_session(config, **kwargs)
        
        async def execute_and_cleanup():
            try:
                result = await session.execute()
                async with self._lock:
                    self._active_sessions.pop(session.session_id, None)
                    self._completed_sessions[session.session_id] = session
                return result
            except Exception as e:
                async with self._lock:
                    self._active_sessions.pop(session.session_id, None)
                raise e
        
        async with self._lock:
            self._active_sessions[session.session_id] = session
        
        if wait:
            async with self._semaphore:
                return await execute_and_cleanup()
        else:
            # Start execution in background
            asyncio.create_task(execute_and_cleanup())
            return None
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        async with self._lock:
            return self._sessions.get(session_id)
    
    async def cancel_session(self, session_id: str) -> bool:
        """Cancel a session."""
        session = await self.get_session(session_id)
        if session:
            await session.cancel()
            async with self._lock:
                self._active_sessions.pop(session_id, None)
            return True
        return False
    
    async def list_sessions(
        self,
        status: Optional[SessionStatus] = None
    ) -> List[Session]:
        """List sessions by status."""
        async with self._lock:
            if status is None:
                return list(self._sessions.values())
            
            if status == SessionStatus.IN_PROGRESS:
                return list(self._active_sessions.values())
            elif status in [SessionStatus.COMPLETED, SessionStatus.FAILED, SessionStatus.TIMEOUT]:
                return [
                    s for s in self._completed_sessions.values()
                    if s.state.status == status
                ]
            else:
                return [
                    s for s in self._sessions.values()
                    if s.state.status == status
                ]
    
    async def cleanup_sessions(
        self,
        older_than_seconds: Optional[float] = None,
        max_sessions: Optional[int] = None
    ):
        """Clean up old or excess sessions."""
        async with self._lock:
            now = time.time()
            
            # Clean up by age
            if older_than_seconds:
                sessions_to_remove = [
                    sid for sid, session in self._sessions.items()
                    if (now - session.state.created_at.timestamp()) > older_than_seconds
                ]
                for sid in sessions_to_remove:
                    del self._sessions[sid]
                    self._active_sessions.pop(sid, None)
                    self._completed_sessions.pop(sid, None)
            
            # Clean up by count
            if max_sessions and len(self._sessions) > max_sessions:
                # Sort by creation time and remove oldest
                sorted_sessions = sorted(
                    self._sessions.items(),
                    key=lambda x: x[1].state.created_at.timestamp()
                )
                sessions_to_remove = [
                    sid for sid, _ in sorted_sessions[:len(self._sessions) - max_sessions]
                ]
                for sid in sessions_to_remove:
                    del self._sessions[sid]
                    self._active_sessions.pop(sid, None)
                    self._completed_sessions.pop(sid, None)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of session manager state."""
        return {
            "total_sessions": len(self._sessions),
            "active_sessions": len(self._active_sessions),
            "completed_sessions": len(self._completed_sessions),
            "max_concurrent": self.max_concurrent_sessions,
        }
    
    async def close(self):
        """Clean up all sessions."""
        async with self._lock:
            for session in self._sessions.values():
                await session.cancel()
            self._sessions.clear()
            self._active_sessions.clear()
            self._completed_sessions.clear()
