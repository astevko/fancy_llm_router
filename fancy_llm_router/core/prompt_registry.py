"""SQLite-backed prompt roots, variants, and baseline results."""

import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from fancy_llm_router.schemas.prompts import (
    BaselineResult,
    BaselineRun,
    DeploymentPairState,
    JudgeResult,
    PromptDeploymentState,
    PromptRoot,
    PromptVariant,
    ResolvedPrompt,
)
from fancy_llm_router.utils.hash_utils import prompt_hash

logger = logging.getLogger(__name__)

Base = declarative_base()


class PromptRootDB(Base):
    __tablename__ = "prompt_roots"

    root_id = Column(String, primary_key=True)
    generic_text = Column(Text, nullable=False)
    generic_hash = Column(String, index=True)
    category = Column(String)
    expected_answer = Column(Text)
    source = Column(String, default="burner")
    created_at = Column(DateTime)
    metadata_json = Column(Text)


class PromptVariantDB(Base):
    __tablename__ = "prompt_variants"

    variant_id = Column(String, primary_key=True)
    root_id = Column(String, index=True, nullable=False)
    deployment_id = Column(String, index=True, nullable=False)
    revision = Column(Integer, default=1)
    parent_variant_id = Column(String)
    prompt_text = Column(Text, nullable=False)
    prompt_hash = Column(String, index=True)
    mutation_reason = Column(Text)
    judge_passed = Column(Boolean, default=False)
    created_at = Column(DateTime)
    metadata_json = Column(Text)


class BaselineRunDB(Base):
    __tablename__ = "baseline_runs"

    run_id = Column(String, primary_key=True)
    run_type = Column(String, default="smoke")
    prompt_scope = Column(String, default="single")
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    config_snapshot_json = Column(Text)


class BaselineResultDB(Base):
    __tablename__ = "baseline_results"

    result_id = Column(String, primary_key=True)
    run_id = Column(String, index=True, nullable=False)
    root_id = Column(String, index=True, nullable=False)
    deployment_id = Column(String, index=True, nullable=False)
    variant_id = Column(String)
    generic_prompt = Column(Text)
    prompt_used = Column(Text)
    response_text = Column(Text)
    response_hash = Column(String)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    total_cost = Column(Float, default=0.0)
    latency_ms = Column(Float, default=0.0)
    judge_pass = Column(Boolean, default=False)
    judge_accuracy = Column(Float, default=0.0)
    judge_rationale = Column(Text)
    is_canonical = Column(Boolean, default=False)
    created_at = Column(DateTime)
    metadata_json = Column(Text)


class PromptDeploymentStateDB(Base):
    __tablename__ = "prompt_deployment_states"

    root_id = Column(String, primary_key=True)
    deployment_id = Column(String, primary_key=True)
    state = Column(String, default="unset", nullable=False)
    notes = Column(Text)
    updated_at = Column(DateTime)


class PromptRegistry:
    """Persist and resolve generic prompts vs per-deployment variants."""

    def __init__(self, db_path: str = "data/metrics.db"):
        self.db_path = db_path
        self._engine = None
        self._Session = None

    def initialize(self) -> None:
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._engine = create_engine(f"sqlite:///{self.db_path}")
        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)
        logger.info("Initialized prompt registry at %s", self.db_path)

    def _session(self):
        if self._Session is None:
            self.initialize()
        return self._Session()

    def ensure_root(
        self,
        root_id: str,
        generic_text: str,
        expected_answer: Optional[str] = None,
        category: Optional[str] = None,
        source: str = "burner",
    ) -> PromptRoot:
        import json

        session = self._session()
        try:
            row = session.get(PromptRootDB, root_id)
            gh = prompt_hash(generic_text)
            if row is None:
                row = PromptRootDB(
                    root_id=root_id,
                    generic_text=generic_text,
                    generic_hash=gh,
                    category=category,
                    expected_answer=expected_answer,
                    source=source,
                    created_at=datetime.utcnow(),
                    metadata_json="{}",
                )
                session.add(row)
            else:
                row.generic_text = generic_text
                row.generic_hash = gh
                if expected_answer is not None:
                    row.expected_answer = expected_answer
                if category is not None:
                    row.category = category
            session.commit()
            return PromptRoot(
                root_id=row.root_id,
                generic_text=row.generic_text,
                generic_hash=row.generic_hash,
                category=row.category,
                expected_answer=row.expected_answer,
                source=row.source or source,
                created_at=row.created_at or datetime.utcnow(),
            )
        finally:
            session.close()

    def get_root(self, root_id: str) -> Optional[PromptRoot]:
        session = self._session()
        try:
            row = session.get(PromptRootDB, root_id)
            if row is None:
                return None
            return PromptRoot(
                root_id=row.root_id,
                generic_text=row.generic_text,
                generic_hash=row.generic_hash,
                category=row.category,
                expected_answer=row.expected_answer,
                source=row.source or "burner",
                created_at=row.created_at or datetime.utcnow(),
            )
        finally:
            session.close()

    def save_variant(
        self,
        root_id: str,
        deployment_id: str,
        prompt_text: str,
        parent_variant_id: Optional[str] = None,
        mutation_reason: Optional[str] = None,
        judge_passed: bool = False,
    ) -> PromptVariant:
        session = self._session()
        try:
            latest = (
                session.query(PromptVariantDB)
                .filter_by(root_id=root_id, deployment_id=deployment_id)
                .order_by(PromptVariantDB.revision.desc())
                .first()
            )
            revision = (latest.revision + 1) if latest else 1
            variant_id = str(uuid.uuid4())
            row = PromptVariantDB(
                variant_id=variant_id,
                root_id=root_id,
                deployment_id=deployment_id,
                revision=revision,
                parent_variant_id=parent_variant_id or (latest.variant_id if latest else None),
                prompt_text=prompt_text,
                prompt_hash=prompt_hash(prompt_text),
                mutation_reason=mutation_reason,
                judge_passed=judge_passed,
                created_at=datetime.utcnow(),
                metadata_json="{}",
            )
            session.add(row)
            session.commit()
            return self._variant_from_row(row)
        finally:
            session.close()

    def mark_variant_passed(self, variant_id: str) -> None:
        session = self._session()
        try:
            row = session.get(PromptVariantDB, variant_id)
            if row:
                row.judge_passed = True
                session.commit()
        finally:
            session.close()

    def resolve_prompt(
        self,
        root_id: str,
        deployment_id: str,
        generic_fallback: str,
    ) -> ResolvedPrompt:
        """Pick the best specialized variant for production, else generic."""
        session = self._session()
        try:
            root = session.get(PromptRootDB, root_id)
            generic = root.generic_text if root else generic_fallback
            pair_state = self._get_pair_state_row(session, root_id, deployment_id)
            state = pair_state.state if pair_state else DeploymentPairState.UNSET.value

            if state == DeploymentPairState.BLOCKED.value:
                return ResolvedPrompt(
                    root_id=root_id,
                    deployment_id=deployment_id,
                    generic_text=generic,
                    prompt_text=generic,
                    variant_id=None,
                )

            use_tuned = state in {
                DeploymentPairState.PREFERRED.value,
                DeploymentPairState.UNSET.value,
            }
            if state == DeploymentPairState.NEEDS_IMPROVEMENT.value:
                use_tuned = False

            variant = None
            if use_tuned:
                variant = (
                    session.query(PromptVariantDB)
                    .filter_by(
                        root_id=root_id,
                        deployment_id=deployment_id,
                        judge_passed=True,
                    )
                    .order_by(PromptVariantDB.revision.desc())
                    .first()
                )
            if variant:
                return ResolvedPrompt(
                    root_id=root_id,
                    deployment_id=deployment_id,
                    generic_text=generic,
                    prompt_text=variant.prompt_text,
                    variant_id=variant.variant_id,
                )
            return ResolvedPrompt(
                root_id=root_id,
                deployment_id=deployment_id,
                generic_text=generic,
                prompt_text=generic,
                variant_id=None,
            )
        finally:
            session.close()

    @staticmethod
    def _get_pair_state_row(session, root_id: str, deployment_id: str):
        return (
            session.query(PromptDeploymentStateDB)
            .filter_by(root_id=root_id, deployment_id=deployment_id)
            .first()
        )

    def get_pair_state(
        self,
        root_id: str,
        deployment_id: str,
    ) -> PromptDeploymentState:
        session = self._session()
        try:
            row = self._get_pair_state_row(session, root_id, deployment_id)
            if row is None:
                return PromptDeploymentState(
                    root_id=root_id,
                    deployment_id=deployment_id,
                    state=DeploymentPairState.UNSET,
                )
            return PromptDeploymentState(
                root_id=row.root_id,
                deployment_id=row.deployment_id,
                state=DeploymentPairState(row.state),
                updated_at=row.updated_at or datetime.utcnow(),
                notes=row.notes,
            )
        finally:
            session.close()

    def list_pair_states(self, root_id: str) -> List[PromptDeploymentState]:
        session = self._session()
        try:
            rows = (
                session.query(PromptDeploymentStateDB)
                .filter_by(root_id=root_id)
                .order_by(PromptDeploymentStateDB.deployment_id.asc())
                .all()
            )
            return [
                PromptDeploymentState(
                    root_id=row.root_id,
                    deployment_id=row.deployment_id,
                    state=DeploymentPairState(row.state),
                    updated_at=row.updated_at or datetime.utcnow(),
                    notes=row.notes,
                )
                for row in rows
            ]
        finally:
            session.close()

    def is_deployment_blocked(self, root_id: str, deployment_id: str) -> bool:
        return (
            self.get_pair_state(root_id, deployment_id).state
            == DeploymentPairState.BLOCKED
        )

    def set_pair_state(
        self,
        root_id: str,
        deployment_id: str,
        state: DeploymentPairState,
        notes: Optional[str] = None,
    ) -> PromptDeploymentState:
        session = self._session()
        try:
            if state == DeploymentPairState.PREFERRED:
                session.query(PromptDeploymentStateDB).filter_by(
                    root_id=root_id,
                    state=DeploymentPairState.PREFERRED.value,
                ).update(
                    {
                        "state": DeploymentPairState.UNSET.value,
                        "updated_at": datetime.utcnow(),
                    }
                )

            row = self._get_pair_state_row(session, root_id, deployment_id)
            if state == DeploymentPairState.UNSET:
                if row is not None:
                    session.delete(row)
                session.commit()
                return PromptDeploymentState(
                    root_id=root_id,
                    deployment_id=deployment_id,
                    state=DeploymentPairState.UNSET,
                )

            if row is None:
                row = PromptDeploymentStateDB(
                    root_id=root_id,
                    deployment_id=deployment_id,
                )
                session.add(row)
            row.state = state.value
            row.notes = notes
            row.updated_at = datetime.utcnow()
            session.commit()
            return PromptDeploymentState(
                root_id=row.root_id,
                deployment_id=row.deployment_id,
                state=DeploymentPairState(row.state),
                updated_at=row.updated_at,
                notes=row.notes,
            )
        finally:
            session.close()

    def get_latest_result(
        self,
        root_id: str,
        deployment_id: str,
    ) -> Optional[BaselineResult]:
        session = self._session()
        try:
            row = (
                session.query(BaselineResultDB)
                .filter_by(root_id=root_id, deployment_id=deployment_id)
                .order_by(BaselineResultDB.created_at.desc())
                .first()
            )
            if row is None:
                return None
            return self._result_from_row(row)
        finally:
            session.close()

    def create_run(
        self,
        run_type: str = "smoke",
        prompt_scope: str = "single",
        config_snapshot: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> BaselineRun:
        import json

        run_id = run_id or str(uuid.uuid4())
        session = self._session()
        try:
            row = BaselineRunDB(
                run_id=run_id,
                run_type=run_type,
                prompt_scope=prompt_scope,
                started_at=datetime.utcnow(),
                config_snapshot_json=json.dumps(config_snapshot or {}),
            )
            session.add(row)
            session.commit()
            return BaselineRun(
                run_id=run_id,
                run_type=run_type,
                prompt_scope=prompt_scope,
                started_at=row.started_at,
                config_snapshot=config_snapshot or {},
            )
        finally:
            session.close()

    def ensure_run(
        self,
        run_id: str,
        run_type: str = "client",
        prompt_scope: str = "single",
        config_snapshot: Optional[Dict[str, Any]] = None,
    ) -> BaselineRun:
        """Create a baseline run row if the client supplied an id that is not registered yet."""
        session = self._session()
        try:
            if session.get(BaselineRunDB, run_id) is not None:
                return self._run_from_db(session.get(BaselineRunDB, run_id))
        finally:
            session.close()
        return self.create_run(
            run_type=run_type,
            prompt_scope=prompt_scope,
            config_snapshot=config_snapshot,
            run_id=run_id,
        )

    def backfill_runs_from_results(self) -> int:
        """Persist baseline_runs rows for result groups that were stored without a run record."""
        import json

        from sqlalchemy import func

        session = self._session()
        try:
            groups = (
                session.query(
                    BaselineResultDB.run_id,
                    func.min(BaselineResultDB.created_at),
                    func.max(BaselineResultDB.created_at),
                    func.count(BaselineResultDB.result_id),
                )
                .group_by(BaselineResultDB.run_id)
                .all()
            )
            created = 0
            for run_id, started_at, completed_at, result_count in groups:
                if session.get(BaselineRunDB, run_id) is not None:
                    continue
                session.add(
                    BaselineRunDB(
                        run_id=run_id,
                        run_type="client",
                        prompt_scope="deployments",
                        started_at=started_at,
                        completed_at=completed_at,
                        config_snapshot_json=json.dumps(
                            {"result_count": int(result_count), "backfilled": True}
                        ),
                    )
                )
                created += 1
            if created:
                session.commit()
            return created
        finally:
            session.close()

    def complete_run(self, run_id: str) -> None:
        session = self._session()
        try:
            row = session.get(BaselineRunDB, run_id)
            if row:
                row.completed_at = datetime.utcnow()
                session.commit()
        finally:
            session.close()

    def store_result(self, result: BaselineResult) -> BaselineResult:
        import json

        session = self._session()
        try:
            row = BaselineResultDB(
                result_id=result.result_id,
                run_id=result.run_id,
                root_id=result.root_id,
                deployment_id=result.deployment_id,
                variant_id=result.variant_id,
                generic_prompt=result.generic_prompt,
                prompt_used=result.prompt_used,
                response_text=result.response_text,
                response_hash=result.response_hash,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.total_tokens,
                total_cost=result.total_cost,
                latency_ms=result.latency_ms,
                judge_pass=result.judge.pass_,
                judge_accuracy=result.judge.accuracy_score,
                judge_rationale=result.judge.rationale,
                is_canonical=result.is_canonical,
                created_at=result.created_at,
                metadata_json=json.dumps(result.metadata),
            )
            session.add(row)
            session.commit()
            return result
        finally:
            session.close()

    def list_results(self, run_id: str) -> List[BaselineResult]:
        session = self._session()
        try:
            rows = (
                session.query(BaselineResultDB)
                .filter_by(run_id=run_id)
                .order_by(BaselineResultDB.created_at.asc())
                .all()
            )
            return [self._result_from_row(r) for r in rows]
        finally:
            session.close()

    def list_results_for_root(self, root_id: str) -> List[BaselineResult]:
        session = self._session()
        try:
            rows = (
                session.query(BaselineResultDB)
                .filter_by(root_id=root_id)
                .order_by(BaselineResultDB.created_at.desc())
                .all()
            )
            return [self._result_from_row(r) for r in rows]
        finally:
            session.close()

    def list_results_for_root_deployment(
        self,
        root_id: str,
        deployment_id: str,
    ) -> List[BaselineResult]:
        session = self._session()
        try:
            rows = (
                session.query(BaselineResultDB)
                .filter_by(root_id=root_id, deployment_id=deployment_id)
                .order_by(BaselineResultDB.created_at.desc())
                .all()
            )
            return [self._result_from_row(r) for r in rows]
        finally:
            session.close()

    def list_deployments_for_root(self, root_id: str) -> List[Dict[str, Any]]:
        from sqlalchemy import func

        session = self._session()
        try:
            rows = (
                session.query(
                    BaselineResultDB.deployment_id,
                    func.count(BaselineResultDB.result_id),
                    func.count(func.distinct(BaselineResultDB.run_id)),
                    func.max(BaselineResultDB.created_at),
                )
                .filter_by(root_id=root_id)
                .group_by(BaselineResultDB.deployment_id)
                .order_by(func.max(BaselineResultDB.created_at).desc())
                .all()
            )
            deployments: List[Dict[str, Any]] = []
            for deployment_id, _result_count, run_count, last_at in rows:
                pass_rows = (
                    session.query(BaselineResultDB.result_id)
                    .filter_by(
                        root_id=root_id,
                        deployment_id=deployment_id,
                        judge_pass=True,
                    )
                    .count()
                )
                run_count = int(run_count or 0)
                state_row = self._get_pair_state_row(session, root_id, deployment_id)
                deployments.append(
                    {
                        "deployment_id": deployment_id,
                        "run_count": run_count,
                        "pass_count": int(pass_rows),
                        "fail_count": run_count - int(pass_rows),
                        "last_measured_at": last_at,
                        "state": state_row.state if state_row else "unset",
                    }
                )
            return deployments
        finally:
            session.close()

    def list_measured_roots(self, limit: int = 100) -> List[Dict[str, Any]]:
        from sqlalchemy import func

        session = self._session()
        try:
            rows = (
                session.query(
                    BaselineResultDB.root_id,
                    func.count(BaselineResultDB.result_id),
                    func.count(func.distinct(BaselineResultDB.run_id)),
                    func.max(BaselineResultDB.created_at),
                )
                .group_by(BaselineResultDB.root_id)
                .order_by(func.max(BaselineResultDB.created_at).desc())
                .limit(limit)
                .all()
            )
            roots: List[Dict[str, Any]] = []
            for root_id, result_count, run_count, last_at in rows:
                meta = self.get_root(root_id)
                sample = (
                    session.query(BaselineResultDB.generic_prompt)
                    .filter_by(root_id=root_id)
                    .order_by(BaselineResultDB.created_at.desc())
                    .first()
                )
                generic_prompt = ""
                if meta:
                    generic_prompt = meta.generic_text
                elif sample and sample[0]:
                    generic_prompt = sample[0]
                roots.append(
                    {
                        "root_id": root_id,
                        "generic_prompt": generic_prompt,
                        "category": meta.category if meta else None,
                        "expected_answer": meta.expected_answer if meta else None,
                        "run_count": int(run_count or 0),
                        "result_count": int(result_count or 0),
                        "last_measured_at": last_at,
                    }
                )
            return roots
        finally:
            session.close()

    def list_runs(self, limit: int = 50) -> List[BaselineRun]:
        self.backfill_runs_from_results()
        import json

        from sqlalchemy import func

        session = self._session()
        try:
            rows = (
                session.query(BaselineRunDB)
                .order_by(BaselineRunDB.started_at.desc())
                .limit(limit)
                .all()
            )
            runs: List[BaselineRun] = []
            known_ids: set[str] = set()
            for row in rows:
                known_ids.add(row.run_id)
                runs.append(self._run_from_db(row))

            # Safety net if results exist without a run row (should be rare after backfill).
            orphan_rows = (
                session.query(
                    BaselineResultDB.run_id,
                    func.min(BaselineResultDB.created_at),
                    func.max(BaselineResultDB.created_at),
                    func.count(BaselineResultDB.result_id),
                )
                .group_by(BaselineResultDB.run_id)
                .order_by(func.max(BaselineResultDB.created_at).desc())
                .limit(limit)
                .all()
            )
            for run_id, started_at, completed_at, _count in orphan_rows:
                if run_id in known_ids:
                    continue
                runs.append(
                    BaselineRun(
                        run_id=run_id,
                        run_type="client",
                        prompt_scope="deployments",
                        started_at=started_at,
                        completed_at=completed_at,
                        config_snapshot={},
                    )
                )

            runs.sort(
                key=lambda run: run.started_at or datetime.min,
                reverse=True,
            )
            return runs[:limit]
        finally:
            session.close()

    @staticmethod
    def _run_from_db(row: BaselineRunDB) -> BaselineRun:
        import json

        snapshot: Dict[str, Any] = {}
        if row.config_snapshot_json:
            try:
                snapshot = json.loads(row.config_snapshot_json)
            except json.JSONDecodeError:
                snapshot = {}
        return BaselineRun(
            run_id=row.run_id,
            run_type=row.run_type or "smoke",
            prompt_scope=row.prompt_scope or "single",
            started_at=row.started_at,
            completed_at=row.completed_at,
            config_snapshot=snapshot,
        )

    def get_run(self, run_id: str) -> Optional[BaselineRun]:
        from sqlalchemy import func

        session = self._session()
        try:
            row = session.get(BaselineRunDB, run_id)
            if row is not None:
                return self._run_from_db(row)

            agg = (
                session.query(
                    func.min(BaselineResultDB.created_at),
                    func.max(BaselineResultDB.created_at),
                )
                .filter_by(run_id=run_id)
                .one_or_none()
            )
            if agg is None or agg[0] is None:
                return None
            return BaselineRun(
                run_id=run_id,
                run_type="client",
                prompt_scope="deployments",
                started_at=agg[0],
                completed_at=agg[1],
                config_snapshot={},
            )
        finally:
            session.close()

    def get_variant(self, variant_id: str) -> Optional[PromptVariant]:
        session = self._session()
        try:
            row = session.get(PromptVariantDB, variant_id)
            return self._variant_from_row(row) if row else None
        finally:
            session.close()

    @staticmethod
    def _variant_from_row(row: PromptVariantDB) -> PromptVariant:
        return PromptVariant(
            variant_id=row.variant_id,
            root_id=row.root_id,
            deployment_id=row.deployment_id,
            revision=row.revision,
            parent_variant_id=row.parent_variant_id,
            prompt_text=row.prompt_text,
            prompt_hash=row.prompt_hash,
            mutation_reason=row.mutation_reason,
            judge_passed=bool(row.judge_passed),
            created_at=row.created_at or datetime.utcnow(),
        )

    @staticmethod
    def _result_from_row(row: BaselineResultDB) -> BaselineResult:
        import json

        metadata: dict = {}
        if row.metadata_json:
            try:
                metadata = json.loads(row.metadata_json)
            except json.JSONDecodeError:
                metadata = {}
        warnings = metadata.get("judge_warnings", [])
        return BaselineResult(
            result_id=row.result_id,
            run_id=row.run_id,
            root_id=row.root_id,
            deployment_id=row.deployment_id,
            variant_id=row.variant_id,
            generic_prompt=row.generic_prompt or "",
            prompt_used=row.prompt_used or "",
            response_text=row.response_text or "",
            response_hash=row.response_hash or "",
            prompt_tokens=row.prompt_tokens or 0,
            completion_tokens=row.completion_tokens or 0,
            total_tokens=row.total_tokens or 0,
            total_cost=row.total_cost or 0.0,
            latency_ms=row.latency_ms or 0.0,
            judge=JudgeResult(
                pass_=bool(row.judge_pass),
                accuracy_score=row.judge_accuracy or 0.0,
                rationale=row.judge_rationale or "",
                warnings=warnings,
            ),
            is_canonical=bool(row.is_canonical),
            created_at=row.created_at or datetime.utcnow(),
            metadata=metadata,
        )
