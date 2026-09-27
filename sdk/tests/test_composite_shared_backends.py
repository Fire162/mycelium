"""Composite parent control across independently constructed shared backends."""

from __future__ import annotations

import inspect
import uuid

import pytest

from mycelium import (
    CompositeAuthorityError,
    CompositeBusyError,
    PostgresLedgerStorage,
    RedisLedgerStorage,
    SideEffectClass,
    ToolTransitionBinding,
    composite,
    ledger_sync,
    register_composite_helper,
    side_effect,
)
from mycelium.composite import _ControlStore


def _binding() -> ToolTransitionBinding:
    return ToolTransitionBinding.for_tool(
        agent_id="shared-composite-test",
        policy_version="1",
        side_effect_class=SideEffectClass.KEYED_MUTATE,
        provider_idempotency_key_param="idempotency_key",
    )


def _exercise_shared_resume(first: object, second: object) -> None:
    calls: list[str] = []
    fail = {"enabled": True}
    operation_id = uuid.uuid4().hex

    @ledger_sync(storage=first, transition_binding=_binding())
    def effect(idempotency_key: str) -> str:
        with side_effect():
            calls.append(idempotency_key)
        return "done"

    def crash_after_child() -> None:
        if fail["enabled"]:
            raise RuntimeError("crash after child")

    register_composite_helper(crash_after_child)

    @composite(first)
    def initial(operation_id: str) -> str:
        result = effect(idempotency_key=operation_id)
        crash_after_child()
        return result

    resumed = composite(second)(inspect.unwrap(initial))

    with pytest.raises(RuntimeError, match="crash after child"):
        initial(operation_id=operation_id)
    fail["enabled"] = False
    assert resumed(operation_id=operation_id) == "done"
    assert calls == [operation_id]

    record = second._composite_atomic_backend.get(
        "composite-control-v1", f"mycelium:{operation_id}"
    )
    assert record is not None
    assert record.value["status"] == "COMPLETED"
    assert record.value["children"]

    # A live parent lease is visible to a different worker through the CAS store.
    control_a = _ControlStore(first)
    control_b = _ControlStore(second)
    manifest = initial._mycelium_composite_manifest
    key = f"mycelium:contended-{operation_id}"
    control_a.create_or_load(key, manifest, "mycelium")
    acquired = control_a.acquire(key, "worker-a", 30)
    with pytest.raises(CompositeBusyError):
        control_b.acquire(key, "worker-b", 30)
    control_a.release(key, "worker-a", acquired["fence"])
    assert control_b.acquire(key, "worker-b", 30)["fence"] > acquired["fence"]
    with pytest.raises(CompositeAuthorityError, match="lost parent authority"):
        control_a.renew(key, "worker-a", acquired["fence"], 30)


def test_redis_composite_shared_parent_control(monkeypatch: pytest.MonkeyPatch) -> None:
    fakeredis = pytest.importorskip("fakeredis")
    import redis

    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(redis.Redis, "from_url", lambda _url, **_kwargs: fake)
    prefix = f"test:composite:{uuid.uuid4().hex}:"
    first = RedisLedgerStorage("redis://unused", prefix=prefix)
    second = RedisLedgerStorage("redis://unused", prefix=prefix)
    _exercise_shared_resume(first, second)


def test_postgres_composite_shared_parent_control() -> None:
    from backend_gates import require_postgres_dsn_or_skip

    dsn = require_postgres_dsn_or_skip()
    first = PostgresLedgerStorage(dsn)
    second = PostgresLedgerStorage(dsn)
    try:
        _exercise_shared_resume(first, second)
    finally:
        first.close()
        second.close()
