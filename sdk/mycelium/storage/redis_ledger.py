"""Redis-backed ledger storage with atomic SET NX claim semantics."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any, TypeVar

from mycelium.storage._helpers import ClaimOutcome
from mycelium.storage.transition_query import TransitionPage, decode_cursor, encode_cursor

E = TypeVar("E")


def _require_redis() -> Any:
    try:
        import redis
    except ImportError as exc:
        raise ImportError(
            "Redis storage requires the 'redis' package. "
            "Install with: pip install 'mycelium-runtime[redis]'"
        ) from exc
    return redis


class RedisEntryStorage:
    """Generic Redis KV store for ledger entries keyed by request_id.

    In-flight keys may carry a TTL. A durable **tombstone** (no TTL) is written
    alongside every ``set`` so a TTL eviction cannot look like "never claimed"
    — ``get`` / ``try_claim_inflight`` rehydrate an EXPIRED ghost from the
    tombstone and let ActionLedger death / hard-block / reconcile gates run.
    """

    def __init__(
        self,
        url: str,
        *,
        prefix: str,
        from_dict: Callable[[dict[str, Any]], E],
        in_flight_ttl: float | None = 604800.0,
        retention_seconds: float | None = None,
    ) -> None:
        redis = _require_redis()
        self._client = redis.Redis.from_url(url, decode_responses=True)
        self._prefix = prefix
        self._from_dict = from_dict
        self._in_flight_ttl = in_flight_ttl
        self.retention_seconds = retention_seconds
        self._indexes_ready = False

    def _key(self, request_id: str) -> str:
        return f"{self._prefix}{request_id}"

    def _tombstone_key(self, request_id: str) -> str:
        # Sibling namespace so list_all(prefix*) never treats tombs as entries.
        base = self._prefix.rstrip(":")
        return f"{base}-tomb:{request_id}"

    def _effect_key(self, effect_id: str) -> str:
        base = self._prefix.rstrip(":")
        return f"{base}-effect:{effect_id}"

    def _index_key(self, name: str) -> str:
        return f"{self._prefix.rstrip(':')}-index:{name}"

    @staticmethod
    def _payload_outcome(payload: dict[str, Any]) -> str:
        terminal = payload.get("terminal_outcome")
        if terminal:
            return str(terminal)
        status = payload.get("status")
        if status == "completed":
            return "COMPLETED"
        if status == "failed":
            return "FAILED_BEFORE_EFFECT"
        return "IN_FLIGHT"

    def _queue_index_update(
        self,
        pipe: Any,
        payload: dict[str, Any],
        *,
        previous: dict[str, Any] | None = None,
    ) -> None:
        # Redis clients provide sorted-set commands; the guard also preserves
        # compatibility with minimal transactional test doubles.
        if not hasattr(pipe, "zadd"):
            return
        request_id = str(payload["request_id"])
        if previous is not None:
            old_outcome = self._payload_outcome(previous)
            old_tool = str(previous.get("tool") or "")
            pipe.zrem(self._index_key(f"outcome:{old_outcome}"), request_id)
            if old_tool:
                pipe.zrem(self._index_key(f"tool:{old_tool}"), request_id)
            old_parent = str(previous.get("parent_request_id") or "")
            if old_parent:
                pipe.zrem(self._index_key(f"parent:{old_parent}"), request_id)
            pipe.zrem(self._index_key("finished"), request_id)
        started_at = float(payload.get("started_at") or 0.0)
        pipe.zadd(self._index_key("started"), {request_id: started_at})
        pipe.zadd(
            self._index_key(f"outcome:{self._payload_outcome(payload)}"),
            {request_id: started_at},
        )
        tool = str(payload.get("tool") or "")
        if tool:
            pipe.zadd(self._index_key(f"tool:{tool}"), {request_id: started_at})
        parent = str(payload.get("parent_request_id") or "")
        if parent:
            pipe.zadd(self._index_key(f"parent:{parent}"), {request_id: started_at})
        finished_at = payload.get("finished_at")
        if finished_at is not None:
            pipe.zadd(self._index_key("finished"), {request_id: float(finished_at)})

    def _ensure_indexes(self) -> None:
        if self._indexes_ready:
            return
        marker = self._index_key("ready")
        if not self._client.exists(marker):
            with self._client.pipeline(transaction=True) as pipe:
                for key in self._client.scan_iter(match=f"{self._prefix}*"):
                    raw = self._client.get(key)
                    if raw is None:
                        continue
                    payload = json.loads(raw)
                    if payload.get("request_id"):
                        self._queue_index_update(pipe, payload)
                pipe.set(marker, "1")
                pipe.execute()
        self._indexes_ready = True

    @staticmethod
    def _effect_id_for_entry(entry: Any) -> str:
        return str(getattr(entry, "effect_id", None) or entry.request_id)

    @staticmethod
    def _effect_id_from_payload(payload: dict[str, Any], request_id: str) -> str:
        return str(payload.get("effect_id") or request_id)

    @staticmethod
    def _fence_from_payload(raw: str | None) -> int:
        if raw is None:
            return 0
        return int(json.loads(raw).get("fence") or 0)

    def _read_tombstone(self, request_id: str) -> E | None:
        raw = self._client.get(self._tombstone_key(request_id))
        if raw is None:
            return None
        return self._from_dict(json.loads(raw))

    def _ghost_from_tombstone(self, entry: E, *, now: float) -> E:
        """Rebuild a durable ghost after TTL eviction.

        Non-terminal / in-flight tombs become EXPIRED (lease in the past) so
        reclaim and death-signal gates still apply. Terminal tombs restore as-is.
        """
        from dataclasses import replace

        from mycelium.transition import TerminalOutcome, legacy_status_from_terminal

        fields = getattr(entry, "__dataclass_fields__", {})
        terminal = getattr(entry, "terminal_outcome", None)
        if terminal is None and getattr(entry, "status", None) == "completed":
            terminal = TerminalOutcome.COMPLETED.value
        if terminal in (
            TerminalOutcome.COMPLETED.value,
            TerminalOutcome.FAILED_BEFORE_EFFECT.value,
            TerminalOutcome.FAILED_AFTER_EFFECT.value,
            TerminalOutcome.BLOCKED.value,
            TerminalOutcome.UNKNOWN.value,
        ):
            return entry

        updates: dict[str, Any] = {}
        if "lease_until" in fields:
            updates["lease_until"] = now - 1.0
        if "terminal_outcome" in fields:
            updates["terminal_outcome"] = TerminalOutcome.IN_FLIGHT.value
        if "status" in fields:
            updates["status"] = legacy_status_from_terminal(TerminalOutcome.IN_FLIGHT)
        return replace(entry, **updates) if updates else entry

    def _restore_from_tombstone(self, request_id: str, *, now: float | None = None) -> E | None:
        from redis.exceptions import WatchError

        key = self._key(request_id)
        tomb_key = self._tombstone_key(request_id)
        now = now if now is not None else time.time()
        for _ in range(32):
            try:
                with self._client.pipeline(transaction=True) as pipe:
                    pipe.watch(key, tomb_key)
                    primary_raw = pipe.get(key)
                    if primary_raw is not None:
                        return self._from_dict(json.loads(primary_raw))
                    tomb_raw = pipe.get(tomb_key)
                    if tomb_raw is None:
                        return None
                    tomb = self._from_dict(json.loads(tomb_raw))
                    ghost = self._ghost_from_tombstone(tomb, now=now)
                    effect_id = self._effect_id_for_entry(ghost)
                    effect_key = self._effect_key(effect_id)
                    pipe.watch(effect_key)
                    canonical = pipe.get(effect_key)
                    if canonical is not None and str(canonical) != request_id:
                        canonical_request_id = str(canonical)
                        canonical_raw = pipe.get(self._key(canonical_request_id))
                        if canonical_raw is not None:
                            return self._from_dict(json.loads(canonical_raw))
                        canonical_tomb = pipe.get(self._tombstone_key(canonical_request_id))
                        if canonical_tomb is not None:
                            return self._from_dict(json.loads(canonical_tomb))
                        return None
                    payload = json.dumps(ghost.to_dict(), default=str)
                    pipe.multi()
                    pipe.set(key, payload)
                    pipe.set(tomb_key, payload)
                    self._queue_index_update(pipe, ghost.to_dict())
                    if canonical is None:
                        pipe.set(effect_key, request_id)
                    pipe.execute()
                    return ghost
            except WatchError:
                continue
        raise RuntimeError("Redis tombstone restoration exhausted WATCH retries")

    def _scan_request_id_for_effect(self, effect_id: str) -> str | None:
        pattern = f"{self._prefix}*"
        candidates: list[tuple[float, str]] = []
        for key in self._client.scan_iter(match=pattern):
            raw = self._client.get(key)
            if raw is None:
                continue
            payload = json.loads(raw)
            request_id = str(payload.get("request_id") or "").strip()
            if not request_id:
                continue
            if self._effect_id_from_payload(payload, request_id) != effect_id:
                continue
            started = payload.get("started_at")
            try:
                started_at = float(started) if started is not None else 0.0
            except (TypeError, ValueError):
                started_at = 0.0
            candidates.append((started_at, request_id))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0][1]

    def resolve_request_id(self, effect_id: str) -> str | None:
        effect_key = self._effect_key(effect_id)
        mapped = self._client.get(effect_key)
        if mapped:
            canonical = str(mapped)
            if self.get(canonical) is not None:
                return canonical
        canonical = self._scan_request_id_for_effect(effect_id)
        if canonical is None:
            return None
        self._client.setnx(effect_key, canonical)
        return canonical

    def get_by_effect_id(self, effect_id: str) -> E | None:
        request_id = self.resolve_request_id(effect_id)
        if request_id is None:
            return None
        return self.get(request_id)

    def get(self, request_id: str) -> E | None:
        raw = self._client.get(self._key(request_id))
        if raw is not None:
            return self._from_dict(json.loads(raw))
        return self._restore_from_tombstone(request_id)

    def set(self, entry: E) -> None:
        from redis.exceptions import WatchError

        payload = json.dumps(entry.to_dict(), default=str)
        key = self._key(entry.request_id)
        tomb_key = self._tombstone_key(entry.request_id)
        effect_id = self._effect_id_for_entry(entry)
        effect_key = self._effect_key(effect_id)
        incoming_fence = int(getattr(entry, "fence", 0) or 0)
        for _ in range(32):
            try:
                with self._client.pipeline(transaction=True) as pipe:
                    pipe.watch(key, tomb_key, effect_key)
                    canonical = pipe.get(effect_key)
                    if (
                        canonical is not None
                        and str(canonical) != entry.request_id
                        and pipe.get(key) is None
                        and pipe.get(tomb_key) is None
                    ):
                        return
                    stored_fence = max(
                        self._fence_from_payload(pipe.get(key)),
                        self._fence_from_payload(pipe.get(tomb_key)),
                    )
                    if stored_fence > incoming_fence:
                        return
                    previous_raw = pipe.get(key) or pipe.get(tomb_key)
                    previous = json.loads(previous_raw) if previous_raw is not None else None
                    pipe.multi()
                    if entry.status == "in-flight" and self._in_flight_ttl:
                        pipe.set(key, payload, ex=int(self._in_flight_ttl))
                    else:
                        pipe.set(key, payload)
                    pipe.set(tomb_key, payload)
                    self._queue_index_update(pipe, entry.to_dict(), previous=previous)
                    if canonical is None:
                        pipe.set(effect_key, entry.request_id)
                    pipe.execute()
                    return
            except WatchError:
                continue
        raise RuntimeError("Redis ledger set exhausted WATCH retries")

    def try_claim_inflight(
        self,
        entry: E,
        *,
        lease_ttl: float = 3600.0,
    ) -> tuple[ClaimOutcome, E | None]:
        from mycelium.storage._helpers import claim_inflight_outcome, with_lease

        effect_id = self._effect_id_for_entry(entry)
        ttl = int(max(self._in_flight_ttl or lease_ttl or 0, lease_ttl * 4))

        for _ in range(32):
            canonical_request_id = self.resolve_request_id(effect_id)
            active_request_id = canonical_request_id or entry.request_id
            key = self._key(active_request_id)
            claim_entry = (
                entry
                if active_request_id == entry.request_id
                else replace(entry, request_id=active_request_id)
            )
            existing_raw = self._client.get(key)
            if existing_raw is None:
                restored = self._restore_from_tombstone(active_request_id)
                if restored is not None:
                    existing = restored
                    now = time.time()
                    outcome = claim_inflight_outcome(existing, now=now)
                    if outcome == "completed":
                        return "completed", existing
                    if outcome == "in_flight":
                        return "in_flight", existing
                    # EXPIRED / retryable → CAS reclaim path below
                    reclaimed = self._try_reclaim(
                        key,
                        claim_entry,
                        ttl,
                        lease_ttl,
                        effect_id=effect_id,
                    )
                    if reclaimed is not None:
                        return reclaimed
                    continue

                leased = with_lease(claim_entry, now=time.time(), lease_ttl=lease_ttl)
                if self._try_initial_claim(key, leased, ttl, effect_id=effect_id):
                    return "claimed", None
                continue

            existing = self._from_dict(json.loads(existing_raw))
            now = time.time()
            outcome = claim_inflight_outcome(existing, now=now)
            if outcome == "completed":
                return "completed", existing
            if outcome == "in_flight":
                return "in_flight", existing

            reclaimed = self._try_reclaim(
                key,
                claim_entry,
                ttl,
                lease_ttl,
                effect_id=effect_id,
            )
            if reclaimed is not None:
                return reclaimed

        raise RuntimeError("Redis claim exhausted WATCH retries")

    def _try_initial_claim(self, key: str, entry: E, ttl: int, *, effect_id: str) -> bool:
        from redis.exceptions import WatchError

        tomb_key = self._tombstone_key(entry.request_id)
        effect_key = self._effect_key(effect_id)
        payload = json.dumps(entry.to_dict(), default=str)
        try:
            with self._client.pipeline(transaction=True) as pipe:
                pipe.watch(key, tomb_key, effect_key)
                canonical = pipe.get(effect_key)
                if canonical is not None and str(canonical) != entry.request_id:
                    return False
                if pipe.get(key) is not None or pipe.get(tomb_key) is not None:
                    return False
                pipe.multi()
                if ttl > 0:
                    pipe.set(key, payload, ex=ttl)
                else:
                    pipe.set(key, payload)
                pipe.set(tomb_key, payload)
                self._queue_index_update(pipe, entry.to_dict())
                if canonical is None:
                    pipe.set(effect_key, entry.request_id)
                pipe.execute()
                return True
        except WatchError:
            return False

    def _try_reclaim(
        self,
        key: str,
        entry: E,
        ttl: int,
        lease_ttl: float,
        *,
        effect_id: str,
    ) -> tuple[ClaimOutcome, E | None] | None:
        """CAS reclaim: only overwrite if the stored entry is still
        reclaimable per :func:`claim_inflight_outcome`. Returns ``None``
        when the watch fires (caller retries from the top)."""
        from redis.exceptions import WatchError

        from mycelium.storage._helpers import claim_inflight_outcome, with_lease

        now = time.time()
        effect_key = self._effect_key(effect_id)
        try:
            with self._client.pipeline(transaction=True) as pipe:
                pipe.watch(key, effect_key)
                raw = pipe.get(key)
                if raw is None:
                    # Key evaporated under the watch — tombstone may still exist.
                    restored = self._restore_from_tombstone(entry.request_id, now=now)
                    if restored is not None:
                        return "in_flight", restored
                    return None
                canonical = pipe.get(effect_key)
                if canonical is not None and str(canonical) != entry.request_id:
                    current = self._from_dict(json.loads(raw))
                    return "in_flight", current
                current = self._from_dict(json.loads(raw))
                rerun = claim_inflight_outcome(current, now=time.time())
                if rerun != "claimed":
                    return rerun, current
                # Reclaim: bump the fence past the superseded claim.
                leased = with_lease(entry, now=now, lease_ttl=lease_ttl, prior=current)
                payload = json.dumps(leased.to_dict(), default=str)
                pipe.multi()
                if ttl > 0:
                    pipe.set(key, payload, ex=ttl)
                else:
                    pipe.set(key, payload)
                pipe.set(self._tombstone_key(entry.request_id), payload)
                self._queue_index_update(pipe, leased.to_dict(), previous=json.loads(raw))
                if canonical is None:
                    pipe.set(effect_key, entry.request_id)
                pipe.execute()
                return ("claimed", None)
        except WatchError:
            return None

    def try_transition(
        self,
        entry: E,
        *,
        expected_terminal_outcomes: frozenset[str],
        expected_owner: str | None = None,
        require_lease_held_at: float | None = None,
        expected_fence: int | None = None,
        expected_effect_state: str | None = None,
    ) -> bool:
        from redis.exceptions import WatchError

        from mycelium.storage._helpers import lease_allows_renew

        key = self._key(entry.request_id)
        effect_id = self._effect_id_for_entry(entry)
        effect_key = self._effect_key(effect_id)
        payload = json.dumps(entry.to_dict(), default=str)
        for _ in range(32):
            try:
                with self._client.pipeline(transaction=True) as pipe:
                    pipe.watch(key, effect_key)
                    raw = pipe.get(key)
                    if raw is None:
                        # TTL eviction mid-transition: restore history, then retry.
                        restored = self._restore_from_tombstone(entry.request_id)
                        if restored is None:
                            return False
                        continue
                    existing = json.loads(raw)
                    if existing.get("terminal_outcome") not in expected_terminal_outcomes:
                        return False
                    if expected_owner is not None and existing.get("owner") != expected_owner:
                        return False
                    if (
                        expected_fence is not None
                        and int(existing.get("fence") or 0) != expected_fence
                    ):
                        return False
                    if (
                        expected_effect_state is not None
                        and (existing.get("effect_phase") or "INTENDED") != expected_effect_state
                    ):
                        return False
                    if require_lease_held_at is not None and not lease_allows_renew(
                        existing.get("lease_until"),
                        now=require_lease_held_at,
                    ):
                        return False
                    canonical = pipe.get(effect_key)
                    pipe.multi()
                    pipe.set(key, payload)
                    pipe.set(self._tombstone_key(entry.request_id), payload)
                    self._queue_index_update(pipe, entry.to_dict(), previous=existing)
                    if canonical is None:
                        pipe.set(effect_key, entry.request_id)
                    pipe.execute()
                    return True
            except WatchError:
                continue
        raise RuntimeError("Redis transition exhausted WATCH retries")

    def list_all(self) -> list[E]:
        entries: list[E] = []
        cursor: str | None = None
        while True:
            page = self.list_page(limit=1000, cursor=cursor)
            entries.extend(page.entries)
            cursor = page.next_cursor
            if cursor is None:
                return entries

    def list_page(
        self,
        *,
        limit: int = 100,
        cursor: str | None = None,
        tool: str | None = None,
        outcome: str | None = None,
        parent_request_id: str | None = None,
        started_after: float | None = None,
        started_before: float | None = None,
        finished_before: float | None = None,
    ) -> TransitionPage[E]:
        if limit < 1 or limit > 10_000:
            raise ValueError("transition page limit must be between 1 and 10000")
        self._ensure_indexes()
        if finished_before is not None:
            index = self._index_key("finished")
        elif outcome:
            index = self._index_key(f"outcome:{outcome}")
        elif tool:
            index = self._index_key(f"tool:{tool}")
        elif parent_request_id:
            index = self._index_key(f"parent:{parent_request_id}")
        else:
            index = self._index_key("started")
        decoded = decode_cursor(cursor)
        lower_bound = None if finished_before is not None else started_after
        if decoded is not None:
            lower_bound = max(lower_bound or float("-inf"), decoded[0])
        minimum = lower_bound if lower_bound is not None else "-inf"
        maximum = (
            f"({finished_before}"
            if finished_before is not None
            else f"({started_before}" if started_before is not None else "+inf"
        )
        offset = 0
        matched: list[E] = []
        chunk_size = max(100, min(limit * 2, 1000))
        while len(matched) <= limit:
            request_ids = self._client.zrangebyscore(
                index, minimum, maximum, start=offset, num=chunk_size
            )
            if not request_ids:
                break
            offset += len(request_ids)
            raws = self._client.mget([self._key(str(item)) for item in request_ids])
            for request_id, raw in zip(request_ids, raws):
                if raw is None:
                    raw = self._client.get(self._tombstone_key(str(request_id)))
                if raw is None:
                    continue
                entry = self._from_dict(json.loads(raw))
                order_time = entry.finished_at if finished_before is not None else entry.started_at
                key = (float(order_time or 0.0), str(entry.request_id))
                if decoded is not None and key <= decoded:
                    continue
                if outcome is not None and self._payload_outcome(entry.to_dict()) != outcome:
                    continue
                if tool is not None and getattr(entry, "tool", None) != tool:
                    continue
                if (
                    parent_request_id is not None
                    and getattr(entry, "parent_request_id", None) != parent_request_id
                ):
                    continue
                matched.append(entry)
                if len(matched) > limit:
                    break
            if len(request_ids) < chunk_size:
                break
        entries = matched[:limit]
        next_cursor = None
        if len(matched) > limit and entries:
            last = entries[-1]
            cursor_time = last.finished_at if finished_before is not None else last.started_at
            next_cursor = encode_cursor(float(cursor_time or 0.0), last.request_id)
        return TransitionPage(entries, next_cursor)

    def delete_entries(self, request_ids: list[str]) -> int:
        deleted = 0
        for request_id in request_ids:
            raw = self._client.get(self._key(request_id))
            if raw is None:
                raw = self._client.get(self._tombstone_key(request_id))
            if raw is None:
                continue
            payload = json.loads(raw)
            effect_id = self._effect_id_from_payload(payload, request_id)
            with self._client.pipeline(transaction=True) as pipe:
                pipe.delete(
                    self._key(request_id),
                    self._tombstone_key(request_id),
                    self._effect_key(effect_id),
                )
                pipe.zrem(self._index_key("started"), request_id)
                pipe.zrem(self._index_key("finished"), request_id)
                pipe.zrem(
                    self._index_key(f"outcome:{self._payload_outcome(payload)}"), request_id
                )
                tool = str(payload.get("tool") or "")
                if tool:
                    pipe.zrem(self._index_key(f"tool:{tool}"), request_id)
                parent = str(payload.get("parent_request_id") or "")
                if parent:
                    pipe.zrem(self._index_key(f"parent:{parent}"), request_id)
                results = pipe.execute()
            if int(results[0]) > 0:
                deleted += 1
        return deleted


class RedisLedgerStorage:
    """Redis storage for :class:`~mycelium.action_ledger.LedgerEntry`."""

    def __init__(
        self,
        url: str,
        *,
        prefix: str = "mycelium:action:",
        in_flight_ttl: float | None = 604800.0,
        retention_seconds: float | None = None,
    ) -> None:
        from mycelium.ledger_model import LedgerEntry
        from mycelium.storage.atomic_state import RedisAtomicStateBackend

        self._inner = RedisEntryStorage(
            url,
            prefix=prefix,
            from_dict=LedgerEntry.from_dict,
            in_flight_ttl=in_flight_ttl,
            retention_seconds=retention_seconds,
        )
        self.retention_seconds = retention_seconds
        self._composite_atomic_backend = RedisAtomicStateBackend(
            url, prefix=f"{prefix.rstrip(':')}-composite:"
        )

    def get(self, request_id: str) -> Any:
        return self._inner.get(request_id)

    def set(self, entry: Any) -> None:
        self._inner.set(entry)

    def try_claim_inflight(
        self,
        entry: Any,
        *,
        lease_ttl: float = 3600.0,
    ) -> tuple[ClaimOutcome, Any | None]:
        return self._inner.try_claim_inflight(entry, lease_ttl=lease_ttl)

    def list_all(self) -> list[Any]:
        return self._inner.list_all()

    def list_page(self, **kwargs: Any) -> TransitionPage[Any]:
        return self._inner.list_page(**kwargs)

    def delete_entries(self, request_ids: list[str]) -> int:
        return self._inner.delete_entries(request_ids)

    def resolve_request_id(self, effect_id: str) -> str | None:
        return self._inner.resolve_request_id(effect_id)

    def get_by_effect_id(self, effect_id: str) -> Any | None:
        return self._inner.get_by_effect_id(effect_id)

    def try_transition(
        self,
        entry: Any,
        *,
        expected_terminal_outcomes: frozenset[str],
        expected_owner: str | None = None,
        require_lease_held_at: float | None = None,
        expected_fence: int | None = None,
        expected_effect_state: str | None = None,
    ) -> bool:
        return self._inner.try_transition(
            entry,
            expected_terminal_outcomes=expected_terminal_outcomes,
            expected_owner=expected_owner,
            require_lease_held_at=require_lease_held_at,
            expected_fence=expected_fence,
            expected_effect_state=expected_effect_state,
        )


class RedisTaskLedgerStorage:
    """Redis storage for :class:`~mycelium.task_ledger.TaskLedgerEntry`."""

    def __init__(
        self,
        url: str,
        *,
        prefix: str = "mycelium:task:",
        in_flight_ttl: float | None = 604800.0,
    ) -> None:
        from mycelium.task_ledger import TaskLedgerEntry

        self._inner = RedisEntryStorage(
            url,
            prefix=prefix,
            from_dict=TaskLedgerEntry.from_dict,
            in_flight_ttl=in_flight_ttl,
        )

    def get(self, request_id: str) -> Any:
        return self._inner.get(request_id)

    def set(self, entry: Any) -> None:
        self._inner.set(entry)

    def try_claim_inflight(
        self,
        entry: Any,
        *,
        lease_ttl: float = 3600.0,
    ) -> tuple[ClaimOutcome, Any | None]:
        return self._inner.try_claim_inflight(entry, lease_ttl=lease_ttl)

    def try_transition(
        self,
        entry: Any,
        *,
        expected_terminal_outcomes: frozenset[str],
        expected_owner: str | None = None,
        require_lease_held_at: float | None = None,
        expected_fence: int | None = None,
        expected_effect_state: str | None = None,
    ) -> bool:
        return self._inner.try_transition(
            entry,
            expected_terminal_outcomes=expected_terminal_outcomes,
            expected_owner=expected_owner,
            require_lease_held_at=require_lease_held_at,
            expected_fence=expected_fence,
            expected_effect_state=expected_effect_state,
        )

    def list_all(self) -> list[Any]:
        return self._inner.list_all()

    def resolve_request_id(self, effect_id: str) -> str | None:
        return self._inner.resolve_request_id(effect_id)

    def get_by_effect_id(self, effect_id: str) -> Any | None:
        return self._inner.get_by_effect_id(effect_id)
