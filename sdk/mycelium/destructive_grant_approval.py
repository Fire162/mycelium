"""Two-person approval for host-issued destructive grants."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

from mycelium.storage.atomic_state import AtomicStateBackend, InMemoryAtomicStateBackend


class DestructiveGrantApprovalError(PermissionError):
    """A destructive grant was refused before it could be stored."""


@dataclass(frozen=True)
class DestructiveGrantApprovalRequest:
    """The exact, canonical grant fields approved before issuance."""

    operation: str
    object_type: str
    object_id: str
    request_id: str | None
    run_id: str | None
    thread_id: str | None
    tenant: str | None
    account: str | None
    expires_in: float
    max_uses: int
    policy_version: str
    case_sensitive: bool
    bind_request_id: bool
    bind_run_id: bool
    bind_thread_id: bool

    def key(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


class DualControlDestructiveGrantAuthorizer:
    """Authenticate two distinct operators and consume one exact approval.

    ``authenticate`` is host-owned and must validate the claimed operator's
    credential. Supply a durable atomic backend when workers share issuance.
    """

    def __init__(
        self,
        authenticate: Callable[[str, str | None], bool],
        *,
        approval_backend: AtomicStateBackend | None = None,
        approval_namespace: str = "destructive_grant_approval",
        approval_ttl: float = 900.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not approval_namespace:
            raise ValueError("approval_namespace must not be empty")
        if not math.isfinite(approval_ttl) or approval_ttl <= 0:
            raise ValueError("approval_ttl must be finite and positive")
        self._authenticate = authenticate
        self._approvals = (
            approval_backend if approval_backend is not None else InMemoryAtomicStateBackend()
        )
        self._namespace = approval_namespace
        self._approval_ttl = approval_ttl
        self._clock = clock

    def approve(
        self,
        request: DestructiveGrantApprovalRequest,
        *,
        operator_id: str,
        credential: str | None,
    ) -> bool:
        """Persist the first authenticated approval for this exact grant."""
        try:
            if not operator_id or self._authenticate(operator_id, credential) is not True:
                return False
            key = request.key()
            current = self._approvals.get(self._namespace, key)
            if current is not None:
                approved_at = current.value.get("approved_at")
                if (
                    isinstance(approved_at, (int, float))
                    and self._clock() < float(approved_at) + self._approval_ttl
                ):
                    return False
                if not self._approvals.delete(
                    self._namespace, key, expected_version=current.version
                ):
                    return False
            return self._approvals.create(
                self._namespace,
                key,
                {"operator_id": operator_id, "approved_at": self._clock()},
            )
        except Exception:
            return False

    def authorize_issuance(
        self,
        request: DestructiveGrantApprovalRequest,
        *,
        operator_id: str | None,
        credential: str | None,
    ) -> str | None:
        """Consume approval and return the first operator for grant evidence."""
        try:
            if not operator_id:
                return None
            key = request.key()
            current = self._approvals.get(self._namespace, key)
            if current is None:
                return None
            first = current.value.get("operator_id")
            approved_at = current.value.get("approved_at")
            if (
                not isinstance(first, str)
                or not first
                or not isinstance(approved_at, (int, float))
                or self._clock() >= float(approved_at) + self._approval_ttl
            ):
                self._approvals.delete(
                    self._namespace, key, expected_version=current.version
                )
                return None
            if (
                first == operator_id
                or self._authenticate(operator_id, credential) is not True
            ):
                return None
            consumed = self._approvals.delete(
                self._namespace, key, expected_version=current.version
            )
            return first if consumed else None
        except Exception:
            return None


__all__ = [
    "DestructiveGrantApprovalError",
    "DestructiveGrantApprovalRequest",
    "DualControlDestructiveGrantAuthorizer",
]
