"""Two-person approval at the destructive grant issuance boundary."""

from __future__ import annotations

import hmac
from concurrent.futures import ThreadPoolExecutor

import pytest

from mycelium import (
    DestructiveGrantApprovalError,
    DualControlDestructiveGrantAuthorizer,
    FileAtomicStateBackend,
    InMemoryAtomicStateBackend,
    destructive_grants,
    issue_destructive_grant,
    prepare_destructive_grant_approval_request,
    set_destructive_grant_approval_authorizer,
)
from mycelium.destructive_confirm import (
    InMemoryDestructiveGrantStore,
    reset_destructive_confirm_state,
)

FIELDS = {
    "operation": "refund",
    "object_type": "payment",
    "object_id": "pay_123",
    "request_id": "refund-123",
    "run_id": "run-1",
    "tenant": "acme",
    "account": "acct-1",
    "expires_in": 120.0,
    "max_uses": 1,
    "policy_version": "2026.09",
    "bind_request_id": True,
    "bind_run_id": True,
}
TOKENS = {"alice": "first-secret", "bob": "second-secret", "carol": "third-secret"}


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_destructive_confirm_state()
    yield
    reset_destructive_confirm_state()


def _authenticate(operator_id: str, credential: str | None) -> bool:
    expected = TOKENS.get(operator_id)
    return expected is not None and credential is not None and hmac.compare_digest(
        expected, credential
    )


def _authorizer(**kwargs: object) -> DualControlDestructiveGrantAuthorizer:
    return DualControlDestructiveGrantAuthorizer(_authenticate, **kwargs)


def _request(**overrides: object):
    return prepare_destructive_grant_approval_request(**{**FIELDS, **overrides})


def _issue(store: InMemoryDestructiveGrantStore, **overrides: object):
    return issue_destructive_grant(
        **{**FIELDS, **overrides},
        store=store,
        approval_operator_id="bob",
        approval_credential=TOKENS["bob"],
    )


def test_two_distinct_approvals_are_required_before_any_grant_is_stored() -> None:
    store = InMemoryDestructiveGrantStore()
    authorizer = _authorizer()
    set_destructive_grant_approval_authorizer(authorizer)

    with pytest.raises(DestructiveGrantApprovalError):
        _issue(store)
    assert not store._records

    request = _request()
    assert not authorizer.approve(request, operator_id="alice", credential="wrong")
    assert authorizer.approve(request, operator_id="alice", credential=TOKENS["alice"])
    assert not authorizer.approve(request, operator_id="carol", credential=TOKENS["carol"])
    with pytest.raises(DestructiveGrantApprovalError):
        issue_destructive_grant(
            **FIELDS,
            store=store,
            approval_operator_id="alice",
            approval_credential=TOKENS["alice"],
        )
    assert not store._records

    grant = destructive_grants.issue(
        **FIELDS,
        store=store,
        approval_operator_id="bob",
        approval_credential=TOKENS["bob"],
    )
    assert store.get(grant.grant_id) is not None
    assert grant.object_id == "pay_123"
    assert grant.approval_operator_ids == ("alice", "bob")
    assert store.get(grant.grant_id)["grant"]["approval_operator_ids"] == ["alice", "bob"]
    with pytest.raises(DestructiveGrantApprovalError):
        _issue(store)
    assert len(store._records) == 1


@pytest.mark.parametrize(
    "drift",
    [
        {"object_id": "pay_456"},
        {"tenant": "other"},
        {"account": "acct-2"},
        {"expires_in": 240.0},
        {"max_uses": 2},
        {"policy_version": "2026.10"},
        {"run_id": "run-2"},
        {"bind_run_id": False},
    ],
)
def test_changed_grant_fields_invalidate_approval(drift: dict[str, object]) -> None:
    store = InMemoryDestructiveGrantStore()
    authorizer = _authorizer()
    set_destructive_grant_approval_authorizer(authorizer)
    assert authorizer.approve(_request(), operator_id="alice", credential=TOKENS["alice"])

    with pytest.raises(DestructiveGrantApprovalError):
        _issue(store, **drift)
    assert not store._records
    assert _issue(store).object_id == "pay_123"


def test_expired_approval_can_be_replaced_but_cannot_issue() -> None:
    now = [100.0]
    authorizer = _authorizer(approval_ttl=10, clock=lambda: now[0])
    store = InMemoryDestructiveGrantStore()
    set_destructive_grant_approval_authorizer(authorizer)
    assert authorizer.approve(_request(), operator_id="alice", credential=TOKENS["alice"])
    now[0] = 110.0
    with pytest.raises(DestructiveGrantApprovalError):
        _issue(store)
    assert not store._records
    assert authorizer.approve(_request(), operator_id="alice", credential=TOKENS["alice"])
    assert _issue(store).object_id == "pay_123"


def test_approval_is_shared_across_independent_durable_backend_instances(tmp_path) -> None:
    path = tmp_path / "approvals.json"
    first = _authorizer(approval_backend=FileAtomicStateBackend(path))
    second = _authorizer(approval_backend=FileAtomicStateBackend(path))
    store = InMemoryDestructiveGrantStore()
    assert first.approve(_request(), operator_id="alice", credential=TOKENS["alice"])
    set_destructive_grant_approval_authorizer(second)

    assert _issue(store).approval_operator_ids == ("alice", "bob")
    with pytest.raises(DestructiveGrantApprovalError):
        _issue(store)


def test_concurrent_second_approvals_issue_only_one_grant() -> None:
    store = InMemoryDestructiveGrantStore()
    authorizer = _authorizer()
    set_destructive_grant_approval_authorizer(authorizer)
    assert authorizer.approve(_request(), operator_id="alice", credential=TOKENS["alice"])

    def issue(operator_id: str):
        try:
            return issue_destructive_grant(
                **FIELDS,
                store=store,
                approval_operator_id=operator_id,
                approval_credential=TOKENS[operator_id],
            )
        except DestructiveGrantApprovalError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        grants = list(pool.map(issue, ("bob", "carol")))
    assert sum(grant is not None for grant in grants) == 1
    assert len(store._records) == 1


def test_approval_storage_failure_never_mints_a_grant(monkeypatch) -> None:
    backend = InMemoryAtomicStateBackend()
    authorizer = _authorizer(approval_backend=backend)
    store = InMemoryDestructiveGrantStore()
    set_destructive_grant_approval_authorizer(authorizer)
    assert authorizer.approve(_request(), operator_id="alice", credential=TOKENS["alice"])

    def fail_delete(*_args, **_kwargs):
        raise RuntimeError("approval backend unavailable")

    monkeypatch.setattr(backend, "delete", fail_delete)
    with pytest.raises(DestructiveGrantApprovalError):
        _issue(store)
    assert not store._records


def test_approval_credentials_without_bound_authorizer_fail_closed() -> None:
    with pytest.raises(DestructiveGrantApprovalError, match="no destructive grant approval"):
        _issue(InMemoryDestructiveGrantStore())


@pytest.mark.parametrize("expires_in", [True, float("nan"), float("inf")])
def test_invalid_expiry_cannot_be_approved(expires_in: object) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        _request(expires_in=expires_in)
