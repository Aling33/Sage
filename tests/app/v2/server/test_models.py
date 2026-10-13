from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.v2.server.catalog.records import apply_upsert, empty_catalog
from app.v2.server.routers.schemas.catalog import ModelBody
from app.v2.server.runtime.models import HostModelProvider, _build_catalog_provider
from sagents.v2.model.contracts import ModelEventKind, ModelRequest
from tests.app.v2.server.conftest import scripted_hello
from tests.app.v2.server.fakes import MemoryCatalogStore


async def _save_demo_model(catalog: MemoryCatalogStore, user_id: str = "user-1"):
    return await catalog.upsert_model(
        user_id,
        {
            "protocol": "openai-chat-completions",
            "base_url": "https://example.invalid/v1",
            "model": "demo-model",
            "api_key": "sk-test",
            "is_default": True,
        },
    )


@pytest.mark.asyncio
async def test_host_provider_uses_session_bind_when_contextvar_empty(monkeypatch):
    catalog = MemoryCatalogStore()
    await _save_demo_model(catalog)
    monkeypatch.setattr(
        "app.v2.server.runtime.models._build_catalog_provider",
        lambda record: scripted_hello(),
    )

    async def get_run(run_id: str):
        assert run_id == "run_sagents"
        return SimpleNamespace(session_id="thread-1")

    provider = HostModelProvider(
        catalog, session_store=SimpleNamespace(get_run=get_run)
    )
    provider.bind_session_user("thread-1", "user-1")

    events = [
        event
        async for event in provider.stream(
            ModelRequest(
                request_id="request_1",
                run_id="run_sagents",
                model_binding="primary",
                messages=(),
            )
        )
    ]
    assert any(event.kind == ModelEventKind.TEXT_DELTA for event in events)


def test_catalog_uses_orcarouter_default_base_url_when_omitted():
    record, _ = apply_upsert(
        empty_catalog(),
        {
            "protocol": "orcarouter",
            "model": "orcarouter-model",
            "api_key": "test-key",
        },
    )
    assert record.base_url == "https://api.orcarouter.ai/v1"


def test_catalog_preserves_custom_base_url_for_orcarouter():
    record, _ = apply_upsert(
        empty_catalog(),
        {
            "protocol": "orcarouter",
            "base_url": "https://gateway.example/v1",
            "model": "orcarouter-model",
            "api_key": "test-key",
        },
    )
    assert record.base_url == "https://gateway.example/v1"


@pytest.mark.parametrize("base_url", [None, "https://gateway.example/v1"])
def test_orcarouter_api_payload_builds_openai_compatible_provider(
    monkeypatch, base_url
):
    from sagents.v2.model import OpenAIChatCompletionsModelProvider

    monkeypatch.setattr(
        "sagents.v2.model.plugins.openai_compatible.AsyncOpenAI",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    payload = {
        "protocol": "orcarouter",
        "model": "orcarouter/auto",
        "api_key": "test-key",
    }
    if base_url is not None:
        payload["base_url"] = base_url
    record, _ = apply_upsert(empty_catalog(), ModelBody(**payload).model_dump())
    provider = _build_catalog_provider(record)

    assert isinstance(provider, OpenAIChatCompletionsModelProvider)
    assert provider.config.base_url == (base_url or "https://api.orcarouter.ai/v1")
    assert provider.config.model == "orcarouter/auto"
