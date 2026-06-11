from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_shared_llm() -> None:
    import research_swarm.llm.client as llm_client

    llm_client._default_llm = None
    yield
    llm_client._default_llm = None
