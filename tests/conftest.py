from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure src/ is on the Python path for test imports
_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


@pytest.fixture(autouse=True)
def _reset_shared_llm() -> None:
    import llm.client as llm_client

    llm_client._default_llm = None
    yield
    llm_client._default_llm = None
