"""Shared fixtures: synthetic papers and a network block.

The network block fails any outbound TCP connection attempted during a test,
which is how the suite proves the comparison pipeline runs fully offline.
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
sys.path.insert(0, str(TESTS.parent / "src"))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from pdfgen import build_paper_a, build_paper_b, build_paper_c  # noqa: E402


class _NetworkBlocked(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise _NetworkBlocked("network access attempted during a test")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield


@pytest.fixture(scope="session")
def papers(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("papers")
    return {
        "a": build_paper_a(d / "paper_a.pdf"),
        "b": build_paper_b(d / "paper_b.pdf"),
        "c": build_paper_c(d / "paper_c.pdf"),
    }


@pytest.fixture(scope="session")
def cache_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("cache")
