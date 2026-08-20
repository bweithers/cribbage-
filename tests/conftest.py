"""Shared pytest configuration.

The exhaustive scoring check scores 13M hands per pass, so it is opt-in
via ``--runslow`` rather than running on every invocation.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False,
        help="also run tests marked slow (the exhaustive scoring check)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: exhaustive; needs --runslow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="needs --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
