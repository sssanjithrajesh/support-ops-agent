from __future__ import annotations

import pytest

from soa.config import policy
from soa.kb import KnowledgeBase
from soa.tickets import training_tickets
from soa.triage import Triager


@pytest.fixture(scope="session")
def pol():
    return policy()


@pytest.fixture(scope="session")
def triager(pol):
    return Triager(pol, training_tickets())


@pytest.fixture(scope="session")
def kb(pol):
    return KnowledgeBase.load(pol.kb_dir)
