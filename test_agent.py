import pytest
from unittest.mock import MagicMock
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore

from agent import (
    internet_search,
    make_backend,
    create_deep_agent,
    research_instructions,
)


# --- Fixtures ---
@pytest.fixture
def mock_tavily(monkeypatch):
    mock_client = MagicMock()
    mock_client.search.return_value = {"results": ["mock result"]}
    monkeypatch.setattr("agent.tavily_client", mock_client)
    return mock_client


# --- Tests ---


def test_internet_search_returns_results(mock_tavily):
    result = internet_search("python testing", max_results=3, topic="general")
    assert "results" in result
    assert result["results"][0] == "mock result"
    mock_tavily.search.assert_called_once_with(
        "python testing", max_results=3, include_raw_content=False, topic="general"
    )


def test_make_backend_creates_composite():
    runtime = MagicMock()
    backend = make_backend(runtime)
    assert isinstance(backend, CompositeBackend)
    # Ensure routes are set correctly
    assert "/memories/" in backend.routes


def test_agent_creation(mock_tavily):
    agent = create_deep_agent(
        tools=[internet_search],
        system_prompt=research_instructions,
        store=InMemoryStore(),
        backend=make_backend,
    )
    assert agent is not None


def test_agent_invoke_with_message(mock_tavily):
    agent = create_deep_agent(
        tools=[internet_search],
        system_prompt=research_instructions,
        store=InMemoryStore(),
        backend=make_backend,
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Hello"}]}, thread_id="test-thread"
    )
    # The agent should return a dict with messages
    assert "messages" in result
    assert isinstance(result["messages"], list)


def test_agent_memory_persists_across_turns(mock_tavily):
    agent = create_deep_agent(
        tools=[internet_search],
        system_prompt=research_instructions,
        store=InMemoryStore(),
        backend=make_backend,
    )

    thread_id = "memory-thread"
    messages = [{"role": "user", "content": "What is Python?"}]

    # First turn
    result1 = agent.invoke(
        {"messages": messages},
        thread_id=thread_id,
    )
    assert any(m.content for m in result1["messages"])

    messages += [{"role": "user", "content": "Can you summarize that?"}]
    # Second turn (follow-up)
    result2 = agent.invoke(
        {"messages": messages},
        thread_id=thread_id,
    )

    # The agent should remember the first turn and include context
    all_contents = [m.content for m in result2["messages"]]
    assert any(
        "Python" in c for c in all_contents
    ), "Agent did not recall previous context"
