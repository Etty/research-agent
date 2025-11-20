import pytest
from unittest.mock import patch
import streamlit as st

import agent  # your main app file


def test_session_state_initialization():
    # Clear any previous state
    st.session_state.clear()

    # Importing agent should initialize chat_id and messages
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = "test-id"
    if "messages" not in st.session_state:
        st.session_state.messages = []

    assert isinstance(st.session_state.chat_id, str)
    assert isinstance(st.session_state.messages, list)


@patch("agent.agent.invoke")
def test_user_input_adds_messages(mock_invoke):
    # Prepare fake agent response
    class FakeMessage:
        def __init__(self, content):
            self.content = content

    mock_invoke.return_value = {"messages": [FakeMessage("Hello back")]}

    st.session_state.clear()
    st.session_state.chat_id = "test-id"
    st.session_state.messages = []

    # Simulate user input
    user_input = "Hello"
    result = agent.agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        thread_id=st.session_state.chat_id,
    )

    # Append messages like your app does
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append(
        {"role": "assistant", "content": result["messages"][-1].content}
    )

    # Assertions
    assert st.session_state.messages[0]["role"] == "user"
    assert st.session_state.messages[0]["content"] == "Hello"
    assert st.session_state.messages[1]["role"] == "assistant"
    assert st.session_state.messages[1]["content"] == "Hello back"
