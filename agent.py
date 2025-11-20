import os
import toml
from typing import Literal
import uuid
from tavily import TavilyClient
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore

from deepagents import create_deep_agent
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
config = toml.load(".streamlit/secrets.toml")
os.environ["TAVILY_API_KEY"] = config["settings"]["TAVILY_API_KEY"]
os.environ["ANTHROPIC_API_KEY"] = config["settings"]["ANTHROPIC_API_KEY"]

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


def make_backend(runtime):
    return CompositeBackend(
        default=StateBackend(runtime),
        routes={"/memories/": StoreBackend(runtime)},
    )


research_instructions = """You are an expert researcher. Your job is to conduct thorough research and then write a polished report."""

agent = create_deep_agent(
    tools=[internet_search],
    system_prompt=research_instructions,
    store=InMemoryStore(),
    backend=make_backend,
)

st.subheader("💬 Internet Research Helper")

if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("Ask me anything...")

if user_input:
    result = agent.invoke(
        {
            "messages": st.session_state.messages
            + [{"role": "user", "content": user_input}]
        },
        thread_id=st.session_state.chat_id,
    )

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append(
        {"role": "assistant", "content": result["messages"][-1].content}
    )


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
