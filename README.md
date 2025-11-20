# Internet Research Bot With Memory

Internet Research Agent with Tavily DeepAgents, and Streamlit

## Quick Start

1. Clone repository.
2. Create `.env` file inside of project root.
3. Get Tavily API key at https://app.tavily.com/home . Make sure you selected "Limit monthly usage\*" option.
   and add it as `TAVILY_API_KEY` to `.env`.
4. Get Anthropic API key at https://console.anthropic.com/ and add it as `ANTHROPIC_API_KEY` to `.env`.
5. Run `source bin/activate` from project root to activate virtual environment.
6. Run `poetry install` from virtual environment.
7. Run `streamlit run agent.py` from virtual environment.
