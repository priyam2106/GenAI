import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)

print("Environment file:", ENV_FILE)
print("Groq key loaded:", bool(os.getenv("GROQ_API_KEY")))
print("Serper key loaded:", bool(os.getenv("SERPER_API_KEY")))

from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

search = GoogleSerperAPIWrapper()


agent = create_agent(
    model=llm,
    tools=[search.run],
    system_prompt="You are a agent and can search for any question on google.",
    checkpointer=MemorySaver(),
    )


while True:
    query = input("User: ")
    if query.lower() == "quit":
        print("Good Bye")
        break
    response = agent.invoke({"messages":[{"role":"user","content":query}]},
                            {"configurable":{"thread_id":"xyz123"}})
    print("AI", response["messages"][-1].content)
