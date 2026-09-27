
from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver


# -----------------------------
# LLM
# -----------------------------
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    streaming=True
)


# -----------------------------
# Google Search Tool
# -----------------------------
search = GoogleSerperAPIWrapper()

tools = [search.run]


# -----------------------------
# Session Memory
# -----------------------------
if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()

if "history" not in st.session_state:
    st.session_state.history = []


# -----------------------------
# Agent
# -----------------------------
agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=st.session_state.memory,
    system_prompt=(
        "You are an amazing AI agent. "
        "You can search Google when necessary. "
        "Give clear, accurate and helpful answers."
    )
)


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("QuickAnswer")
st.subheader("Answers at the speed of thought")


# Display previous messages
for message in st.session_state.history:
    role = message["role"]
    content = message["content"]

    with st.chat_message(role):
        st.markdown(content)


# Chat input
query = st.chat_input("Ask Anything?")


if query:

    # Display user message
    with st.chat_message("user"):
        st.markdown(query)

    # Save user message
    st.session_state.history.append(
        {
            "role": "user",
            "content": query
        }
    )

    # -----------------------------
    # Stream Agent Response
    # -----------------------------
    response = agent.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ]
        },
        {
            "configurable": {
                "thread_id": "1"
            }
        },
        stream_mode="messages"
    )

    # AI response container
    with st.chat_message("assistant"):

        message = ""
        response_placeholder = st.empty()

        for chunk, metadata in response:

            # chunk is an AIMessageChunk
            content = chunk.content

            if content:
                message += content

                # Update Streamlit UI
                response_placeholder.markdown(message)

    # Save AI response
    st.session_state.history.append(
        {
            "role": "assistant",
            "content": message
        }
    )
