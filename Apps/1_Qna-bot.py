from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import streamlit as st
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)
st.title("AskBuddy- AI Qna Bot")
st.markdown("My Qna Bot with langchain and Google Gemini !")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("Ask Anything")
if query:
    st.session_state.messages.append({"role":"user", "content":query})
    st.chat_message("User").markdown(query)
    res=llm.invoke(query)
    st.chat_message("AI").markdown(res.content)
    st.session_state.messages.append({"role":"AI", "content":query})