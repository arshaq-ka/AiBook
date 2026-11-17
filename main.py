import sqlite3
import uuid
from pathlib import Path
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import streamlit as st


llm = ChatOllama(model="llama3.2:latest")


class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]



def chat_node(state: ChatState):
    #take user query from state
    messages = state['messages']
    # send to llm
    response = llm.invoke(messages)
    # response store state
    return {'messages': [response]}



@st.cache_resource
def get_checkpoint() -> SqliteSaver:
    checkpoint_path = Path.home() / ".aibook" / "chat_checkpoints.sqlite"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
    checkpointer = SqliteSaver(connection)
    checkpointer.setup()
    return checkpointer


checkpoint = get_checkpoint()

graph = StateGraph(ChatState)

# add nodes
graph.add_node('chat_node', chat_node)

#add edges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)


chatbot = graph.compile(checkpointer=checkpoint)


def run_streamlit_app():
    st.set_page_config(page_title="Ollama Chat", page_icon="💬")
    st.title("Ollama Chat")

    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = uuid.uuid4().hex
            st.query_params["thread_id"] = thread_id
        st.session_state.thread_id = thread_id

    thread_config = {
        "configurable": {"thread_id": st.session_state.thread_id}
    }
    if "chat_history" not in st.session_state:
        saved_state = chatbot.get_state(thread_config)
        messages = saved_state.values.get("messages", [])
        st.session_state.chat_history = [
            {
                "role": "user" if message.type == "human" else "assistant",
                "content": message.content,
            }
            for message in messages
            if message.type in ("human", "ai")
        ]

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Send a message")
    if prompt:
        st.session_state.chat_history.append(
            {"role": "user", "content": prompt}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = chatbot.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config=thread_config,
                )
                response = result["messages"][-1].content
            st.markdown(response)
        st.session_state.chat_history.append(
            {"role": "assistant", "content": response}
        )


if __name__ == "__main__":
    run_streamlit_app()