from contextlib import closing
import sqlite3
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


DATABASE_PATH = Path(__file__).resolve().with_name("chat_history.sqlite")
LEGACY_DATABASE_PATH = Path.home() / ".aibook" / "chat_checkpoints.sqlite"

llm = ChatOllama(model="llama3.2:latest")


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def get_checkpoint() -> SqliteSaver:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATABASE_PATH.exists() and LEGACY_DATABASE_PATH.exists():
        with closing(sqlite3.connect(LEGACY_DATABASE_PATH)) as source:
            with closing(sqlite3.connect(DATABASE_PATH)) as destination:
                source.backup(destination)

    connection = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(connection)
    checkpointer.setup()
    return checkpointer


checkpoint = get_checkpoint()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpoint)


def get_saved_chats() -> list[tuple[str, str]]:
    chats: dict[str, tuple[str, str]] = {}
    for saved_checkpoint in checkpoint.list(None):
        thread_id = saved_checkpoint.config["configurable"]["thread_id"]
        messages = saved_checkpoint.checkpoint.get("channel_values", {}).get(
            "messages", []
        )
        first_user_message = next(
            (message for message in messages if message.type == "human"),
            None,
        )

        title = chats.get(thread_id, ("", ""))[1]
        if not title and first_user_message:
            title = " ".join(str(first_user_message.content).split())
            title = title[:48] + ("..." if len(title) > 48 else "")
        if title:
            timestamp = str(saved_checkpoint.checkpoint.get("ts", ""))
            existing = chats.get(thread_id)
            if existing is None or timestamp > existing[0]:
                chats[thread_id] = (timestamp, title)

    ordered_chats = sorted(
        chats.items(),
        key=lambda item: item[1][0],
        reverse=True,
    )
    return [(thread_id, title) for thread_id, (_, title) in ordered_chats]
