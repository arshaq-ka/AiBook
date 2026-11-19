import uuid

import streamlit as st
from langchain_core.messages import HumanMessage

from backend import chatbot, get_saved_chats


def select_prompt(prompt: str) -> None:
    st.session_state["suggested_prompt"] = prompt


def run_streamlit_app():
    st.set_page_config(
        page_title="EON Chat",
        page_icon="💬",
        initial_sidebar_state="expanded",
    )

    saved_chats = get_saved_chats()
    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = saved_chats[0][0] if saved_chats else uuid.uuid4().hex
            st.query_params["thread_id"] = thread_id
        st.session_state.thread_id = thread_id

    with st.sidebar:
        st.markdown("## EON Chat")
        st.markdown("### Chat history")
        if st.button("➕ New chat", use_container_width=True):
            st.session_state.thread_id = uuid.uuid4().hex
            st.query_params["thread_id"] = st.session_state.thread_id
            st.rerun()

        for saved_thread_id, title in saved_chats:
            if st.button(
                title,
                key=f"chat_{saved_thread_id}",
                use_container_width=True,
            ):
                st.session_state.thread_id = saved_thread_id
                st.query_params["thread_id"] = saved_thread_id
                st.rerun()

    thread_config = {
        "configurable": {"thread_id": st.session_state.thread_id}
    }
    if st.session_state.get("chat_history_thread_id") != st.session_state.thread_id:
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
        st.session_state.chat_history_thread_id = st.session_state.thread_id

    suggested_prompt = st.session_state.pop("suggested_prompt", None)

    if not st.session_state.chat_history and not suggested_prompt:
        st.markdown(
            """
            <style>
            .welcome-panel {
                max-width: 720px;
                margin: 10vh auto 1.5rem;
                text-align: center;
            }
            .welcome-panel h2 {
                font-size: 2rem;
                font-weight: 600;
                margin-bottom: 0.5rem;
            }
            .welcome-panel p {
                color: var(--text-color);
                opacity: 0.7;
                font-size: 1.05rem;
            }
            </style>
            <div class="welcome-panel">
                <h2>What can I help you with?</h2>
                <p>Ask a question or choose a place to start.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        prompt_columns = st.columns(2)
        suggestions = [
            (
                "🧠 Explain a topic",
                "Break down a tricky idea",
                "Explain a difficult topic in simple terms, with a helpful example.",
            ),
            (
                "✍️ Help me write",
                "Draft something clear and polished",
                "Help me write a clear, friendly email about an important topic.",
            ),
            (
                "💡 Brainstorm ideas",
                "Find inspiration for a project",
                "Give me five creative ideas for a personal project.",
            ),
            (
                "📚 Find my next book",
                "Get a thoughtful recommendation",
                "Recommend a book and explain why I might enjoy reading it.",
            ),
        ]
        for index, (label, description, prompt_text) in enumerate(suggestions):
            with prompt_columns[index % 2]:
                with st.container(border=True):
                    st.button(
                        label,
                        key=f"suggestion_{index}",
                        use_container_width=True,
                        on_click=select_prompt,
                        args=(prompt_text,),
                    )
                    st.caption(description)

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Send a message")
    if suggested_prompt:
        prompt = suggested_prompt

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
