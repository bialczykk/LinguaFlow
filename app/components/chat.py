"""Reusable chat interface component for conversational projects."""

from typing import Callable

import streamlit as st


def render(
    *,
    history_key: str,
    on_user_message: Callable[[str], str],
    placeholder: str = "Type a message...",
    intro_message: str | None = None,
) -> None:
    """Render a chat interface with message history.

    Args:
        history_key: Session state key for this chat's message history.
            The history is a list of dicts: {"role": "user"|"assistant", "content": str}
        on_user_message: Callback that takes the user's message string and returns
            the assistant's response string. Called when the user submits a message.
        placeholder: Placeholder text for the chat input.
        intro_message: Optional welcome message shown as the first assistant message.
    """
    # -- Initialize history if needed --
    if history_key not in st.session_state:
        st.session_state[history_key] = []
        if intro_message:
            st.session_state[history_key].append(
                {"role": "assistant", "content": intro_message}
            )

    # -- Display existing messages --
    for msg in st.session_state[history_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # -- Chat input --
    if user_input := st.chat_input(placeholder):
        # Show user message immediately
        st.session_state[history_key].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = on_user_message(user_input)
            st.markdown(response)
        st.session_state[history_key].append({"role": "assistant", "content": response})
