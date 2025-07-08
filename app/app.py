"""
Streamlit-based frontend for the Running Assistant RAG application.

This interactive UI allows users to input running-related questions and receive
AI-generated answers with semantic search via a RAG pipeline. The app logs each
conversation, tracks token usage and OpenAI API costs, and stores user feedback
in a PostgreSQL database. It also displays recent conversations and aggregated
feedback statistics for monitoring and evaluation.

Main features:
- Text input for user questions
- Real-time answers from the RAG pipeline
- Conversation logging and token cost tracking
- Feedback buttons (+1 / -1) for user evaluation
- Recent conversation browser with relevance filtering
- Aggregated thumbs-up/down statistics
"""

import time
import uuid
import streamlit as st

from rag import rag_pipeline
from db import (
    save_conversation,
    save_feedback,
    get_recent_conversations,
    get_feedback_stats,
)


def print_log(message):
    """
    Print a log message immediately, flushing the output buffer.
    """
    print(message, flush=True)


def main():
    """
    Run the Streamlit app for the Running Assistant, handling user input, displaying answers,
    managing feedback, and showing recent conversations and feedback statistics.
    """
    print_log("Starting the Running Assistant application")
    st.title("This is your personal Running Assistant")

    # Session state initialization
    # if "conversation_id" not in st.session_state:
    #     st.session_state.conversation_id = str(uuid.uuid4())
    #     print_log(
    #         f"New conversation started with ID: {st.session_state.conversation_id}"
    #     )
    if "count" not in st.session_state:
        st.session_state.count = 0
        print_log("Feedback count initialized to 0")


    # User input
    user_input = st.text_input("Enter your question:")

    if st.button("Ask"):
        # Generate a new conversation ID for the question
        st.session_state.conversation_id = str(uuid.uuid4())
        print_log(f"User asked: '{user_input}'")
        with st.spinner("Processing..."):
            print_log(
                "Getting answer from running assistant"
            )
            start_time = time.time()
            answer_data = rag_pipeline(user_input)
            end_time = time.time()
            print_log(f"Answer received in {end_time - start_time:.2f} seconds")
            st.success("Completed!")
            st.write(answer_data["answer"])

            # Display monitoring information
            st.write(f"Response time: {answer_data['response_time']:.2f} seconds")
            st.write(f"Relevance: {answer_data['relevance']}")
            st.write(f"Total tokens: {answer_data['total_tokens']}")
            if answer_data["openai_cost"] > 0:
                st.write(f"OpenAI cost: ${answer_data['openai_cost']:.4f}")

            # Save conversation to database
            print_log("Saving conversation to database")
            save_conversation(
                st.session_state.conversation_id, user_input, answer_data
            )
            print_log("Conversation saved successfully")
            # Generate a new conversation ID for next question
            # st.session_state.conversation_id = str(uuid.uuid4())

    # Feedback buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("+1"):
            st.session_state.count += 1
            print_log(
                f"Positive feedback received. New count: {st.session_state.count}"
            )
            save_feedback(st.session_state.conversation_id, 1)
            print_log("Positive feedback saved to database")
    with col2:
        if st.button("-1"):
            st.session_state.count -= 1
            print_log(
                f"Negative feedback received. New count: {st.session_state.count}"
            )
            save_feedback(st.session_state.conversation_id, -1)
            print_log("Negative feedback saved to database")

    st.write(f"Current count: {st.session_state.count}")

    # Display recent conversations
    st.subheader("Recent Conversations")
    relevance_filter = st.selectbox(
        "Filter by relevance:", ["All", "RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT"]
    )
    recent_conversations = get_recent_conversations(
        limit=5, relevance=relevance_filter if relevance_filter != "All" else None
    )
    for conv in recent_conversations:
        st.write(f"Q: {conv['question']}")
        st.write(f"A: {conv['answer']}")
        st.write(f"Relevance: {conv['relevance']}")
        st.write("---")

    # Display feedback stats
    feedback_stats = get_feedback_stats()
    st.subheader("Feedback Statistics")
    st.write(f"Thumbs up: {feedback_stats['thumbs_up']}")
    st.write(f"Thumbs down: {feedback_stats['thumbs_down']}")


print_log("Streamlit app loop completed")


if __name__ == "__main__":
    print_log("Running Assistant application started")
    main()
