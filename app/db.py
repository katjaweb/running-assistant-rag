"""
Database utilities for managing conversation and feedback data
for the Running Assistant application.

This module provides functions to initialize the PostgreSQL database schema,
save user conversations and feedback, and retrieve recent conversation records
along with aggregated feedback statistics. It uses psycopg2 for database interaction
and ensures all timestamps are stored with the correct timezone (Europe/Berlin).
"""

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import psycopg2
from psycopg2.extras import DictCursor

tz = ZoneInfo("Europe/Berlin")


def get_db_connection():
    """
    Establish and return a connection to the PostgreSQL database.
    """
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        database=os.getenv("POSTGRES_DB", "running_assistant"),
        user=os.getenv("POSTGRES_USER", "your_username"),
        password=os.getenv("POSTGRES_PASSWORD", "your_password"),
    )


def init_db():
    """
    Initialize the PostgreSQL database by creating the conversations and feedback tables.
    Drops existing tables if they exist.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS feedback")
            cur.execute("DROP TABLE IF EXISTS conversations")

            cur.execute("""
                CREATE TABLE conversations (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    response_time FLOAT NOT NULL,
                    relevance TEXT NOT NULL,
                    relevance_explanation TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    eval_prompt_tokens INTEGER NOT NULL,
                    eval_completion_tokens INTEGER NOT NULL,
                    eval_total_tokens INTEGER NOT NULL,
                    openai_cost FLOAT NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE feedback (
                    id SERIAL PRIMARY KEY,
                    conversation_id TEXT REFERENCES conversations(id),
                    feedback INTEGER NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
            """)
        conn.commit()
    finally:
        conn.close()


def save_conversation(conversation_id, question, answer_data, timestamp=None):
    """
    Save a conversation record to the database, including metadata and evaluation metrics.

    Args:
        conversation_id (str): Unique identifier for the conversation.
        question (str): The user's question.
        answer_data (dict): Dictionary containing the answer and related metadata.
        timestamp (datetime, optional): Timestamp of the conversation. Defaults to current time.
    """
    if timestamp is None:
        timestamp = datetime.now(tz)

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations 
                (id, question, answer, response_time, relevance, 
                relevance_explanation, prompt_tokens, completion_tokens, total_tokens, 
                eval_prompt_tokens, eval_completion_tokens, eval_total_tokens, openai_cost, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, COALESCE(%s, CURRENT_TIMESTAMP))
            """,
                (
                    conversation_id,
                    question,
                    answer_data["answer"],
                    answer_data["response_time"],
                    answer_data["relevance"],
                    answer_data["relevance_explanation"],
                    answer_data["prompt_tokens"],
                    answer_data["completion_tokens"],
                    answer_data["total_tokens"],
                    answer_data["eval_prompt_tokens"],
                    answer_data["eval_completion_tokens"],
                    answer_data["eval_total_tokens"],
                    answer_data["openai_cost"],
                    timestamp,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def save_feedback(conversation_id, feedback, timestamp=None):
    """
    Save user feedback for a specific conversation to the database.

    Args:
        conversation_id (str): ID of the conversation the feedback relates to.
        feedback (int): Feedback score (e.g., 1 for positive, -1 for negative).
        timestamp (datetime, optional): Timestamp of the feedback. Defaults to current time.
    """
    if timestamp is None:
        timestamp = datetime.now(tz)

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO feedback (conversation_id, feedback, timestamp) VALUES (%s, %s, COALESCE(%s, CURRENT_TIMESTAMP))",
                (conversation_id, feedback, timestamp),
            )
        conn.commit()
    finally:
        conn.close()


def get_recent_conversations(limit=5, relevance=None):
    """
    Retrieve a list of recent conversations, optionally filtered by relevance.

    Args:
        limit (int): Maximum number of conversations to return. Defaults to 5.
        relevance (str, optional): Filter conversations by relevance category.

    Returns:
        list[dict]: List of conversation records with optional feedback.
    """
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            query = """
                SELECT c.*, f.feedback
                FROM conversations c
                LEFT JOIN feedback f ON c.id = f.conversation_id
            """
            if relevance:
                query += f" WHERE c.relevance = '{relevance}'"
            query += " ORDER BY c.timestamp DESC LIMIT %s"

            cur.execute(query, (limit,))
            return cur.fetchall()
    finally:
        conn.close()


def get_feedback_stats():
    """
    Retrieve aggregated feedback statistics, including counts of positive and negative feedback.

    Returns:
        dict: Dictionary with keys 'thumbs_up' and 'thumbs_down'.
    """
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute("""
                SELECT 
                    SUM(CASE WHEN feedback > 0 THEN 1 ELSE 0 END) as thumbs_up,
                    SUM(CASE WHEN feedback < 0 THEN 1 ELSE 0 END) as thumbs_down
                FROM feedback
            """)
            return cur.fetchone()
    finally:
        conn.close()
