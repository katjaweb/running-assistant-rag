"""
Synthetic Data Generator for Running Assistant

This script generates synthetic conversation and feedback data to populate
the PostgreSQL database for testing and development purposes.

It supports two modes:
- Historical data generation over a specified time range
- Continuous live data generation

Each conversation includes a randomly selected running-related question,
a predefined answer, metadata such as response time, token usage, relevance,
and optional feedback.

Generated data is saved using the `save_conversation` and `save_feedback` functions from the db module.
"""

import time
import random
import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from db import save_conversation, save_feedback

# Set the timezone to CET (Europe/Berlin)
tz = ZoneInfo("Europe/Berlin")

# List of sample questions and answers
SAMPLE_QUESTIONS = [
    "What is interval training?",
    "What are long, slow runs good for?",
    "Why is recovery important in running?",
    "Why is hydration important when running?",
    "How can you recognize overtraining?",
    "How do I find the right running shoe?",
    "What role does nutrition play in running?",
    "Why is warming up important?",
    "Should you stretch before or after running?",
    "What are tempo runs?",
]

SAMPLE_ANSWERS = [
    "Interval training is a training method in which phases of intense effort alternate with recovery phases. This alternation between high and low intensity effectively improves endurance and speed, as the body is challenged both anaerobically and aerobically.",
    "Long, slow runs are a central component of endurance training. They help improve basic endurance, strengthen the cardiovascular system, and increase the body's ability to burn fat. These runs also help prepare muscles and tendons for longer periods of exertion.",
    "Recovery is an essential part of the training process. During recovery phases, the body can recover from training stimuli, repair muscle tissue, and adapt to the exertion. Without sufficient recovery, the risk of injury, overuse, and decreased performance increases.",
    "Hydration is very important when running, as the body loses a lot of water through sweating. Inadequate hydration can lead to dehydration, which in turn can cause fatigue, cramps, and decreased performance. Therefore, you should drink regularly before, during, and after training.",
    "Overtraining occurs when training volume or intensity is too high and recovery time is too short. Typical symptoms include persistent fatigue, decreased performance, sleep disturbances, and increased susceptibility to injury. It is important to recognize the body's warning signs and respond accordingly.",
    "The optimal running shoe should fit well, provide sufficient cushioning, and match your running style to help prevent injuries.",
    "A balanced diet with sufficient carbohydrates, proteins, and fats supports performance and recovery in running.",
    "Warming up before a run prepares the body for exertion and reduces the risk of injury by increasing blood flow and mobilizing the joints.",
    "Stretching after a run promotes flexibility and supports recovery, but it should not be too intense or static before running.",
    "Tempo runs are runs at a faster but controlled pace that help increase the anaerobic threshold and improve running speed.",
]

RELEVANCE = ["RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT"]


def generate_synthetic_data(start_time, end_time):
    """
    Generates and saves synthetic conversation and feedback data within a given time range.

    Args:
        start_time (datetime): Start timestamp for data generation.
        end_time (datetime): End timestamp for data generation.

    Each entry includes a random question, answer, relevance, token stats, and optional feedback.
    """
    current_time = start_time
    conversation_count = 0
    print(f"Starting historical data generation from {start_time} to {end_time}")
    while current_time < end_time:
        conversation_id = str(uuid.uuid4())
        question = random.choice(SAMPLE_QUESTIONS)
        answer = random.choice(SAMPLE_ANSWERS)
        relevance = random.choice(RELEVANCE)

        openai_cost = 0

        openai_cost = random.uniform(0.001, 0.1)

        answer_data = {
            "answer": answer,
            "response_time": random.uniform(0.5, 5.0),
            "relevance": relevance,
            "relevance_explanation": f"This answer is {relevance.lower()} to the question.",
            "prompt_tokens": random.randint(50, 200),
            "completion_tokens": random.randint(50, 300),
            "total_tokens": random.randint(100, 500),
            "eval_prompt_tokens": random.randint(50, 150),
            "eval_completion_tokens": random.randint(20, 100),
            "eval_total_tokens": random.randint(70, 250),
            "openai_cost": openai_cost,
        }

        save_conversation(conversation_id, question, answer_data, current_time)
        print(
            f"Saved conversation: ID={conversation_id}, Time={current_time}"
        )

        if random.random() < 0.7:
            feedback = 1 if random.random() < 0.8 else -1
            save_feedback(conversation_id, feedback, current_time)
            print(
                f"Saved feedback for conversation {conversation_id}: {'Positive' if feedback > 0 else 'Negative'}"
            )

        current_time += timedelta(minutes=random.randint(1, 15))
        conversation_count += 1
        if conversation_count % 10 == 0:
            print(f"Generated {conversation_count} conversations so far...")

    print(
        f"Historical data generation complete. Total conversations: {conversation_count}"
    )


def generate_live_data():
    """
    Continuously generates and saves synthetic live conversation and feedback data.

    Simulates real-time interactions by creating random conversation entries and feedback
    in a loop with a 1-second delay between entries.
    """
    conversation_count = 0
    print("Starting live data generation...")
    while True:
        current_time = datetime.now(tz)
        # current_time = None
        conversation_id = str(uuid.uuid4())
        question = random.choice(SAMPLE_QUESTIONS)
        answer = random.choice(SAMPLE_ANSWERS)
        relevance = random.choice(RELEVANCE)

        openai_cost = 0

        openai_cost = random.uniform(0.001, 0.1)

        answer_data = {
            "answer": answer,
            "response_time": random.uniform(0.5, 5.0),
            "relevance": relevance,
            "relevance_explanation": f"This answer is {relevance.lower()} to the question.",
            "prompt_tokens": random.randint(50, 200),
            "completion_tokens": random.randint(50, 300),
            "total_tokens": random.randint(100, 500),
            "eval_prompt_tokens": random.randint(50, 150),
            "eval_completion_tokens": random.randint(20, 100),
            "eval_total_tokens": random.randint(70, 250),
            "openai_cost": openai_cost,
        }

        save_conversation(conversation_id, question, answer_data, current_time)
        print(
            f"Saved live conversation: ID={conversation_id}, Time={current_time}"
        )

        if random.random() < 0.7:
            feedback = 1 if random.random() < 0.8 else -1
            save_feedback(conversation_id, feedback, current_time)
            print(
                f"Saved feedback for live conversation {conversation_id}: {'Positive' if feedback > 0 else 'Negative'}"
            )

        conversation_count += 1
        if conversation_count % 10 == 0:
            print(f"Generated {conversation_count} live conversations so far...")

        time.sleep(1)


if __name__ == "__main__":
    print(f"Script started at {datetime.now(tz)}")
    end_time = datetime.now(tz)
    start_time = end_time - timedelta(hours=6)
    print(f"Generating historical data from {start_time} to {end_time}")
    generate_synthetic_data(start_time, end_time)
    print("Historical data generation complete.")

    print("Starting live data generation... Press Ctrl+C to stop.")
    try:
        generate_live_data()
    except KeyboardInterrupt:
        print(f"Live data generation stopped at {datetime.now(tz)}.")
    finally:
        print(f"Script ended at {datetime.now(tz)}")
