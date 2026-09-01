"""
CODSOFT Internship - Task 3
"""

import re
import random
from datetime import datetime

RULES = [
    (
        [r"\bhi\b", r"\bhello\b", r"\bhey\b"],
        ["Hello! How can I help you today?", "Hi there! What's on your mind?"],
    ),
    (
        [r"how are you"],
        ["I'm just a program, but I'm running smoothly! How about you?"],
    ),
    (
        [r"what('?s| is) your name", r"who are you"],
        ["I'm a simple rule-based chatbot built for the CODSOFT internship!"],
    ),
    (
        [r"\btime\b"],
        [f"The current time is {datetime.now().strftime('%H:%M:%S')}."],
    ),
    (
        [r"\bdate\b|today('?s)? date"],
        [f"Today's date is {datetime.now().strftime('%Y-%m-%d')}."],
    ),
    (
        [r"your (creator|developer|maker)"],
        ["I was created as part of a CODSOFT internship project."],
    ),
    (
        [r"(thank you|thanks)"],
        ["You're welcome!", "Happy to help!"],
    ),
    (
        [r"\bjoke\b"],
        [
            "Why do programmers prefer dark mode? Because light attracts bugs!",
            "Why did the developer go broke? Because they used up all their cache.",
        ],
    ),
    (
        [r"\bweather\b"],
        ["I can't check live weather right now, but I hope it's nice where you are!"],
    ),
    (
        [r"(bye|goodbye|exit|quit)"],
        ["Goodbye! Have a great day!"],
    ),
    (
        [r"(help|what can you do)"],
        ["I can chat about basic topics like greetings, time, date, and jokes. "
         "Try asking me 'what's the time?' or 'tell me a joke'."],
    ),
]

FALLBACK_RESPONSES = [
    "I'm not sure I understand. Could you rephrase that?",
    "Sorry, I don't have an answer for that yet.",
    "Interesting! Could you tell me more, or ask something else?",
]

EXIT_PATTERNS = [r"\bbye\b", r"\bgoodbye\b", r"\bexit\b", r"\bquit\b"]


def get_response(user_input: str) -> str:
    text = user_input.lower().strip()
    for patterns, responses in RULES:
        for pattern in patterns:
            if re.search(pattern, text):
                return random.choice(responses)
    return random.choice(FALLBACK_RESPONSES)


def is_exit(user_input: str) -> bool:
    text = user_input.lower().strip()
    return any(re.search(p, text) for p in EXIT_PATTERNS)


def chat():
    print("Chatbot: Hi! I'm your rule-based assistant. Type 'bye' to exit.\n")
    while True:
        user_input = input("You: ")
        if is_exit(user_input):
            print("Chatbot:", get_response(user_input))
            break
        print("Chatbot:", get_response(user_input))


if __name__ == "__main__":
    chat()