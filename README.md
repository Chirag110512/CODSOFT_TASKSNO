# CODSOFT_TASKSNO

CODSOFT Internship Tasks — Recommendation System | Face Detection | Chatbot (Python, scikit-learn, OpenCV)

This repository contains the tasks completed as part of my **CODSOFT Internship** (AI / Data Science track).

## 📌 Task 1: Movie Recommendation System
**File:** `task1_recommendation_system_FINAL.py`

Recommends movies using two approaches:
- **Content-Based Filtering** — recommends movies with similar genres using TF-IDF + cosine similarity
- **Collaborative Filtering** — recommends movies liked by users with similar taste

**Run:**
```bash
python task1_recommendation_system_FINAL.py
```

## 📌 Task 2: Face Detection & Recognition
**File:** `task2_face_detection.py` | **Sample image:** `sample.jpg`

Detects faces in images/video using OpenCV's Haar Cascade classifier, with optional face recognition using LBPH.

**Run:**
```bash
python task2_face_detection.py --image sample.jpg
python task2_face_detection.py --webcam
```
# Task 3: Rule-Based Chatbot

A simple chatbot that responds to user input using **regex-based pattern matching** and predefined rules — a foundational look at how conversation flow and basic NLP work before moving to statistical/neural approaches.

## 📌 What it does
- Matches user input against a set of predefined regex patterns (greetings, questions about time/date, jokes, thanks, help, exit, etc.)
- Returns an appropriate response for matched patterns, with slight randomness for variety
- Falls back to a generic response when no rule matches
- Runs as an interactive terminal chat loop until the user types an exit word (bye/exit/quit)

## 🛠️ Tech Stack
- Python
- `re` (regex) — no external NLP libraries required

## 📂 Files
- `task3_chatbot.py` — main chatbot script

## ▶️ How to Run
```bash
python task3_chatbot.py
```

Then just type messages when prompted. Example: 

Chatbot: Hi! I'm your rule-based assistant. Type 'bye' to exit.

You: hi
Chatbot: Hello! How can I help you today?

You: tell me a joke
Chatbot: Why do programmers prefer dark mode? Because light attracts bugs!

You: bye
Chatbot: Goodbye! Have a great day!

## 🛠️ Tech Stack
Python, pandas, scikit-learn, OpenCV

## 🙋 About
Completed by [Chirag Kabra] as part of the CODSOFT Internship Program.
