# 🗽 NYC AI Assistant – Taxi Fare Predictor & City Guide

This project is a multimodal AI assistant that predicts NYC taxi fares and answers questions about city attractions and events. It combines traditional machine learning (XGBoost), retrieval-augmented generation (RAG), and large language models (LLMs) to create an intelligent, speech-enabled assistant deployed on Google Cloud.

---

## 🚀 Features

### 🎯 Fare Prediction (XGBoost)
- Predicts NYC taxi fares based on pickup/dropoff locations, time, and passenger count
- Trained using **XGBoost regression** with custom feature engineering
- Achieves **RMSE of 3.72**
- Served using **Vertex AI Prediction Endpoint**

### 📚 NYC Attraction & Event Search (RAG)
- Answers open-ended questions about landmarks, museums, and local events
- Uses **Vertex AI RAG Engine** with `text-embedding-004` for semantic search
- Powered by **LLaMA 3.1-70B-Instruct** for grounded answer generation

### 🤖 Intelligent Agent Routing (Gemini + LangGraph)
- Uses **Gemini 1.5 Flash** as a reasoning agent via LangChain
- Routes user queries to the correct tool using a **LangGraph workflow**
- Final responses are formatted by Gemini and returned in text or speech

### 🗣️ Multimodal Interaction
- Accepts both **speech and text input**
- Converts speech using **Google Speech-to-Text**
- Responds with synthesized voice using **Google Text-to-Speech**

---

## 📦 Tech Stack

| Category        | Technology                                                                 |
|----------------|------------------------------------------------------------------------------|
| Machine Learning | XGBoost (regression), custom feature engineering                          |
| LLMs             | Gemini 1.5 Flash, LLaMA 3.1-70B-Instruct (Vertex AI Model Garden)         |
| RAG              | Vertex AI RAG Engine + `text-embedding-004`                               |
| Agent Orchestration | LangChain + LangGraph                                                  |
| Speech APIs      | Google Speech-to-Text, Text-to-Speech                                     |
| Backend          | Flask REST API                                                            |
| NLP              | Named Entity Recognition for location parsing                             |
| Location Data    | Google Maps API                                                           |
| Hosting/Infra    | Vertex AI, Cloud Storage, Google Cloud Platform                           |

---

## 🧠 How It Works


```text
        [Text or Speech Input]
                   ↓
        Gemini Agent (LangGraph)
                   ↓
   ┌─────────────────────────────┐
   │   Tool Selection by Agent   │
   └─────────────────────────────┘
           ↓             ↓
   ┌────────────────┐  ┌────────────────────┐
   │ fare_prediction│  │  nyc_attractions   │
   │     _tool      │  │       _tool        │
   └────────────────┘  └────────────────────┘
           ↓             ↓
   Vertex AI Prediction   Vertex AI RAG Retrieval
           ↓             ↓
   └──────────── Final Response ─────────────┘
                   ↓
          Text Output / Synthesized Speech

