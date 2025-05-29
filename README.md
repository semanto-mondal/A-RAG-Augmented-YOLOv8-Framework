# ☕ Coffee Leaf Disease Detection & Remedy Assistant 🌿

Check out the [arXiv preprint](https://arxiv.org/abs/2505.21544) for detailed methodology and results.

An intelligent, end-to-end AI system for detecting coffee leaf diseases and providing expert-level remedies through a conversational interface powered by YOLOv8 and a RAG-based LLM pipeline.

## 📌 Problem Statement

Timely and accurate identification of plant diseases is vital to prevent yield loss and excessive pesticide use. Traditional inspection methods are manual, error-prone, and not scalable. Even when diseases are detected, actionable, trustworthy remedies are often unavailable to farmers in real-time.

This project solves this by combining state-of-the-art object detection (YOLOv8) with a Retrieval-Augmented Generation (RAG) pipeline and Large Language Models (LLMs) to automatically detect diseases from images, retrieve relevant knowledge, and provide remedies and explanations through an interactive chatbot.

---

## 🚀 Features

- 🔍 **Leaf Disease Detection** using a fine-tuned YOLOv8 model
- 📚 **Knowledge-Rich Answers** using a FAISS-based vector store and domain-specific documentation
- 🤖 **RAG-powered Chatbot** using LLaMA 3 via Groq API for remedy generation and user interaction
- 🧠 **Contextual Chat History** managed by LangChain memory
- 📈 **Streamlit Web Interface** for user-friendly access
- 🔁 **Multi-turn Chat Support** with references to knowledge source

---

## 🧪 Experimental Setup

- **Model**: YOLOv8n (fine-tuned)
- **Resolution**: 640×640 (original: 2048×1024)
- **Training Epochs**: 100
- **System**: 16GB RAM, NVIDIA RTX 4060 (8GB), CUDA 11.8
- **Frameworks**: Ultralytics YOLO, LangChain, Hugging Face, Groq API
- **Vector Store**: FAISS with domain documents as the knowledge base

---

## 📊 Model Performance (YOLOv8)

| Disease Class | Precision | Recall | mAP50 | mAP50-95 |
|---------------|-----------|--------|--------|-----------|
| **Cercospora** | 0.546     | 0.630  | 0.575  | 0.329     |
| **Miner**      | 0.823     | 0.849  | 0.894  | 0.650     |
| **Phoma**      | 0.727     | 0.877  | 0.839  | 0.612     |
| **Rust**       | 0.561     | 0.316  | 0.415  | 0.223     |
| **Overall**    | 0.664     | 0.668  | 0.681  | 0.454     |

---
## 🧰 Tech Stack
- **YOLOv8** - Object Detection (Ultralytics)

- **Streamlit** - Web UI

- **Groq API** - Fast LLM access (LLaMA 3)

- **Hugging Face** - Embedding models

- **LangChain** - Conversational memory and RAG

- **FAISS** - Vector search for knowledge retrieval
  
---
## 📌 Future Work
- Upgrade to YOLOv10 or YOLOv11 for better detection accuracy

- Enrich the knowledge base with expert-curated agricultural content

- Expand the system to support other crops and diseases

- Improve multi-language support for non-English-speaking users


