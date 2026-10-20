# 🚗 OBD AI — Multi-Agent Vehicle Diagnostic Assistant

**An AI-powered vehicle diagnostic assistant that combines live OBD-II data, CrewAI multi-agent systems, and Retrieval-Augmented Generation (RAG) to turn raw vehicle readings into understandable diagnostic insights.**

## 💡 About the Project

Traditional OBD-II scanners primarily provide diagnostic trouble codes (DTCs) and sensor readings, leaving vehicle owners to interpret the results themselves.

**OBD AI goes a step further.** It uses specialized AI agents to analyze available vehicle data, research potential causes, retrieve relevant technical knowledge, and generate a structured diagnostic report with possible causes and recommended next checks.

## ✨ Key Features

- **Live OBD-II Data:** Collects available vehicle diagnostic readings through a compatible ELM327 adapter.
- **CrewAI Multi-Agent System:** Separates vehicle monitoring, diagnostic research, and final diagnosis into specialized tasks.
- **RAG-Based Knowledge Retrieval:** Retrieves relevant information from vehicle documents to provide technical context beyond the LLM's built-in knowledge.
- **AI-Powered Diagnostic Research:** Uses configured research tools to investigate potential causes and support troubleshooting.
- **Interactive Dashboard:** Streamlit interface for viewing available readings and reviewing diagnostic reports.

## 🧠 How It Works

`OBD-II Data → Vehicle Monitoring → Diagnostic Research + RAG → Final AI Diagnosis → Streamlit Dashboard`

The system combines observed vehicle data with retrieved technical information to generate explanations that are more useful than raw fault codes alone. AI-generated findings remain hypotheses until verified through appropriate diagnostic procedures.

## 🛠️ Tech Stack

**Python · CrewAI · Generative AI / LLMs · RAG · Streamlit · OBD-II · Ollama / Configured LLM APIs**

## 🚀 Current Development

**This project is actively under development.** I am continuing to improve the diagnostic reasoning workflow, enhance knowledge retrieval, and expand the range of diagnostic features and vehicle parameters supported.

My focus is on making the system more reliable, evidence-driven, and useful for real-world vehicle troubleshooting.

## 👨‍💻 Author

**Amith R** — B.Tech Mechanical Engineering

Interested in AI/ML Engineering, Generative AI, multi-agent systems, RAG, and practical AI applications in automotive diagnostics.
