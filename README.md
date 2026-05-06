# 🌍 Digital Trade Compliance AI Platform

An AI-powered hybrid framework for **pre-screening digital trade compliance records**, integrating carbon disclosure, rules-of-origin verification, and supply chain traceability.

---

## 🚀 Overview

Global trade is rapidly transitioning toward **digital, data-driven compliance systems**, driven by:

- Carbon border regulations (e.g., CBAM)
- Supply chain traceability requirements
- ESG and sustainability reporting frameworks

However, current verification processes remain:

- Manual  
- Fragmented  
- Difficult to scale  

---

### 💡 This platform solves that problem

It provides a **hybrid AI-driven compliance screening system** that:

- Detects inconsistencies in trade records  
- Identifies hidden risks using machine learning  
- Explains decisions transparently  
- Prioritises records for human review  

---

## 🧠 Key Features

### ✅ Hybrid AI Verification
- Rule-based validation (deterministic checks)
- Machine learning (anomaly detection + classification)
- Combined into a unified risk score

---

### ✅ ICC-Aligned Data Model
Aligned with:
- International Chamber of Commerce (ICC) Digital Trade Toolkit

Supports:
- Product data  
- Origin declarations  
- Carbon disclosures  
- Traceability metadata  
- Documentation quality  

---

### ✅ Explainable AI
- SHAP-based feature attribution  
- Human-readable explanations  
- Audit-ready decision trace  

---

### ✅ Workflow-Oriented Design
- Pre-screening, not full automation  
- Supports human-in-the-loop review  
- Enables prioritisation of high-risk records  

---

### ✅ Full-Stack Platform
- Backend API (FastAPI)
- AI engine (Python, Scikit-learn)
- Frontend UI (Streamlit)
- Database (PostgreSQL)
- Docker deployment support  

---

## 🏗️ System Architecture

```plaintext
User Interface (Streamlit)
        ↓
API Layer (FastAPI)
        ↓
Pipeline Engine
        ↓
[ Ingestion → ICC Mapping → Rules → ML → Hybrid → Explainability ]
        ↓
Database + Audit Logging

digital_trade_ai_upgrade/

├── src/                  # Core AI engine
│   ├── ingestion/        # Data loading
│   ├── translation/      # ICC mapping
│   ├── schema/           # Canonical data model
│   ├── rules/            # Rule engine
│   ├── ml/               # Machine learning models
│   ├── hybrid/           # Risk scoring
│   ├── explainability/   # SHAP explanations
│   ├── database/         # DB models and connection
│   ├── audit/            # Logging
│   └── api/              # FastAPI endpoints
│
├── app/                  # Streamlit UI
│   ├── pages/            # UI pages
│   ├── state/            # Session state
│   └── api_client.py     # Backend communication
│
├── config/               # Mapping + weights config
├── models/               # Trained ML models
├── data/                 # Synthetic & real datasets
├── backend/              # Docker setup
└── README.md

⚙️ Installation
1. Clone repository
git clone <repo-url>
cd digital_trade_ai_upgrade
2. Install dependencies
pip install -r requirements.txt
▶️ Running the System
🔹 Option 1: Local Development
Start backend API
uvicorn src.api.main:app --reload
Start UI
streamlit run app/streamlit_app.py
🔹 Option 2: Docker (Recommended)
cd backend
docker-compose up --build