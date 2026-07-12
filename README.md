# 🤖 AI Resume Screening System

An end-to-end AI system that evaluates candidate resumes against job requirements using Machine Learning and Generative AI.

---

## Features

- 📄 Upload Resume PDF
- 🤖 Resume Classification
- 📊 Confidence Score
- 🏆 Top 3 Predictions
- 📈 Resume ↔ Job Description Matching
- ⭐ ATS Resume Score
- ✅ Skill Gap Detection
- 💡 AI Feedback using Gemini AI

---

## Technologies Used

- Python
- Scikit-Learn
- TF-IDF
- Logistic Regression
- Streamlit
- Google Gemini AI
- PDFPlumber
- Pandas
- Joblib

---
## Project Architecture

                User
                  │
                  ▼
             Streamlit UI
                  │
                  ▼
              LangGraph
                  │
                  ▼
               Router
                  │
      ┌───────────┼───────────┐
      ▼           ▼           ▼
 Calculator   Wikipedia     PDF RAG
      │           │           │
      └───────────┼───────────┘
                  ▼
             Evaluator
                  │
                  ▼
               Memory
                  │
                  ▼
               Response

---
## Project Workflow

Resume PDF
↓
Text Extraction
↓
Text Preprocessing
↓
TF-IDF Vectorization
↓
Resume Classification
↓
Resume-JD Similarity
↓
Skill Gap Detection
↓
ATS Score
↓
Gemini AI Feedback

---

## Folder Structure

```
Final_project/

├── app/
│ ├── app.py
│ ├── train_model.py
│ └── utils/
│ ├── preprocessing.py
│ ├── pdf_parser.py
│ ├── similarity.py
│ ├── skills.py
│ └── feedback.py
│
├── data/
│ └── raw/
│ └── resumes_dataset.jsonl
│
├── models/
│ ├── resume_classifier.pkl
│ ├── tfidf_vectorizer.pkl
│ └── label_encoder.pkl
│
├── README.md
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

Run Model Training

```bash
python app/train_model.py
```

Run Application

```bash
streamlit run app/app.py
```

---

# Future Enhancements

The current version is a Minimum Viable Product (MVP). Future improvements include:

### 1. Retrieval-Augmented Generation (RAG)
- Build a knowledge base using resume writing guides, interview preparation resources, and career documents.
- Store embeddings in a FAISS Vector Database.
- Retrieve relevant information before generating AI feedback.

### 2. Conversational AI Career Assistant
- Add a chatbot that allows candidates to ask questions like:
  - "How can I improve my resume?"
  - "What skills should I learn?"
  - "Suggest projects for my career path."

### 3. Deep Learning Resume Classification
- Replace TF-IDF + Logistic Regression with transformer-based models such as:
  - BERT
  - RoBERTa
  - Sentence Transformers

### 4. Resume Ranking System
- Compare multiple resumes against a single job description.
- Rank candidates based on ATS score and semantic similarity.

### 5. Explainable AI
- Highlight keywords that contributed to the prediction.
- Explain why the resume received a particular ATS score.

### 6. Cloud Deployment
- FastAPI Backend
- Docker
- Render / AWS / Azure Deployment

### 7. Authentication
- Recruiter Login
- Candidate Login
- Resume History
- Dashboard Analytics

### 8. Interview Question Generator
- Generate technical interview questions based on:
  - Resume
  - Job Description
  - Missing Skills

### 9. Resume Improvement Suggestions
- ATS Keyword Optimization
- Grammar Correction
- Professional Summary Enhancement
- Project Description Enhancement

### 10. Multi-language Resume Support
- Parse and evaluate resumes written in multiple languages.



## Author

Lakshit Soni

B.Tech CSE

AI & Machine Learning Enthusiast
