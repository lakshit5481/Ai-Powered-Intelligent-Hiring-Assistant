import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

from utils.preprocessing import clean_resume
from utils.pdf_parser import extract_text_from_pdf
from utils.similarity import calculate_similarity
from utils.skills import missing_skills
from utils.feedback import generate_feedback

# ============================================
# Paths
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "resume_classifier.pkl"
TFIDF_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"
LABEL_PATH = BASE_DIR / "models" / "label_encoder.pkl"

# ============================================
# Load Models
# ============================================

model = joblib.load(MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)
label_encoder = joblib.load(LABEL_PATH)

# ============================================
# Streamlit Page
# ============================================

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Resume Screening System")

st.caption(
    "Evaluate candidate resumes against job descriptions using Machine Learning and Generative AI."
)

# ============================================
# Sidebar
# ============================================

st.sidebar.title("📋 Project Features")

st.sidebar.markdown("""
### Implemented Modules

- 📄 Resume PDF Upload
- 🤖 Resume Classification
- 📊 Confidence Score
- 🏆 Top 3 Predictions
- 📈 Resume-JD Similarity
- ✅ Skill Gap Detection
- ⭐ ATS Resume Score
- 💡 Gemini AI Feedback

---

### Technologies

- Python
- Scikit-Learn
- TF-IDF
- Logistic Regression
- Streamlit
- Gemini AI
""")

# ============================================
# Input
# ============================================

uploaded_file = st.file_uploader(
    "📄 Upload Resume",
    type=["pdf"]
)

job_description = st.text_area(
    "📋 Paste Job Description",
    height=220
)

# ============================================
# Prediction
# ============================================

if uploaded_file:

    resume_text = extract_text_from_pdf(uploaded_file)

    cleaned_text = clean_resume(resume_text)

    with st.expander("📄 Resume Preview"):
        st.text(resume_text[:2000])

    vector = tfidf.transform([cleaned_text])

    prediction = model.predict(vector)[0]

    category = label_encoder.inverse_transform([prediction])[0]

    probabilities = model.predict_proba(vector)[0]

    confidence = probabilities.max() * 100

    st.success("Prediction Completed Successfully!")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🎯 Predicted Category")

        st.success(category)

    with col2:

        st.subheader("📊 Confidence")

        st.progress(min(int(confidence),100))

        st.write(f"### {confidence:.2f}%")

    # ============================================
    # Top 3 Predictions
    # ============================================

    st.subheader("🏆 Top 3 Predictions")

    top3 = probabilities.argsort()[-3:][::-1]

    df_results = pd.DataFrame({
        "Category":[label_encoder.inverse_transform([i])[0] for i in top3],
        "Confidence (%)":[round(probabilities[i]*100,2) for i in top3]
    })

    st.table(df_results)

    # ============================================
    # Resume vs JD
    # ============================================

    if job_description.strip():

        similarity = calculate_similarity(
            tfidf,
            cleaned_text,
            clean_resume(job_description)
        )

        st.subheader("📈 Resume Match Score")

        st.progress(min(int(similarity),100))

        st.write(f"## {similarity}% Match")

        resume_skills, jd_skills, missing = missing_skills(
            cleaned_text,
            job_description
        )

        ats_score = round(
            similarity * 0.7 +
            confidence * 0.3,
            2
        )

        st.subheader("⭐ ATS Resume Score")

        st.progress(min(int(ats_score),100))

        st.write(f"# {ats_score}/100")

        c1,c2,c3 = st.columns(3)

        with c1:
            st.subheader("✅ Resume Skills")

            if resume_skills:
                st.write(", ".join(sorted(resume_skills)))
            else:
                st.info("No skills detected.")

        with c2:
            st.subheader("📋 Required Skills")

            if jd_skills:
                st.write(", ".join(sorted(jd_skills)))
            else:
                st.info("No skills detected.")

        with c3:
            st.subheader("❌ Missing Skills")

            if missing:
                for skill in sorted(missing):
                    st.write(f"• {skill}")
            else:
                st.success("No Missing Skills")

        st.divider()

        st.subheader("🤖 AI Career Advisor")

        with st.spinner("Generating AI Feedback..."):

            feedback = generate_feedback(
                category,
                similarity,
                resume_skills,
                missing
            )

        st.markdown(feedback)

else:

    st.info("👆 Upload a PDF Resume to begin.")