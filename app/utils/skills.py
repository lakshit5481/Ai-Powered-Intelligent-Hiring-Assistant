import re

COMMON_SKILLS = [
    "python","java","c++","c","sql","mysql","postgresql","mongodb",
    "html","css","javascript","react","angular","node","django","flask",
    "fastapi","spring","hibernate",
    "docker","kubernetes","aws","azure","gcp",
    "tensorflow","pytorch","scikit-learn","pandas","numpy",
    "machine learning","deep learning","nlp",
    "git","github","linux","spark","hadoop",
    "power bi","tableau","excel"
]

def extract_skills(text):
    text = text.lower()

    found = set()

    for skill in COMMON_SKILLS:
        if re.search(r"\b" + re.escape(skill) + r"\b", text):
            found.add(skill)

    return found


def missing_skills(resume_text, jd_text):

    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    missing = jd_skills - resume_skills

    return resume_skills, jd_skills, missing