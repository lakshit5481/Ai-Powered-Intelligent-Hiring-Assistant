import google.generativeai as genai

# Replace with your Gemini API Key
genai.configure(api_key="api key")

model = genai.GenerativeModel("gemini-2.5-flash")


def generate_feedback(category,
                      similarity_score,
                      resume_skills,
                      missing_skills):

    prompt = f"""
You are an expert HR Recruiter.

Candidate Category:
{category}

Resume Match:
{similarity_score}%

Resume Skills:
{', '.join(sorted(resume_skills))}

Missing Skills:
{', '.join(sorted(missing_skills))}

Generate:

1. Overall Evaluation

2. Strengths

3. Weaknesses

4. Resume Improvements

5. Courses to Learn

6. Projects to Build

Keep the answer professional.
"""

    response = model.generate_content(prompt)

    return response.text