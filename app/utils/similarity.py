from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity(tfidf, resume_text, job_description):
    """
    Calculate cosine similarity between resume and job description.
    Returns a percentage score.
    """

    resume_vector = tfidf.transform([resume_text])
    jd_vector = tfidf.transform([job_description])

    similarity = cosine_similarity(
        resume_vector,
        jd_vector
    )[0][0]

    return round(similarity * 100, 2)