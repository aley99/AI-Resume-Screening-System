
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("AI Resume Screening System")

job_desc = st.text_area("Paste Job Description")
resume = st.text_area("Paste Resume")

if st.button("Check Match"):
    if job_desc and resume:
        docs = [job_desc, resume]
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(docs)
        score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        st.success(f"Resume Match Score: {round(score*100, 2)}%")
    else:
        st.warning("Please enter both fields")
