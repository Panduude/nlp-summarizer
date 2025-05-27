import streamlit as st
from transformers import pipeline

st.title("Text Summarizer")

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

text = st.text_area("Enter text to summarize", height=200)
min_length = st.slider("Minimum summary length", 10, 100, 30)
max_length = st.slider("Maximum summary length", 50, 300, 130)

if st.button("Summarize"):
    if text:
        summary = summarizer(text, min_length=min_length, max_length=max_length, do_sample=False)
        st.write("**Summary:**")
        st.success(summary[0]['summary_text'])
    else:
        st.warning("Please enter some text.")
