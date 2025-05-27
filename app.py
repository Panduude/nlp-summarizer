import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="Text Summarizer",
    page_icon="📝",
    layout="wide"
)

st.title("📝 Text Summarizer")

# Load the summarization model
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

summarizer = load_summarizer()

# Text input
text = st.text_area("Enter text to summarize", height=200)

max_length = st.slider("Max summary length", 50, 500, 150)
min_length = st.slider("Min summary length", 10, 100, 30)

if st.button("Summarize"):
    if text.strip():
        with st.spinner("Summarizing..."):
            # Handle long text by chunking if needed
            if len(text.split()) > 800:
                chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
                summary = ""
                for chunk in chunks:
                    result = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
                    summary += result[0]['summary_text'] + " "
            else:
                result = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
                summary = result[0]['summary_text']
            st.success(summary)
    else:
        st.warning("Please enter some text.")
