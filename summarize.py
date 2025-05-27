from transformers import pipeline

summarizer = pipeline("summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=-1
)

def summarize(text, max_length=150):
    return summarizer(
        text,
        max_length=max_length,
        min_length=30,
        do_sample=False,
        truncation=True
    )[0]['summary_text']

if __name__ == "__main__":
    input_text = "Your long text here..."
    print(summarize(input_text, max_length=200))