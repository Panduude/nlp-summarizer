from transformers import pipeline

# Initialize summarizer for CPU
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",  # Lightweight model
    device=-1  # Force CPU
)

def summarize(text, max_length=150):
    return summarizer(
        text,
        max_length=max_length,
        min_length=30,
        do_sample=False,  # Disable random sampling
        truncation=True   # Automatically truncate long inputs
    )[0]['summary_text']

if __name__ == "__main__":
    input_text = "Your long text here..."  # Replace with actual text
    print(summarize(input_text, max_length=200))