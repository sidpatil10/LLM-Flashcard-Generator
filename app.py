import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    return tokenizer, model

tokenizer, model = load_model()

# Function to generate flashcards
def generate_flashcards(text, num_flashcards=5):
    prompt = f"""
You are a helpful assistant. Generate {num_flashcards} educational flashcards from the following text.
Each flashcard must be in Q&A format, numbered like this:

Q1: ...
A1: ...

Q2: ...
A2: ...

Text:
{text}
"""

    inputs = tokenizer(prompt.strip(), return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        num_beams=5,
        early_stopping=True
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

# Streamlit UI
st.title("📚 Flashcard Generator (Open-Source LLM)")

text_input = st.text_area("Paste educational content here:", height=300)
num_flashcards = st.slider("Number of flashcards", 3, 15, 5)

if st.button("Generate Flashcards"):
    if text_input.strip():
        with st.spinner("Generating..."):
            output = generate_flashcards(text_input, num_flashcards)
        st.success("Flashcards Generated:")
        st.markdown(output.replace("\n", "  \n"))
    else:
        st.warning("Please paste some text.")
