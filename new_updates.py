import streamlit as st
from dotenv import load_dotenv
import os
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import google.generativeai as genai
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load environment variables
load_dotenv()

# Configure Google Generative AI (assuming gemini-pro setup)
genai.configure(api_key=os.getenv("AIzaSyCYOfsGu4s6Qd304XNvt1pRFpaQFGo15HY"))  

# Initialize the summarization pipeline
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

prompt = """You are an expert video summarizer. You will be taking the transcript text 
and summarizing the entire video and providing the important summary in points within 250 words.
Please provide the summary of the text given here: """

# Function to extract transcript details from YouTube video
def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("=")[-1]
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([i["text"] for i in transcript_text])
        return transcript
    except Exception as e:
        raise e

# Function to generate summary using Google Gemini
def generate_gemini_content(transcript_text, prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt + transcript_text)
    return response.text

# Function to split text into chunks of a given maximum length
def split_text_into_chunks(text, max_length=1024):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 <= max_length:
            current_chunk.append(word)
            current_length += len(word) + 1
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Function to generate summary using Hugging Face transformer model
def generate_transformer_summary(transcript_text):
    chunks = split_text_into_chunks(transcript_text)
    summaries = []

    for chunk in chunks:
        summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])

    return " ".join(summaries)

# Function to generate summary using LSTM model
def generate_lstm_summary(transcript_text, max_length=50):
    # Tokenize the text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([transcript_text])
    sequence = tokenizer.texts_to_sequences([transcript_text])[0]

    # Pad the sequence
    sequence = pad_sequences([sequence], maxlen=max_length, padding='post')

    # Create LSTM model
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_length))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Dummy training for the sake of example (train on dummy data)
    # Note: In a real scenario, you would train the model on a proper dataset
    dummy_data = np.random.randint(1, len(tokenizer.word_index) + 1, size=(1000, max_length))
    dummy_labels = np.random.randint(2, size=(1000, 1))
    model.fit(dummy_data, dummy_labels, epochs=1, batch_size=32, verbose=0)

    # Generate summary (for demonstration purposes, we use the input itself)
    summary = " ".join([tokenizer.index_word.get(idx, '') for idx in sequence[0]])

    return summary

# Streamlit app
st.title("YouTube Transcript to Detailed Notes Converter")
youtube_link = st.text_input("Enter YouTube Video Link:")

if youtube_link:
    video_id = youtube_link.split("=")[-1]
    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)
    
if st.button("Get Detailed Notes"):
    transcript_text = extract_transcript_details(youtube_link)
    
    if transcript_text:
        st.markdown("## Google Gemini Summary:")
        gemini_summary = generate_gemini_content(transcript_text, prompt)
        st.write(gemini_summary)

        st.markdown("## Transformer Model Summary:")
        transformer_summary = generate_transformer_summary(transcript_text)
        st.write(transformer_summary)

        st.markdown("## LSTM Model Summary:")
        lstm_summary = generate_lstm_summary(transcript_text)
        st.write(lstm_summary)
