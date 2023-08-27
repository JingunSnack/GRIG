import os
from dotenv import load_dotenv
import openai
import streamlit as st

from grig import summary, image, diary

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]

st.markdown("# GRIG: Your Personal Life Logger!")

st.markdown(
    """
GRIG is an innovative platform designed to add a new dimension to the way you document your life.
We take your voice recordings, transcribe them, analyze the content, and provide you with both
a unique DALL-E generated image and a casual daily note that captures the essence of your
experiences and thoughts.
"""
)


uploaded_file = st.file_uploader("Choose a recording file")
if uploaded_file is not None:
    content = ""
    with st.spinner("Transcribing your recording..."):
        content = openai.Audio.transcribe("whisper-1", uploaded_file)

    st.success("Transcription done!")

    result = ""
    if content and content.get("text"):
        with st.spinner("Analyzing the transcript..."):
            result = summary.generate(content.get("text"))
            st.code(result)
        st.success("Analysis done!")

    image_url = ""
    if result:
        with st.spinner("Creating an image..."):
            image_url = image.generate(result)
        st.success("Image ready!")

    daily_note = ""
    if result:
        with st.spinner("Generating a daily Note"):
            daily_note = diary.generate(result)
        st.success("Daily note ready!")

    if image_url:
        st.image(image_url)

    if daily_note:
        st.markdown(daily_note)

    if image_url and daily_note:
        st.balloons()
