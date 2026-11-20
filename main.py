import streamlit as st
import os
from langchain_huggingface import HuggingFaceEndpoint

# Token-a code la ezhudha koodadhu.
# .env file la podu illa Streamlit Secrets la podu
API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

st.set_page_config(
    page_title="Large Language Model - ChatBot",
    page_icon="💻",
    layout="centered"
)

st.title("Large Language Model 💻🔗 Text-Generation")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_prompt = st.chat_input("Ask Large Language Model")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Mistral Model - Hugging Face
    LargeLanguageModel = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        max_length=200,
        temperature=0.3,
        token=API_TOKEN
    )

    response = LargeLanguageModel.invoke(user_prompt)

    st.session_state.chat_history.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)
