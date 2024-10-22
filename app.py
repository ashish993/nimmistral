import os
import streamlit as st
import openai
import time
from mistralai import Mistral

# Initialize NVIDIA NIM client

nim_client = openai.OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=nim_api_key
)

# Initialize Mistral client

mistral_client = Mistral(api_key=mistral_api_key)

# Models
nim_model = "mistralai/mistral-7b-instruct-v0.3"
mistral_model = "mistral-large-latest"

# Function to get response from NVIDIA NIM
def get_nim_response(prompt):
    start_time = time.time()
    response = nim_client.chat.completions.create(
        model=nim_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        stream=False
    )
    end_time = time.time()
    execution_time = end_time - start_time
    response_text = response.choices[0].message.content.strip()
    tokens = len(response_text.split())
    return response_text, execution_time, tokens

# Function to get response from Mistral AI
def get_mistral_response(prompt):
    start_time = time.time()
    chat_response = mistral_client.chat.complete(
        model=mistral_model,
        messages=[{"role": "user", "content": prompt}],
    )
    end_time = time.time()
    execution_time = end_time - start_time
    response_text = chat_response.choices[0].message.content
    tokens = len(response_text.split())
    return response_text, execution_time, tokens

# Streamlit UI
st.title("NVIDIA NIM and Mistral AI Chatbot Comparison :robot_face:")

# Session state to store the conversation
if "messages" not in st.session_state:
    st.session_state.messages = []
if "metrics" not in st.session_state:
    st.session_state.metrics = []

# User input
if prompt := st.chat_input("You:"):
    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get responses from both APIs
    nim_response, nim_time, nim_tokens = get_nim_response(prompt)
    mistral_response, mistral_time, mistral_tokens = get_mistral_response(prompt)

    # Store assistant responses and metrics
    st.session_state.messages.append({"role": "assistant", "content": ""})  # Placeholder for responses
    st.session_state.metrics.append({
        'nim_execution_time': nim_time,
        'nim_tokens': nim_tokens,
        'nim_tokens_per_second': nim_tokens / nim_time if nim_time > 0 else float('inf'),
        'mistral_execution_time': mistral_time,
        'mistral_tokens': mistral_tokens,
        'mistral_tokens_per_second': mistral_tokens / mistral_time if mistral_time > 0 else float('inf'),
    })

# Display conversation history
for i, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    elif message["role"] == "assistant":
        metrics = st.session_state.metrics[i // 2]  # Get metrics for the corresponding assistant response
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### NVIDIA NIM Response")
            st.markdown(nim_response)
            st.info(
                f"""**Duration:** :green[{metrics['nim_execution_time']:.2f} secs]  
                    **Tokens:** :green[{metrics['nim_tokens']} tokens]  
                    **Eval rate:** :green[{metrics['nim_tokens_per_second']:.2f} tokens/s]"""
            )

        with col2:
            st.markdown("### Mistral Response")
            st.markdown(mistral_response)
            st.info(
                f"""**Duration:** :green[{metrics['mistral_execution_time']:.2f} secs]  
                    **Tokens:** :green[{metrics['mistral_tokens']} tokens]  
                    **Eval rate:** :green[{metrics['mistral_tokens_per_second']:.2f} tokens/s]"""
            )
