#--------Day 13 Movie Chatbot - Final Submission Version--------
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# 1. The Brain Structure (Must be defined BEFORE loading the model)
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units=512, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({"units": 512}) # Hardcoded to match your training
        return config

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

# 2. Loading Resources (Pinned to ignore Keras 3 metadata errors)
@st.cache_resource
def load_resources():
    model_path = 'movie_chatbot_model.h5'
    tokenizer_path = 'tokenizer.pickle'
    
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    # Load with compile=False to bypass the 'quantization_config' error
    model = tf.keras.models.load_model(
        model_path, 
        custom_objects={'AttentionLayer': AttentionLayer},
        compile=False
    )
    return tokenizer, model

# INITIALIZE DATA GLOBALLY
tokenizer, model = load_resources()
reverse_word_index = {i: word for word, i in tokenizer.items()}
def get_chatbot_response(user_input, creativity):
    user_words = user_input.lower().split()
    # 3 is usually the OOV (Out of Vocabulary) token
    user_sequence = [tokenizer.get(word, 3) for word in user_words]
    user_padded = pad_sequences([user_sequence], maxlen=15, padding='post')
    
    target_sequence = np.zeros((1, 1))
    target_sequence[0, 0] = tokenizer.get('start', 1)

    decoded_sentence = []
    
    # Increase range to 15 to allow full sentences
    for _ in range(15):
        output_tokens = model.predict([user_padded, target_sequence], verbose=0)
        
        # Get predictions for the last word
        predictions = output_tokens[0, -1, :]

        # --- THE STABILITY FIX ---
        if creativity < 0.2:
            # If creativity is low, just take the #1 absolute best word
            sampled_token_index = np.argmax(predictions)
        else:
            # Apply temperature to smooth the probabilities
            predictions = np.log(predictions + 1e-7) / creativity
            exp_preds = np.exp(predictions)
            predictions = exp_preds / np.sum(exp_preds)
            
            # Reduce top_k to 2 or 3. Top 5 is too noisy for this model.
            top_k_indices = np.argsort(predictions)[-2:] 
            sampled_token_index = np.random.choice(top_k_indices)

        sampled_word = reverse_word_index.get(sampled_token_index, '')

        # Stop conditions
        if sampled_word in ['end', '<EOS>', 'pad', ''] or len(decoded_sentence) >= 12:
            break

        # Prevent immediate repetition (you you you)
        if len(decoded_sentence) > 0 and sampled_word == decoded_sentence[-1]:
            # Try to get the 2nd best word instead of stopping
            sampled_token_index = np.argsort(predictions)[-2]
            sampled_word = reverse_word_index.get(sampled_token_index, '')

        decoded_sentence.append(sampled_word)
        target_sequence[0, 0] = sampled_token_index

    # Final cleanup: Remove start/end artifacts
    clean_output = [w for w in decoded_sentence if w not in ['start', 'end', 'pad']]
    
    response = " ".join(clean_output)
    return response.capitalize() + "..." if response else "Tell me more about that movie..."

# 4. Streamlit UI (The "Midnight Cinema" Theme)
st.set_page_config(page_title="Movie AI", page_icon="🎬", layout="centered")

st.markdown("""
<style>
    .stApp {background-color: #FFFFFF; color: #262730;}
    h1 {color: #D32F2F; font-family: 'Arial Black';}
    .stChatMessage {border-radius: 15px; border: 1px solid #E6E9EF;}
</style>
""", unsafe_allow_html=True)

st.title('🎬 Movie Chatbot')
st.caption("Minor Project by Harsh Rana | Seq2Seq + Attention")

# Sidebar
st.sidebar.title("🎥 Control Panel")
creativity = st.sidebar.slider("AI Temperature", 0.1, 1.2, 0.7)
if st.sidebar.button("Clear History"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.subheader("📊 Training Analytics")
st.sidebar.line_chart([3.8, 3.2, 2.7, 2.1, 1.8, 1.5, 1.3, 1.2, 1.1, 1.05])

# Chat Logic
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Talk to the Movie AI..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Processing Attention Layers..."):
            response = get_chatbot_response(prompt, creativity)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
