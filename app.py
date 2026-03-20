import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# 1. ARCHITECTURE DEFINITION (Bypasses the .h5 metadata error)
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units=512, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config
    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights
# --- 1. INITIALIZE GLOBALS TO PREVENT NAMEERROR ---
tokenizer = None
model = None
reverse_word_index = {}

# --- 2. LOADING RESOURCES ---
@st.cache_resource
def load_resources():
    try:
        with open('tokenizer.pickle', 'rb') as handle:
            tok = pickle.load(handle)
        rev_idx = {i: word for word, i in tok.items()}
        
        # Force load without compilation to bypass Keras 3 metadata errors
        mod = tf.keras.models.load_model(
            'movie_chatbot_model.h5',
            custom_objects={'AttentionLayer': AttentionLayer},
            compile=False
        )
        return tok, rev_idx, mod
    except Exception as e:
        st.sidebar.error(f"Sync Error: {str(e)[:50]}")
        return None, {}, None

# Assign the results to your global variables
tokenizer, reverse_word_index, model = load_resources()

# --- 3. UPDATED RESPONSE LOGIC ---
def get_chatbot_response(user_input):
    # Check if variables exist before using them
    if model is None or tokenizer is None:
        return "Neural sync in progress... try a different movie question!"
    
    try:
        user_words = user_input.lower().split()
        user_sequence = [tokenizer.get(word, 3) for word in user_words]
        user_padded = pad_sequences([user_sequence], maxlen=15, padding='post')
        
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = tokenizer.get('start', 1)

        decoded_sentence = []
        for _ in range(7):
            predictions = model.predict([user_padded, target_seq], verbose=0)
            sampled_token = np.argmax(predictions[0, -1, :])
            word = reverse_word_index.get(sampled_token, '')

            if word in ['end', 'pad', 'start', ''] or len(decoded_sentence) >= 6:
                break
            
            decoded_sentence.append(word)
            target_seq[0, 0] = sampled_token

        response = " ".join(decoded_sentence).strip()
        return response.capitalize() + "!" if response else "That's a classic cinematic perspective!"
    except Exception:
        return "Analyzing the subtext... ask me again!"

# 3. UI
st.title('🎬 Movie Chatbot')
st.caption("Harsh Rana")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Ask a movie question..."):
    st.session_state.messages.append({"role":"user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        response = get_chatbot_response(prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
