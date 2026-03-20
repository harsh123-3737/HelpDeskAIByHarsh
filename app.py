import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# 1. ATTENTION LAYER (Must be exactly as trained)
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units=512, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({"units": self.units})
        return config

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

# 2. GLOBAL LOADING (This fixes the "Subtext" error)
@st.cache_resource
def load_resources():
    # Load the tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        tok = pickle.load(handle)
    
    rev_idx = {i: word for word, i in tok.items()}
    
    # CRITICAL: Use compile=False to bypass the Keras 3 metadata error
    try:
        mod = tf.keras.models.load_model(
            'movie_chatbot_model.h5', 
            custom_objects={'AttentionLayer': AttentionLayer},
            compile=False
        )
    except TypeError:
        # Emergency fallback for version mismatch
        import keras
        mod = keras.models.load_model(
            'movie_chatbot_model.h5',
            custom_objects={'AttentionLayer': AttentionLayer},
            compile=False
        )
        
    return tok, rev_idx, mod

# Initialize these globally
tokenizer, reverse_word_index, model = load_resources()

# 3. CHAT LOGIC
def get_chatbot_response(user_input):
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
            sampled_word = reverse_word_index.get(sampled_token, '')

            if sampled_word in ['end', 'pad', 'start', ''] or len(decoded_sentence) >= 6:
                break
            
            decoded_sentence.append(sampled_word)
            target_seq[0, 0] = sampled_token

        response = " ".join(decoded_sentence).strip()
        
        # If the neural network fails, give a smart cinematic response
        if not response:
            return "That's an interesting scene. Tell me more about it!"
            
        return response.capitalize() + "!"

    except Exception as e:
        # This will show the error in the Streamlit Sidebar for you to see
        return f"Neural Sync Error: {str(e)[:50]}"

# 4. STREAMLIT UI
st.title('🎬 Movie Chatbot')
st.caption("National Hackathon Edition | Harsh Rana")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Talk to the Movie AI..."):
    st.session_state.messages.append({"role":"user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        response = get_chatbot_response(prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
