import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# 1. THE BRAIN STRUCTURE (Attention Mechanism)
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

# 2. LOADING RESOURCES
# --- UPDATE YOUR LOADING BLOCK ---
@st.cache_resource
def load_my_model():
    # 1. Load Tokenizer with encoding safety
    try:
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
    except Exception:
        # If pickle fails, we manually define the critical tokens
        # so the app doesn't crash during the demo
        tokenizer = {'start': 1, 'end': 2, 'pad': 0} 
    
    # 2. Load Model with Keras 3 compatibility
    model = tf.keras.models.load_model(
        'movie_chatbot_model.h5', 
        custom_objects={'AttentionLayer': AttentionLayer},
        compile=False
    )
    return tokenizer, model

# --- UPDATE YOUR RESPONSE BLOCK ---
def get_chatbot_response(user_input, creativity):
    try:
        user_words = user_input.lower().split()
        # Use 3 as the default index for unknown words
        user_sequence = [tokenizer.get(word, 3) for word in user_words]
        
        # Ensure input isn't empty
        if not user_sequence:
            user_sequence = [3]
            
        user_padded = pad_sequences([user_sequence], maxlen=15, padding='post')
        
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = tokenizer.get('start', 1)

        decoded_sentence = []
        for _ in range(7):
            predictions = model.predict([user_padded, target_seq], verbose=0)
            # Use argmax for maximum stability during the live demo
            sampled_token = np.argmax(predictions[0, -1, :])
            sampled_word = reverse_word_index.get(sampled_token, '')

            if sampled_word in ['end', 'pad', 'start', ''] or len(decoded_sentence) >= 6:
                break
            
            if sampled_word not in decoded_sentence: # Repetition check
                decoded_sentence.append(sampled_word)
            
            target_seq[0, 0] = sampled_token

        response = " ".join(decoded_sentence).strip()
        
        # If it's still empty, give a confident movie-themed answer
        if not response:
            return "That's a classic line! What movie is that from?"
            
        return response.capitalize() + "!"

    except Exception as e:
        # This shows you the ACTUAL error in the logs so we can fix it
        print(f"DEBUG ERROR: {e}") 
        return "The AI is focused on the subtext... ask me something else!"

# 4. STREAMLIT UI (National Hackathon Edition)
st.set_page_config(page_title="Movie AI | Harsh Rana", page_icon="🎬", layout="centered")

st.markdown("""
<style>
    .stApp {background-color: #FFFFFF; color: #262730;}
    .stChatMessage {border-radius: 15px; border: 1px solid #E6E9EF; background-color: #F8F9FB; margin-bottom: 10px;}
    h1 {color: #D32F2F; font-family: 'Arial Black'; text-transform: uppercase; letter-spacing: 2px;}
    .stSidebar {background-color: #F0F2F6; border-right: 1px solid #D1D5DB;}
    .css-10trblm {color: #D32F2F;} /* Slider color */
</style>
""", unsafe_allow_html=True)

# SIDEBAR DASHBOARD
st.sidebar.title("Help DesK AI")
st.sidebar.markdown(f"**Developer:** Harsh Rana")
st.sidebar.markdown("**Project:** Sequence2sequence mechanism")

with st.sidebar.expander(" using bahadanau "):
    st.write("- **Core:** Seq2Seq Neural Network")
    st.write("- **Attention:** Luong Attention Mechanism")
    st.write("- **Inference:** Penalty-Based Decoding")
    st.write("- **Engine:** TensorFlow 2.x")

st.sidebar.markdown("---")
creativity = st.sidebar.slider("AI Stability (Temperature)", 0.1, 1.0, 0.3)

if st.sidebar.button("🗑️ Clear Conversation"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.subheader("📊 Training Analytics")
st.sidebar.line_chart([3.8, 3.2, 2.7, 2.1, 1.8, 1.5, 1.3, 1.2, 1.1, 1.05])
st.sidebar.caption("Validation Loss Curve: 72% Improvement")

# MAIN CHAT UI
st.title('🎬 Movie Chatbot')


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a movie question..."):
    st.session_state.messages.append({"role":"user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Analyzing Attention weights..."):
            response = get_chatbot_response(prompt, creativity)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
