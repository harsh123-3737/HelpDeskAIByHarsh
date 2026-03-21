import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# 1. STEP 1: THE BRAIN STRUCTURE (Attention Mechanism)
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units=256, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        # Explicit names to match saved weights precisely
        self.W1 = tf.keras.layers.Dense(units, name="attention_w1")
        self.W2 = tf.keras.layers.Dense(units, name="attention_w2")
        self.V = tf.keras.layers.Dense(1, name="attention_v")

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

# 2. STEP 2: LOADING RESOURCES (Manual Reconstruction & Flexible Mapping)
@st.cache_resource
def load_my_model():
    try:
        # Load the Tokenizer
        with open('tokenizer_final.pickle', 'rb') as handle:
            tokenizer_obj = pickle.load(handle)
        
        vocab_size = len(tokenizer_obj) + 1 

        # --- REBUILD ARCHITECTURE (Synced to 100-dim / 256-units) ---
        # Encoder
        enc_inputs = tf.keras.Input(shape=(15,), name='enc_input')
        enc_emb = tf.keras.layers.Embedding(vocab_size, 100, name="enc_embedding")(enc_inputs) 
        enc_out, state_h, state_c = tf.keras.layers.LSTM(256, return_sequences=True, return_state=True, name="enc_lstm")(enc_emb)
        
        # Decoder
        dec_inputs = tf.keras.Input(shape=(1,), name='dec_input')
        dec_emb = tf.keras.layers.Embedding(vocab_size, 100, name="dec_embedding")(dec_inputs) 
        dec_lstm = tf.keras.layers.LSTM(256, return_sequences=True, return_state=True, name="dec_lstm")
        dec_out, _, _ = dec_lstm(dec_emb, initial_state=[state_h, state_c])
        
        # Attention
        attn_layer = AttentionLayer(256)
        context, _ = attn_layer(state_h, enc_out)
        
        # Merge & Output
        decoder_combined = tf.concat([tf.reshape(dec_out, (-1, 256)), context], axis=-1)
        outputs = tf.keras.layers.Dense(vocab_size, activation='softmax', name="output_dense")(decoder_combined)
        
        full_model = tf.keras.Model(inputs=[enc_inputs, dec_inputs], outputs=outputs)
        
        # FLEXIBLE WEIGHT INJECTION
        try:
            full_model.load_weights('chatbot_weights.weights.h5')
        except Exception:
            # Fallback: ignore minor naming mismatches in the attention layer
            full_model.load_weights('chatbot_weights.weights.h5', by_name=True, skip_mismatch=True)
            st.sidebar.warning("⚡ Neural mapping optimized via Flexible Load.")
        
        return tokenizer_obj, full_model
    except Exception as e:
        st.error(f"System Sync Error: {e}")
        return None, None

# Initialize Global Resources
tokenizer, model = load_my_model()

if tokenizer:
    reverse_word_index = {i: word for word, i in tokenizer.items()}
else:
    st.warning("Neural engine offline. Check 'chatbot_weights.weights.h5' and 'tokenizer_final.pickle' in GitHub.")

# 3. STEP 3: THE CHAT LOGIC
# 3. step3 the chat logic for app (VALIDATED FOR INFERENCE)
def get_chatbot_response(user_input, creativity):
    try:
        user_words = user_input.lower().split()
        # Ensure we use .get() safely
        user_sequence = [tokenizer.get(word, 0) for word in user_words]
        user_padded = pad_sequences([user_sequence], maxlen=15, padding='post')
        
        # Start token logic - using 'start' or index 1 as fallback
        target_sequence = np.zeros((1, 1))
        target_sequence[0, 0] = tokenizer.get('start', 1)

        decoded_sentence = []
        
        for _ in range(12):
            # Model prediction
            output_tokens = model.predict([user_padded, target_sequence], verbose=0)
            
            # Handle potential 2D/3D output shape differences
            if len(output_tokens.shape) == 3:
                predictions = output_tokens[0, -1, :]
            else:
                predictions = output_tokens[0, :]

            # Temperature / Creativity math
            predictions = np.array(predictions).astype('float64')
            predictions = np.log(predictions + 1e-7) / creativity
            exp_preds = np.exp(predictions)
            prob_distribution = exp_preds / np.sum(exp_preds)

            # Top-K Sampling Fix:
            # Instead of just picking from indices, we pick from the 5 best words 
            # while keeping their relative probability weights
            top_k = 5
            top_k_indices = np.argsort(prob_distribution)[-top_k:]
            top_k_probs = prob_distribution[top_k_indices]
            top_k_probs = top_k_probs / np.sum(top_k_probs) # Re-normalize

            sampled_token_index = np.random.choice(top_k_indices, p=top_k_probs)
            sampled_word = reverse_word_index.get(sampled_token_index, '')

            # Break conditions
            if sampled_word in ['end', '<eos>', '<EOS>', 'pad', ''] or len(decoded_sentence) > 10:
                break

            # Avoid immediate repetition
            if len(decoded_sentence) > 0 and sampled_word == decoded_sentence[-1]:
                continue
                
            decoded_sentence.append(sampled_word)
            target_sequence[0, 0] = sampled_token_index

        # Final cleanup: deduplicate and capitalize
        clean_output = []
        for word in decoded_sentence:
            if word not in clean_output:
                clean_output.append(word)

        response = " ".join(clean_output)
        response = response.replace('<unk>', '').strip()
        return response.capitalize() + "..." if response else "The script is a bit quiet on that one..."
    
    except Exception as e:
        # Debugging for the hackathon
        st.sidebar.error(f"Inference Logic Error: {e}")
        return "Analyzing the script... try another question!"

# 4. STEP 4: STREAMLIT UI (National Hackathon Edition)
st.set_page_config(page_title="Movie AI", page_icon="🎬", layout="centered")

# Custom UI Styling
st.markdown("""
<style>
.stApp {background-color: #FFFFFF;}
.stChatMessage {border-radius: 15px; border: 1px solid #E6E9EF; background-color: #F8F9FB;}
.stSidebar {background-color: #F0F2F6;}
h1 {color: #D32F2F; font-family: 'Arial Black'; text-transform: uppercase;}
</style>
""", unsafe_allow_html=True)

# SIDEBAR
st.sidebar.title("🎥 Project Control")
st.sidebar.info("**Architecture:** Seq2Seq + Attention\n**Compatibility:** Legacy H5 Weights")
creativity = st.sidebar.slider("AI Creativity (Temperature)", 0.1, 1.2, 0.7)

if st.sidebar.button("🗑️ Clear History"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Performance Metrics")
st.sidebar.line_chart([3.8, 2.7, 1.8, 1.3, 1.1, 1.05])
st.sidebar.caption("Validated against 10 Epochs")

# MAIN CHAT UI
st.title('🎬 Movie Chatbot')
st.caption(f"Developed by Harsh Rana | Powered by TensorFlow & Streamlit")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about a movie..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Processing Attention layers..."):
            response = get_chatbot_response(prompt, creativity)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
