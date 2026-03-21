import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# 1. The Attention Mechanism
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

# 2. Manual Model Reconstruction (Bypasses Keras 3 Errors)
@st.cache_resource
def load_my_model():
    with open('tokenizer_final.pickle', 'rb') as handle: 
        tokenizer = pickle.load(handle)
    
    vocab_size = len(tokenizer) + 1
    
    # REBUILDING THE ARCHITECTURE EXACTLY
    # Encoder
    encoder_inputs = tf.keras.Input(shape=(15,), name='enc_input')
    enc_emb = tf.keras.layers.Embedding(vocab_size, 256)(encoder_inputs)
    encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(512, return_sequences=True, return_state=True)(enc_emb)
    
    # Decoder
    decoder_inputs = tf.keras.Input(shape=(1,), name='dec_input')
    dec_emb_layer = tf.keras.layers.Embedding(vocab_size, 256)
    dec_emb = dec_emb_layer(decoder_inputs)
    decoder_lstm = tf.keras.layers.LSTM(512, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=[state_h, state_c])
    
    # Attention
    attn_layer = AttentionLayer(512)
    context_vector, _ = attn_layer(state_h, encoder_outputs)
    
    # Combine and Output
    decoder_outputs = tf.reshape(decoder_outputs, (-1, decoder_outputs.shape[2]))
    concat = tf.concat([decoder_outputs, context_vector], axis=-1)
    dense = tf.keras.layers.Dense(vocab_size, activation='softmax')
    outputs = dense(concat)
    
    model = tf.keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)
    
    # LOAD RAW WEIGHTS (This ignores the 'batch_shape' error!)
    try:
        model.load_weights('chatbot_weights.weights.h5')
    except:
        # Fallback for different naming conventions
        model.load_weights('chatbot_weights.weights.h5', by_name=True, skip_mismatch=True)
        
    return tokenizer, model

tokenizer, model = load_my_model()
reverse_word_index = {i: word for word, i in tokenizer.items()}

# 3. Chat Logic
def get_chatbot_response(user_input, creativity):
    user_words = user_input.lower().split()
    user_sequence = [tokenizer.get(word, 0) for word in user_words]
    user_padded = pad_sequences([user_sequence], maxlen=15, padding='post')
    target_sequence = np.zeros((1, 1))
    target_sequence[0, 0] = tokenizer.get('start', 1)

    decoded_sentence = []
    for _ in range(12):
        predictions = model.predict([user_padded, target_sequence], verbose=0)
        
        # Temperature control
        preds = np.log(predictions + 1e-7) / creativity
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        
        sampled_token_index = np.random.choice(range(len(preds[0])), p=preds[0])
        sampled_word = reverse_word_index.get(sampled_token_index, '')
        
        if sampled_word in ['end', '<eos>', 'pad', ''] or len(decoded_sentence) > 10:
            break
        
        if len(decoded_sentence) > 0 and sampled_word == decoded_sentence[-1]:
            continue
            
        decoded_sentence.append(sampled_word)
        target_sequence[0, 0] = sampled_token_index

    response = " ".join(decoded_sentence).strip()
    return response.capitalize() + "..." if response else "I'm listening..."

# 4. Streamlit UI
st.set_page_config(page_title="Movie AI", page_icon="🎬", layout="centered")

st.markdown("""
<style>
.stApp {background-color: #FFFFFF;}
.stChatMessage {border-radius: 15px; border: 1px solid #E6E9EF; background-color: #F8F9FB;}
h1 {color: #D32F2F; font-family: 'Arial Black';}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🎥 Project Control")
st.sidebar.info("Model: Seq2Seq + Attention\nLegacy Compatibility: Enabled")
creativity = st.sidebar.slider("AI Temperature", 0.1, 1.2, 0.7)

if st.sidebar.button("Clear History"):
    st.session_state.messages = []
    st.rerun()

st.title('🎬 Movie Chatbot')
st.caption(f"Developer: Harsh Rana | National Hackathon Edition")

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
        with st.spinner("Processing neural weights..."):
            response = get_chatbot_response(prompt, creativity)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
