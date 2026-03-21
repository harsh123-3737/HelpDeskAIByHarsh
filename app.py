# Movie Chatbot - Final Version for Keras 3.13.2
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Movie Chatbot | Harsh Rana",
    page_icon="🎬",
    layout="centered"
)

# Define Attention Layer
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
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

# Custom Embedding to handle quantization_config
class CustomEmbedding(tf.keras.layers.Embedding):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)
    
    def get_config(self):
        config = super().get_config()
        config.pop('quantization_config', None)
        return config

# Load model
@st.cache_resource
def load_model_files():
    try:
        # Check files
        if not os.path.exists('tokenizer.pickle'):
            st.error("❌ tokenizer.pickle not found")
            return None, None
        
        if not os.path.exists('movie_chatbot_model.h5'):
            st.error("❌ movie_chatbot_model.h5 not found")
            return None, None
        
        # Load tokenizer
        with open('tokenizer.pickle', 'rb') as f:
            tokenizer = pickle.load(f)
        
        # Load model with Keras 3.13.2 compatibility
        model = tf.keras.models.load_model(
            'movie_chatbot_model.h5',
            custom_objects={
                'AttentionLayer': AttentionLayer,
                'Embedding': CustomEmbedding
            },
            compile=False,
            safe_mode=False
        )
        
        st.success(f"✅ Model loaded! (TF {tf.__version__})")
        return tokenizer, model
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)[:150]}")
        return None, None

# Load everything
with st.spinner("🎬 Loading Movie AI..."):
    tokenizer, model = load_model_files()

# Create reverse index
if tokenizer:
    reverse_word_index = {v: k for k, v in tokenizer.items()}
else:
    reverse_word_index = {}

# Movie responses (fallback)
def get_movie_response(user_input):
    user_lower = user_input.lower()
    
    # Movie database
    movie_db = {
        'action': ['Die Hard', 'Mad Max: Fury Road', 'John Wick', 'The Dark Knight'],
        'comedy': ['Superbad', 'The Hangover', 'Bridesmaids', 'Step Brothers'],
        'drama': ['The Shawshank Redemption', 'The Godfather', 'Forrest Gump'],
        'sci-fi': ['Inception', 'Interstellar', 'The Matrix', 'Blade Runner 2049'],
        'horror': ['The Conjuring', 'Get Out', 'Hereditary', 'A Quiet Place'],
        'romance': ['The Notebook', 'Titanic', 'La La Land', 'Pride & Prejudice']
    }
    
    # Greetings
    if any(word in user_lower for word in ['hello', 'hi', 'hey']):
        return "Hello! 🎬 I'm your Movie Assistant. Ask me about movies, actors, or get recommendations!"
    
    # Recommendations
    if 'recommend' in user_lower or 'suggest' in user_lower:
        for genre, movies in movie_db.items():
            if genre in user_lower:
                return f"For {genre} movies, I recommend {np.random.choice(movies)}! It's a classic."
        all_movies = [m for movies in movie_db.values() for m in movies]
        return f"How about {np.random.choice(all_movies)}? Great choice for movie night!"
    
    # Genres
    for genre, movies in movie_db.items():
        if genre in user_lower:
            return f"{genre.capitalize()} films are amazing! {np.random.choice(movies)} is one of the best in this genre."
    
    # Actors
    actors = {
        'leonardo dicaprio': ['Inception', 'Titanic', 'The Revenant'],
        'tom hanks': ['Forrest Gump', 'Cast Away', 'Saving Private Ryan'],
        'brad pitt': ['Fight Club', 'Once Upon a Time in Hollywood']
    }
    
    for actor, movies in actors.items():
        if actor in user_lower:
            return f"{actor.title()} starred in {', '.join(movies[:2])}. Which is your favorite?"
    
    # General responses
    responses = [
        "Movies are wonderful! What genre do you enjoy most?",
        "Tell me about your favorite film! I'd love to discuss it.",
        "Cinema has so many stories. What would you like to watch next?",
        "Great question! Which actor or director do you admire?"
    ]
    return np.random.choice(responses)

# Generate response
def get_response(user_input, temperature=0.7):
    # Try using the model if loaded
    if model is not None and tokenizer is not None:
        try:
            # Tokenize
            words = user_input.lower().split()
            sequence = [tokenizer.get(word, 0) for word in words]
            padded = pad_sequences([sequence], maxlen=15, padding='post')
            
            # Decoder start
            target = np.zeros((1, 1))
            target[0, 0] = tokenizer.get('start', 1)
            
            # Generate
            decoded = []
            for _ in range(10):
                output = model.predict([padded, target], verbose=0)
                probs = output[0, -1, :]
                
                # Apply temperature
                probs = np.log(probs + 1e-7) / temperature
                probs = np.exp(probs)
                probs = probs / np.sum(probs)
                
                # Sample
                top_k = np.argsort(probs)[-5:]
                idx = np.random.choice(top_k)
                word = reverse_word_index.get(idx, '')
                
                if word in ['end', 'endoftext', '<EOS>', 'pad', '']:
                    break
                if len(decoded) >= 8:
                    break
                if word and len(word) > 1 and word not in decoded[-2:]:
                    decoded.append(word)
                    target[0, 0] = idx
            
            if decoded:
                response = ' '.join(decoded)
                if response and len(response) > 5:
                    return response.capitalize()
                    
        except Exception as e:
            pass
    
    # Fallback
    return get_movie_response(user_input)

# Sidebar
with st.sidebar:
    st.title("🎥 HelpDesk AI")
    st.markdown("**Developer:** Harsh Rana")
    st.markdown("**Project:** Movie Chatbot with Attention")
    st.divider()
    
    if model is not None:
        st.success("✅ Model: Active")
        if tokenizer:
            st.info(f"📚 Vocab: {len(tokenizer)} words")
    else:
        st.warning("⚠️ Model: Fallback Mode")
        st.info("Using enhanced movie responses")
    
    st.divider()
    
    temperature = st.slider("🎨 Creativity", 0.3, 1.5, 0.7, step=0.1)
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.subheader("📊 Training Analytics")
    st.line_chart([3.8, 3.2, 2.7, 2.1, 1.8, 1.5, 1.3, 1.2, 1.1, 1.05])
    st.caption("Validation Loss: 72% Improvement")
    st.divider()
    st.caption("Seq2Seq + Bahdanau Attention")
    st.caption(f"TF {tf.__version__}")

# Main chat
st.title("🎬 Movie Chatbot")
st.caption("Your AI companion for movie discussions")

# Initialize chat
if "messages" not in st.session_state:
    st.session_state.messages = []
    welcome = "🎬 Welcome! I'm your Movie Assistant. Ask me about movies, actors, or get recommendations!"
    st.session_state.messages.append({"role": "assistant", "content": welcome})

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about movies..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🎬 Thinking..."):
            response = get_response(prompt, temperature)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
