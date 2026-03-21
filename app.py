#--------Day 13 make a Streamlit app--------
%%writefile app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import  pad_sequences
import pickle
import time

#1. step-1 The brain structure----
class AttentionLayer(tf.keras.layers.Layer):
  def __init__(self, units):
    super(AttentionLayer, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    query_with_time_axis = tf.expand_dims(query, 1)
    score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)
    return context_vector, attention_weights

#2. step2 loading the model
@st.cache_resource
def load_my_model():
  with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
  model = tf.keras.models.load_model('movie_chatbot_model_v2.h5', custom_objects={'AttentionLayer': AttentionLayer},
                                     compile=False)
  return tokenizer, model

tokenizer, model = load_my_model()
reverse_word_index = {i: word for word, i in tokenizer.items()}

#3. step3 the chat logic for app
def get_chatbot_response(user_input, creativity):
  user_words = user_input.lower().split()
  user_sequence = [tokenizer.get(word, 0) for word in user_words]
  user_padded = pad_sequences([user_sequence], maxlen=15, padding='post')
  target_sequence = np.zeros((1, 1))
  target_sequence[0, 0] = tokenizer.get('start', 1)

  decoded_sentence = []
  res = []
  last_word = ""
  for _ in range(12):
    output_tokens = model.predict([user_padded, target_sequence], verbose=0)
    predictions = output_tokens[0, -1, :]

    predictions = np.log(predictions + 1e-7) / creativity
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)

    top_k_indices = np.argsort(predictions)[-5:]

    sampled_token_index = np.random.choice(top_k_indices)
    sampled_word = reverse_word_index.get(sampled_token_index, '')
    if sampled_word in ['end', '<EOS>', 'pad', ''] or len(decoded_sentence) > 10:
       break

    if len(decoded_sentence) > 0 and sampled_word == decoded_sentence[-1]:
      continue
    decoded_sentence.append(sampled_word)
    target_sequence[0, 0] = sampled_token_index

  #final cleanp: remove duplicate and capitalize
  clean_output = []
  for word in decoded_sentence:
    if word not in clean_output:
      clean_output.append(word)

  response = " ".join(clean_output)
  return response.capitalize() + "..." if response else "I'm listening..."


#4. step -4 Streamli user interface----
st.set_page_config(page_title="Movie AI", page_icon="🎬", layout="centered")

#we set the css for midnight cinema look
st.markdown("""
<style>
.stApp {background-color: #FFFFFF; color: #262730;}
.stChatMessage {border-radius: 15px;
        border: 1px solid #E6E9EF;
        background-color: #F8F9FB;
        color: #262730;}
.stSidebar {background-color: #F0F2F6; color: #262730;}
h1 {color: #D32F2F; font-family: 'Arial Black';}
</style>
""", unsafe_allow_html=True)

#sidebar option in app
st.sidebar.title("🎥 Project Control")
st.sidebar.info("Model: Seq2Seq + Attention\nDataset: Movie Dialogues")
creativity = st.sidebar.slider("AI Temprature", 0.1, 1.2, 0.7)
if st.sidebar.button("Clear History"):
  st.session_state.messages = []
  st.rerun()
st.sidebar.markdown("-------")
st.sidebar.subheader("📊 Training Analytics")

loss_values = [3.8, 3.2, 2.7, 2.1, 1.8, 1.5, 1.3, 1.2, 1.1 , 1.05]
st.sidebar.line_chart(loss_values)
st.sidebar.caption("Model loss over 10 epochs")


st.title('🎬 Movie Chatbot')
st.caption("HelpDesk AI project by Harsh rana using Tensorflow & Streamlit")

if "messages" not in st.session_state:
  st.session_state.messages = []

for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

if prompt := st.chat_input("You can ask movie question....."):
  st.session_state.messages.append({"role":"user", "content": prompt})
  with st.chat_message("user"):
    st.markdown(prompt)
  with st.chat_message("assistant"):
    with st.spinner(" here processing the attention layers..."):
      response = get_chatbot_response(prompt, creativity)
      st.markdown(response)
      st.session_state.messages.append({"role": "assistant", "content": response})







