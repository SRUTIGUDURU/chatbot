import streamlit as st
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

body {
    background-color: #FFDEE9;
    color: #1F1B24;
    font-family: 'Arial', sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    color: #BF1378;
}

.stButton > button {
    background-color: #E4479A;
    color: white;
    border-color: #BF1378;
}

.stButton > button:hover {
    background-color: #730151;
}

.stTextInput > div > input, .stTextArea > div > textarea {
    background-color: #FFF2F7;
    color: #1F1B24;
    border-color: #F782B8;
}

.stMarkdown {
    color: #730151;
}


@st.cache_resource
def load_model_and_tokenizer(model_selected):
    tokenizer = AutoTokenizer.from_pretrained(model_selected)
    model = AutoModelForCausalLM.from_pretrained(model_selected)
    return model, tokenizer

available_models = {
    "GPT-2": "gpt2",
    "DialoGPT": "microsoft/DialoGPT-medium"
}

st.title("Chatbot")

model_name = st.radio("Choose a model:", list(available_models.keys()))
selected_model = available_models[model_name]
st.write(f"You selected: {model_name}")

model, tokenizer = load_model_and_tokenizer(selected_model)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=500)  # Set max_length

llm = HuggingFacePipeline(pipeline=pipe)

memory = ConversationBufferMemory()

conversation_chain = ConversationChain(memory=memory, llm=llm)

def generate_response(user_input):
    response = conversation_chain.run(user_input)
    return response

if "history" not in st.session_state:
    st.session_state.history = []

if st.button("Clear Conversation History"):
    st.session_state.history = []
    st.experimental_rerun()

user_input = st.text_input("You:", key="user_input", placeholder="Type something! Your chatbot is waiting to answer it!")

if st.button("Send"):
    if user_input:
        response = generate_response(user_input)
        st.session_state.history.append((user_input, response))

for user_msg, bot_msg in st.session_state.history:
    st.markdown(f"**You:** {user_msg}")
    st.markdown(f"**Friend:** {bot_msg}")
