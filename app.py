import streamlit as st
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from google.generativeai.types import HarmCategory, HarmBlockThreshold



def user_input(user_question, history):

    prompt_template = """   
    Your are a friendly chatbot named JURIS (Judicial Understanding & Response Intelligent System) who provides information on crime,
    legal matters, and law knowledge while encouraging users to follow the law.Analyze the question correctly and provide an answer in details and in a format.
    If the question is related to any coding or technical aspects, do not provide an answer.
    History: {history}
    Context: {context}?

    Question: {question}

    """
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001", google_api_key="AIzaSyDmFYX77xebjkZppD7FzBtf3qqqrlmAeGo")
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question, k=5)
    print(docs)
    context_str = '\n'.join(doc.page_content for doc in docs)

   
    #chain = get_conversational_chain()
    promt = prompt_template.format(context = context_str, question = user_question, history = history)
    model = genai.GenerativeModel('gemini-pro',safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    })
    response = model.generate_content(promt)
    #print(response.text)
    #print(response.parts)
    #return response

    # Accessing parts of the response
    full_response = ""
    for part in response.parts:
        full_response += part.text
    
    #print(full_response)
    return full_response

# Set Streamlit page configuration
st.set_page_config(page_title="JURIS")

#Title of the app
st.title("JURIS")

#Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to concatenate chat history into a single string
def concatenate_chat_history(chat_history):
    concatenated_history = ""
    for message in chat_history:
        if isinstance(message, dict):  # Check if message is a dictionary
            if message.get("role") == "user":
                concatenated_history += f"user: {message.get('content')}\n"
            elif message.get("role") == "assistant":
                concatenated_history += f"assistant: {message.get('content')}\n"
    return concatenated_history.strip()

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
# Use markdown with CSS to position the button

def core(user_question):
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                #st.session_state.messages.append(prompt)
                history = concatenate_chat_history(st.session_state.messages)
                response = user_input(user_question, history)
                if len(response) == 0:
                    print("String is empty")
                    response = user_input(user_question, history)
                    placeholder = st.empty()
                    full_response = ''
                    for item in response:
                        full_response += item
                        placeholder.markdown(full_response)
                    message = {"role": "assistant", "content": full_response}
                    st.session_state.messages.append(message)
                else:
                    placeholder = st.empty()
                    full_response = ''
                    for item in response:
                        full_response += item
                        placeholder.markdown(full_response)
                    message = {"role": "assistant", "content": full_response}
                    st.session_state.messages.append(message)

example_prompts = [
    "Can you give detail about my fundamental rights in india?",
    "What are my rights if I've been arrested?",
    "Tell me about the famous cases in India.",
    "Explain the Constitution of India in detail",
    "Tell me some facts related to crime in India."
]
# Set the width of the sidebar
st.sidebar.title("Example Prompts")

# Display buttons for each example prompt in the sidebar
for pro in example_prompts:
    if st.sidebar.button(pro):
        #st.write("You selected:", prompts)
        st.session_state.messages.append({"role": "user", "content": pro})
        with st.chat_message("user"):
            st.write(pro)
        core(pro)

        # You can then pass 'prompt' as input to your chatbot function for processing

st.sidebar.title("To Clear Chat History")
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear', on_click=clear_chat_history)


# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    core(prompt)

# Generate a new response if last message is not from assistant
#if st.session_state.messages[-1]["role"] != "assistant":
#    with st.chat_message("assistant"):
#        with st.spinner("Thinking..."):
#            #st.session_state.messages.append(prompt)
#            history = concatenate_chat_history(st.session_state.messages)
#            response = user_input(history)
#            placeholder = st.empty()
#            full_response = ''
#            for item in response:
#                full_response += item.text
#                placeholder.markdown(full_response)
#            placeholder.markdown(full_response)
#    message = {"role": "assistant", "content": full_response}
#    st.session_state.messages.append(message)

