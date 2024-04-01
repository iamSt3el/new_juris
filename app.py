import streamlit as st

import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from haystack.components.websearch import SerperDevWebSearch
from haystack.utils import Secret
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.converters import HTMLToDocument
from urllib.parse import urlparse



def is_pdf(url):
    parsed_url = urlparse(url)
    file_extension = parsed_url.path.split('.')[-1].lower()
    return file_extension == 'pdf'




def google_search(user_question):
    prompt_template = """Convert the given question into an google search.Just give the google search 

    Question: {question}"""

    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001", google_api_key="AIzaSyAlloUClKdRegH-pERfnmdrotLDL2HXIDQ")

    promt = prompt_template.format(question = user_question)
    model = genai.GenerativeModel('gemini-pro',safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        })
    response = model.generate_content(promt)
    print(response.text)
    return response.text
def user_input(user_question, hist, state):
    prompt_template_1 = """You are a chatbot who provide legal knowledge.
                        You also have access to internet, So use that knowledge also.
                        Remember users details
                         Here is a record of previous conversations:
                        {history}
                        
                        The following reference is from internet use this as an exteranl knowledge to improve the user's question
                        Reference: {context}

                        Use this language 
                        {lang}
                        Question: {question}

                        Be informative, gentle, and formal.
                        If you don't have any answer tell to repharse the question, nothing else.
                        Answer:"""
    
    prompt_template_2 = """You are a chatbot who provide legal knowledge.
                        Remember users details
                         Here is a record of previous conversations:
                        {history}
                        
                        Use this language 
                        {lang}
                        Question: {question}

                        Be informative, gentle, and formal.
                        If you don't have any answer tell to repharse the question and if you can't answere because of real time data tell user to on the websearch button.
                        Answer:"""

    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001", google_api_key="AIzaSyAlloUClKdRegH-pERfnmdrotLDL2HXIDQ")
    
    if state:
        web_search = SerperDevWebSearch(api_key=Secret.from_token("522947fe5abceb98df51b4f8789de8b935f0c5cf"))
        output = web_search.run(google_search(user_question))
        link_content = LinkContentFetcher()
        html_converter = HTMLToDocument()

    
        try:
            data = link_content.run(urls = [output['documents'][0].meta['link']])

        except Exception as e:
            if not is_pdf(output['documents'][1].meta['link']):
                data = link_content.run(urls = [output['documents'][1].meta['link']])
            else:
                data = link_content.run(urls = [output['documents'][2].meta['link']])



        docs = html_converter.run(data['streams'])
        print(docs['documents'][0].content)
        promt = prompt_template_1.format(context = (output['documents'][0].content + docs['documents'][0].content), question = user_question, history = hist, lang = "English") 
    else:
        promt = prompt_template_2.format(question = user_question, history = hist, lang="English")
    model = genai.GenerativeModel('gemini-pro',safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    })
    try:
        response = model.generate_content(promt)
    except Exception as e:
        response = model.generate_content(promt)
    return response.text

st.set_page_config(page_title="JURIS")

st.title("JURIS")




if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def concatenate_chat_history(chat_history):
    concatenated_history = ""
    for message in chat_history:
        if isinstance(message, dict):  
            if message.get("role") == "user":
                concatenated_history += f"user: {message.get('content')}\n"
            elif message.get("role") == "assistant":
                concatenated_history += f"assistant: {message.get('content')}\n"
    return concatenated_history.strip()

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Create a toggle switch in the sidebar
state = st.sidebar.toggle("Web Search")

# Print the state
if state:
    st.sidebar.write("Web Search is ON")
else:
    st.sidebar.write("Web Search is OFF")

def core(user_question):
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                #st.session_state.messages.append(prompt)
                history = concatenate_chat_history(st.session_state.messages)
                #print(history)
                response = user_input(user_question, history, state)
                if len(response) == 0:
                    print("String is empty")
                    response = user_input(user_question, history)
                    placeholder = st.empty()
                    full_response = ''
                    for item in response:
                        full_response += item
                        placeholder.markdown(full_response)
                    message = {"role": "assistant", "content": response}
                    st.session_state.messages.append(message)
                else:
                    placeholder = st.empty()
                    full_response = ''
                    for item in response:
                        full_response += item
                        placeholder.markdown(full_response)
                    message = {"role": "assistant", "content": full_response}
                    st.session_state.messages.append(message)




st.sidebar.title("To Clear Chat History")
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear', on_click=clear_chat_history)




if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    core(prompt)


