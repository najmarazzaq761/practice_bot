import streamlit as st
import os
import google.generativeai as genai
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Setting page configuration 
st.set_page_config(page_title="✨ Data Seekho Guide", page_icon="🧠", layout="wide")

# Sidebar content 
st.sidebar.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTNl6Gok8ubOtQLNgMDmKQQGFdV5OtfJWYSOqyYTfM-uNml-vaBpavqlUXpdYdoHWed0LY&usqp=CAU", use_column_width=True)
st.sidebar.markdown("Welcome to the Data Seekho Guide developed by Najma Razzaq. This app is designed to provide you with information about Data Seekho.")

# Main title
st.markdown("<h1 style='text-align: center; color: white;'>✨ Data Seekho Guide</h1>", unsafe_allow_html=True)

# Load data from a website (cache the function to prevent repeated fetching)
@st.cache_data
def load_data():
    loader = WebBaseLoader([
        "https://dataseekho.com/",
        "https://dataseekho.com/free-courses/",
         "https://dataseekho.com/join-us/"
        "https://dataseekho.com/about-us/"
    ])
    return loader.load()

# Split the loaded data into chunks (cached for efficiency)
@st.cache_data
def split_data(_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
    return text_splitter.split_documents(_data)

# Create a vector store using FAISS (cache the vector store creation)
@st.cache_resource
def create_vector_store(_docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=st.secrets["GOOGLE_API_KEY"])
    return FAISS.from_documents(documents=_docs, embedding=embeddings)

# Load and process data
data = load_data()
docs = split_data(data)
vectorstore = create_vector_store(docs)

# Set up retriever and LLM
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=st.secrets["GOOGLE_API_KEY"], temperature=0, max_tokens=None, timeout=None)

# Defining the prompt template
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# User input and response generation
query = st.text_input("🗣️ Enter your query:")

if query:
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": query})
    st.write(response["answer"])
