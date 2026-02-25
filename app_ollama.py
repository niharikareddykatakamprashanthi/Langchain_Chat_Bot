from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
prompt = ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant.Please respond to the user queries"),
    ("user","Question:{question}")

])

# Streamlit Framework
st.title("Langchain Demo with OLLAMA")
user_text = st.text_input("Search the topic you want")

# Ollama LLM
llm = Ollama(model='llama2')
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if user_text:
    st.write(chain.invoke({'question':user_text}))