from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as  st
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found. Check your .env file.")

# Langsmith Tracking
#os.environ["LANGCHAIN_TRACING_V2"]  = "true"
#os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Template for Prompt

prompt = ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant.Please respond to the user queries"),
    ("user","Question:{question}")

])

# Streamlit Framework
st.title("Langchain Demo with OPENAI API")
user_text = st.text_input("Search the topic you want")

# openAI LLM
llm = ChatOpenAI(model = 'gpt-4o-mini',temperature=0.7,max_tokens=50,api_key=api_key)
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if user_text:
    st.write(chain.invoke({'question':user_text}))
