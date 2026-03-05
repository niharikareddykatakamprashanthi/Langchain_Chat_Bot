import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_objectbox.vectorstores import ObjectBox
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
load_dotenv()

# Extract the API key
api_key = os.getenv('OPENAI_API_KEY')

# Assigining the title of the webpage
st.title('ObjectBox VectorstoreDB with gpt-4o-mini')

# Convert into vectors and store in the vectorDB
def vector_embeddings():
    if "vectors_db" not in st.session_state:
        # Load the data from the pdf files in the directory
        st.session_state.ld = PyPDFDirectoryLoader('./objectbox')
        st.session_state.loader = st.session_state.ld.load()
        print(st.session_state.loader)
        # Splitting the data into chunks
        st.session_state.split_document = RecursiveCharacterTextSplitter(chunk_size =1500, chunk_overlap = 300).split_documents(st.session_state.loader)
        # Create the database vector store
        st.session_state.vectors_db = ObjectBox.from_documents(st.session_state.split_document,OpenAIEmbeddings(),embedding_dimensions=1536)

# Creating an LLM
llm = ChatOpenAI(model = 'gpt-4o-mini',temperature=0,api_key = api_key)

# Creating a prompt 
#Use the following piece of context to answer the question asked.
#Please try to provide the answer only based on the context.
prompt = ChatPromptTemplate.from_template(
    """
    Return the EXACT sentence from the context that answers the question.
    Do NOT summarize.
    Do NOT shorten.
    Do NOT paraphrase.
    If the sentence exists, copy it word-for-word.
    {context}
    Question:{input}
    Helpful Answers:
    """
)

# Input prompt
input = st.text_input("Enter your Question from the Document")

if st.button('Document Embedding'):
    vector_embeddings()
    st.write('Objectbox Database is Ready')
    #response = retrieval_chain.invoke({"input":prompt})
    #st.session_state.write(response['answer'])

if input and st.session_state.get("vectors_db") is not None:
    # Connecting the llm and the prompt
    chain = create_stuff_documents_chain(llm,prompt)

    # Convert the vectorDB into search engine
    retrieval = st.session_state.vectors_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # Connecting the document_chain and vectorDB retriever
    retrieval_chain = create_retrieval_chain(retrieval,chain)

    response = retrieval_chain.invoke({'input':input})
    st.write(response['answer'])