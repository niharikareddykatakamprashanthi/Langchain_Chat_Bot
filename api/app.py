from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes
import uvicorn 
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI(
    title = "Langchain Server",
    version = "1.0",
    description ="A simple API Server"
)

add_routes(
    app,
    ChatOpenAI(),
    path="/openai"
)
model = ChatOpenAI(model = 'gpt-4o-mini')

prompt1 =  ChatPromptTemplate.from_template("Write any essay about the {topic} with 100 words")
prompt2 =  ChatPromptTemplate.from_template("Write any poem about the {topic} with 100 words")

add_routes(
    app,
    prompt1 | model,
    path ="/essay"
)
add_routes(
    app,
    prompt2 | model,
    path ="/poem"
)

if __name__ == "__main__":
    uvicorn.run(app,host="localhost",port=8000)