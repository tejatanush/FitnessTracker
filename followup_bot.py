from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")

def run_chatbot(user_input,text1):
    chat = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7,groq_api_key=api_key)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful fitness assistant...answer the user question based on conversation history {text1}"),
        ("human", "{user_input}")
    ])
    chain = prompt | chat
    response = chain.invoke({"user_input": user_input,"text1":text1})
    return response.content