from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")

def bot(feedback, exercise):
    

    answer = []
    for f in feedback:
        chat = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0.7,  
            groq_api_key=api_key
        )
        
        # Option 1: Using f-string (no template variables)
        user_message = f"while doing {exercise} this is the feedback I received: {f}. Suggest me some causes, corrections, Risks & Related information and solution progression"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful gym and workout assistant that always answers in following format:\n\n"
            "**Causes:**\n- [List the causes]\n\n"
            "**Corrections:**\n- [List how to prevent it]\n\n"
            "**Risks & Related information:**\n- [List possible Risks and related info]\n\n"
            "**Solution Progression:**\n- [List possible solutions]\n\n"
            "Please follow this structure strictly."),
            ("human", user_message)
        ])
        
        chain = prompt | chat
        response = chain.invoke({"exercise": exercise, "f": f})  # Consistent variable names
        answer.append(response.content)
    
    return answer