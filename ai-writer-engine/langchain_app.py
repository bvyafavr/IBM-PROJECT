from langchain_google_genai import ChatGoogleGenerativeAI #lets us get the gemini wrapper to use at LLM
from langchain.prompts import ChatPromptTemplate#used for prompt enginnering 
from langchain.memory import ConversationBufferMemory#adds memory 
from langchain.chains import ConversationChain#to make it into a chanin ,so we can user predict() to send input into a chain

#loads the gemini api key from the .env file 
import os
from dotenv import load_dotenv
load_dotenv()
gemini_api_key=os.getenv("GEMINI_API_KEY")


#cretats a llm instance to do stuff 
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-pro",
    google_api_key=gemini_api_key,
    temperature=0.7 #0-deterministic 1-cretavive (how the anser is supposed to be )
)
memory=ConversationBufferMemory()#stores chat memory in a buffer 
conversation=ConversationChain(llm=llm,memory=memory)#combines memory into a chain 

#runs a loop ,takes user input and then uses ai to get resposne nd [rint it ]
while True:
    userInput=input("You:")
    response=conversation.predict(input=userInput)
    print("AI:",response)