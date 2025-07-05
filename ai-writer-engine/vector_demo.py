from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS#FAISS is a vector store helps to store and serach embeddings 

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI#gemini chat model for generating answers 
from dotenv import load_dotenv
import os 

#load environment variable 
load_dotenv()
gemini_api_key=os.getenv("GEMINI_API_KEY")

#LOAD AND SPLITV DOCUMENTS 
loader = DirectoryLoader(
    path="./docs",
    glob="**/*.txt",#select any txt nested file 
    loader_cls=TextLoader
)
documents = loader.load()
#Breaks long files into 500-character chunks with 100-char overlap.

#Necessary because LLMs like Gemini have a context limit.
text_splitter=CharacterTextSplitter(chunk_size=500,chunk_overlap=100)
docs=text_splitter.split_documents(documents)

#convert text to embeddings using gemini (embedding is basically number type )

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=gemini_api_key
)
#these vectors r stored inside FAISS and semantic search is done from here 
vectorstore=FAISS.from_documents(docs,embeddings)

#set up gemini LLM instance just like langchain_app.py

llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-pro",  # ✅ This is the correct chat model
    google_api_key=gemini_api_key,
    temperature=0.5
)

# Build a QA chain with retriever
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),#the retriever searches the valid chunks from faiss 
    return_source_documents=True
)

# Loop for user input
print("🔎 Ask questions about your docs (type 'exit' to quit)")
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    result = qa_chain(query)
    print("\nAI:", result["result"])