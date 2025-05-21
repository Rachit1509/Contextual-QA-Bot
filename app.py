# api.py

import os
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel # For defining request body structure

# LangChain components
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --- Environment Setup (Crucial for API) ---
load_dotenv() # Load env vars from .env file
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it in your .env file or environment.")

# --- RAG Chain Initialization (Runs once when the API starts) ---
# This part is similar to the setup_rag_chain function in your notebook,
# but it's executed directly when the API app starts up.
try:
    print("Initializing RAG chain for API...")
    # 1. Text Loading & Chunking
    data_file_path = "data.txt"
    if not os.path.exists(data_file_path):
        # Create a dummy data.txt if it doesn't exist for initial API testing
        with open(data_file_path, "w") as f:
            f.write("""LangChain is a framework designed to simplify the creation of applications using large language models (LLMs). It provides a modular and flexible architecture that allows developers to chain together different components, such as LLM wrappers, prompt templates, document loaders, and vector stores. This enables building complex applications like chatbots, summarization tools, and question-answering systems.

One of LangChain's core concepts is Retrieval-Augmented Generation (RAG). RAG systems enhance the capabilities of LLMs by allowing them to retrieve information from an external knowledge base before generating a response. This helps overcome the LLM's inherent limitations, such as outdated knowledge or inability to access specific private data. The process typically involves embedding documents into a vector space, storing them in a vector database, and then retrieving relevant documents based on a user's query. The retrieved documents are then provided as context to the LLM, which uses this context to formulate a more accurate and informed answer.

LangChain supports various LLM providers, including OpenAI, Hugging Face, Google, and others. It also integrates with numerous data sources and vector databases. The framework aims to abstract away much of the complexity involved in working directly with LLMs and their ecosystem, making it easier for developers to build powerful AI applications.
""")
        print(f"'{data_file_path}' created with sample content for API.")

    loader = TextLoader(data_file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # 2. Generate Embeddings & Create Vector Store
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    db = FAISS.from_documents(chunks, embeddings_model)
    retriever = db.as_retriever()

    # 3. Initialize LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=google_api_key)

    # 4. Define the RAG Prompt Template
    prompt = ChatPromptTemplate.from_template("""Answer the user's question based on the below context.
    If the answer is not in the context, clearly state that you don't have enough information from the provided context.
    Do not make up information.

    Context:
    {context}

    Question:
    {input}
    """)

    # 5. Create the Document Combining Chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # 6. Create the Retrieval Chain
    rag_chain = create_retrieval_chain(retriever, document_chain)
    print("RAG chain initialized for API successfully.")

except Exception as e:
    print(f"Error during RAG chain initialization for API: {e}")
    rag_chain = None # Set to None to indicate failure


# --- FastAPI Application ---
app = FastAPI(
    title="Contextual Q&A Bot API",
    description="API for answering questions based on provided document context using LangChain and Google Gemini.",
    version="0.1.0",
)

# Define the request body model
class QueryRequest(BaseModel):
    query: str

# Define the response body model
class QueryResponse(BaseModel):
    answer: str

# API Endpoint
@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    if rag_chain is None:
        raise HTTPException(status_code=500, detail="RAG chain not initialized. Check server logs.")

    try:
        # Invoke the RAG chain with the user's query
        # Use .invoke() for synchronous execution if not using async model
        # Use .ainvoke() for asynchronous if your LLM client supports it
        response = rag_chain.invoke({"input": request.query})
        answer = response['answer']
        return QueryResponse(answer=answer)
    except Exception as e:
        print(f"Error processing question: {e}")
        # Consider more specific error handling for different LLM exceptions
        raise HTTPException(status_code=500, detail=f"Failed to get answer from LLM: {e}")

# Basic root endpoint for health check
@app.get("/")
async def root():
    return {"message": "Welcome to the Contextual Q&A API! Use /docs for API documentation."}