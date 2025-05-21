# Contextual Q&A Bot with LangChain & Google Gemini

## Project Overview

This project demonstrates a **Contextual Question-Answering (Q&A) system** built using **LangChain**, powered by **Google's Gemini Large Language Models (LLMs)**, and exposed via a **FastAPI** REST API. The core functionality relies on **Retrieval-Augmented Generation (RAG)**, which enables the LLM to provide accurate and contextually relevant answers by retrieving information from a specific document base.

This project was developed as a rapid prototype to showcase key skills in Generative AI, LLM integration, prompt engineering, and API deployment.

## Key Features

* **Document Ingestion:** Loads text from a simple `.txt` file.
* **Text Chunking:** Breaks down the document into smaller, manageable chunks for efficient retrieval.
* **Vector Embeddings:** Uses Google's `embedding-001` model to convert text chunks into numerical vector representations.
* **Vector Store:** Utilizes `FAISS` (Facebook AI Similarity Search) as an in-memory vector database for efficient semantic search.
* **Retrieval-Augmented Generation (RAG):** Enhances LLM responses by providing relevant context retrieved from the vector store based on the user's query.
* **Large Language Model (LLM):** Integrates with Google's `gemini-1.5-flash` model for generating natural language answers.
* **Prompt Engineering:** Custom prompt template to guide the LLM's response generation.
* **REST API:** Exposes the Q&A functionality via a lightweight and performant FastAPI application.

## Technologies Used

* **Python 3.9+**
* **LangChain:** Framework for developing applications with LLMs.
* **Google Generative AI:** For access to Gemini LLMs and embedding models.
* **FAISS:** For in-memory vector similarity search.
* **FastAPI:** Modern, fast (high-performance) web framework for building APIs.
* **Uvicorn:** ASGI server for running FastAPI applications.
* **python-dotenv:** For securely managing API keys via `.env` files.
* **Git & GitHub:** Version control and code hosting.
* **VS Code:** Integrated Development Environment.

## Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

* Python 3.9+ installed.
* `pip` (Python package installer) installed.
* A Google AI Studio (Gemini) API Key. You can obtain one from [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey). Ensure the "Generative Language API" is enabled for your associated Google Cloud project.
* Git installed on your machine.

### 1. Clone the Repository

Open your terminal or command prompt and clone the project:

```bash
git clone https://github.com/Rachit1509/Contextual-QA-Bot.git
cd Contextual-QA-Bot 

### 2. Set Up Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

# Create the virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS / Linux:
source venv/bin/activate
# On Windows (Command Prompt):
# venv\Scripts\activate.bat
# On Windows (PowerShell):
# .\venv\Scripts\Activate.ps1


ou should see (venv) at the beginning of your terminal prompt, indicating the virtual environment is active.

### 3. Install Dependencies

Install all required Python packages from the requirements.txt file:

pip install -r requirements.txt

### 4. Configure API Key

Create a .env file in the root of your project directory (the same level as README.md and api.py) and add your Google API key:

GOOGLE_API_KEY="your_google_api_key_here"

### 5. Prepare Sample Data

The project uses a data.txt file for its knowledge base. If this file doesn't exist, it will be automatically created with sample content when you run the notebook or API. You can modify data.txt with your own content if you wish.

### How to Run the Project

You have two ways to interact with the Q&A bot: via a Jupyter Notebook for interactive development and testing, or via a REST API for programmatic access.
A. Running via Jupyter Notebook (main.ipynb)

This is ideal for exploring the code step-by-step, debugging, and understanding the RAG pipeline.

    Open VS Code.
    Open the project folder: File > Open Folder... and select Contextual-QA-Bot.
    Open main.ipynb: Navigate to main.ipynb in the VS Code Explorer and click to open it.
    Select the Kernel: In the top right corner of the notebook, ensure your virtual environment ((venv)) is selected as the Python kernel. If not, click on the kernel selector and choose the one associated with your venv.
    Run Cells: Execute each cell sequentially from top to bottom.
        The final cell (Cell 9) will start an interactive Q&A prompt in the notebook's output. Type your questions and press Enter. Type exit to quit.

B. Running as a REST API (api.py)

This allows you to access the Q&A bot programmatically from other applications or tools.

    Ensure your virtual environment is active in your terminal.

    Start the FastAPI server:
    Bash

uvicorn api:app --reload

The server will start on http://127.0.0.1:8000.

Access API Documentation (Swagger UI):

    Open your web browser and go to: http://127.0.0.1:8000/docs
    Here you can interact with the API, send test queries, and see the responses.

Test the /ask endpoint:

    In the Swagger UI (/docs), find the /ask (POST) endpoint.

    Click "Try it out".

    In the "Request body" text area, enter your question in JSON format:

{
  "query": "What is Retrieval-Augmented Generation?"
}

Click "Execute" and observe the response.

Alternatively, you can use curl from your terminal:

curl -X POST "[http://127.0.0.1:8000/ask](http://127.0.0.1:8000/ask)" \
     -H "Content-Type: application/json" \
     -d '{"query": "What LLM providers does LangChain support?"}'

