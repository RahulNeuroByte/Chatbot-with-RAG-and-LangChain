# 🤖 Chatbot with RAG and LangChain  

This project implements a Retrieval-Augmented Generation (RAG)-based chatbot using LangChain, OpenAI API, and FAISS/ChromaDB for intelligent question answering based on custom document ingestion. To get started, ensure you have Python 3.11 or above installed.  

First, clone this repository using:

```
git clone https://github.com/RahulNeuroByte/Chatbot-with-RAG-and-LangChain
cd chatbot-with-rag
```

Then, create and activate a virtual environment:

```
python -m venv venv
myvenv\Scripts\activate   # On Windows
source myvenv/bin/activate  # On MacOS/Linux
```

Next, install the required libraries:

```
pip install -r requirements.txt
```

Make sure you add your OpenAI API key in a `.env` file like this:

```
OPENAI_API_KEY="your_openai_api_key"
```

Finally, run the following commands to ingest your data and start the chatbot:

```
python ingest_database.py
python chatbot.py
```

Enjoy interacting with your custom knowledge-based AI assistant! 🎯