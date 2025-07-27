# Import necessary libraries
import os
from dotenv import load_dotenv
import gradio as gr

from huggingface_hub import InferenceClient

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_chroma import Chroma

# Load .env variables (HUGGINGFACEHUB_API_TOKEN)
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Paths
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

# HuggingFace SentenceTransformer embeddings
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# HuggingFaceHub LLM using Mistral-7B
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={
        "temperature": 0.1,
        "max_new_tokens": 500
    }
)

# OPTIONAL: Test basic LLM call
try:
    test_response = llm.invoke("What is IPC Section 302?")
    print("LLM Test Response:", test_response)
except Exception as e:
    print("LLM Test Error:", e)

# Load Chroma vector store
vector_store = Chroma(
    collection_name="COI_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# Create retriever
retriever = vector_store.as_retriever(search_kwargs={'k': 5})


# Stream response function for Gradio
def stream_response(message, history):
    try:
        # Get top-k context docs
        docs = retriever.invoke(message)
        if not docs:
            yield "Sorry, I couldn't find relevant information for this question."
            return

        # Build context from documents
        knowledge = "\n\n".join([doc.page_content for doc in docs])

        # Create RAG prompt
        rag_prompt = f"""
You are a helpful and honest legal assistant. You should answer **only based on the knowledge** given below.

Question: {message}

Knowledge:
{knowledge}

Chat History: {history}

Instructions:
Avoid adding assumptions or external knowledge. If you don't know something, respond accordingly.
"""

        print("Prompt to LLM:\n", rag_prompt)

        # Get final answer from LLM
        response = llm.invoke(rag_prompt)
        print("Response from LLM:", response)
        yield response

    except Exception as e:
        print("Error occurred:", str(e))
        yield f"An error occurred: {e}"


# Gradio UI setup
chatbot = gr.ChatInterface(
    fn=stream_response,
    title="⚖️ LegalBot - Ask about IPC & Law",
    description="Ask about IPC sections, legal advice, or how to handle false allegations.",
    theme="soft",
    textbox=gr.Textbox(
        placeholder="Ask me about IPC sections, laws, or legal advice...",
        container=False,
        autoscroll=True,
        scale=7
    ),
    examples=[
        "What is IPC Section 302?",
        "How to deal with false FIR?",
        "Is theft bailable?",
        "Can I get bail in a murder case?",
        "What is the punishment for defamation?"
    ]
)

# Launch Gradio app
if __name__ == "__main__":
    chatbot.launch()
