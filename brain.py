import re
from io import BytesIO
from typing import Tuple, List
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pypdf import PdfReader
import faiss

#  1. Parse the PDF
print("Parsing PDF files...")
def parse_pdf(file: BytesIO, filename: str) -> Tuple[List[str], str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        if not text:
            continue
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output, filename

#  2. Split into Chunks & Add Metadata
def text_to_docs(text: List[str], filename: str) -> List[Document]:
    if isinstance(text, str):
        text = [text]
    page_docs = [Document(page_content=page) for page in text]
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    doc_chunks = []
    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc_chunk = Document(
                page_content=chunk, 
                metadata={
                    "page": doc.metadata["page"],
                    "chunk": i,
                    "source": f"{doc.metadata['page']}-{i}",
                    "filename": filename
                }
            )
            doc_chunks.append(doc_chunk)
    return doc_chunks

#  3. Convert Docs to FAISS Index using HuggingFace Embeddings
def docs_to_index(docs: List[Document]):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index = FAISS.from_documents(docs, embeddings)
    return index
print("Index created with FAISS")

#  4. Main Function to Index Multiple PDFs
def get_index_for_pdf(pdf_files: List[bytes], pdf_names: List[str]):
    documents = []
    for pdf_file, pdf_name in zip(pdf_files, pdf_names):
        text, filename = parse_pdf(BytesIO(pdf_file), pdf_name)
        documents.extend(text_to_docs(text, filename))
    index = docs_to_index(documents)
    return index
print("All PDFs indexed successfully")