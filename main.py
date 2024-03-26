import ollama
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from tqdm import tqdm

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(message)s')
LLM_MODEL = 'mixtral:8x7b-instruct-v0.1-q8_0' # 'llama2'
EMBEDDING_MODEL = 'llama2'
logging.error(f"LLM Model: {LLM_MODEL}, Embedding Model: {EMBEDDING_MODEL}")

# load the PDF
loader = PyPDFLoader('data/autodev.pdf')
logging.error("Loading document/s...")
#loader = DirectoryLoader('data', glob="**/*.md", show_progress=True)
docs = loader.load()

# chunk it
logging.error("Splitting document/s...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Create Ollama embeddings and vector store
logging.error("Creating embeddings and vector store...")
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# Create the retriever
retriever = vectorstore.as_retriever()

# Define the Ollama LLM function
def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

# Define the RAG chain
def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = retrieved_docs
    return ollama_llm(question, formatted_context)

# Use the RAG chain
logging.error("Using the RAG chain...")
result = rag_chain("""
What is autodev about?
""")
print(result)
