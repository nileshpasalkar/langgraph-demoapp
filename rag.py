from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    temperature=0
    )

def load_documents(folder_path: str) -> List[Document]:
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif filename.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        else:
            print(f"Unsupported file type: {filename}")
            continue
        documents.extend(loader.load())
        print(f"Loaded {len(documents)} documents from {filename}")
    return documents

folder_path = "docs"
documents = load_documents(folder_path)
print(f"Loaded {len(documents)} documents from the folder.")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

splits = text_splitter.split_documents(documents)
print(f"Split the documents into {len(splits)} chunks.")

embeddings = HuggingFaceEmbeddings()
document_embeddings = embeddings.embed_documents([split.page_content for split in splits])
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
print(f"Created embeddings for {len(document_embeddings)} document chunks.")

collection_name = "my_collection"
vectorstore = Chroma.from_documents(
    collection_name=collection_name,
    documents=splits,
    embedding=embedding_function,
    persist_directory="./chroma_db"
)
print("Vector store created and persisted to './chroma_db'")

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

template = """Answer the question based only on the following context:
{context}
Question: {question}
Answer: """

prompt = ChatPromptTemplate.from_template(template)

def docs2str(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | docs2str, "question": RunnablePassthrough()}
    | prompt
    | llm
)
question = input("Enter your question: ")
response = rag_chain.invoke(question)
print(f"Question: {question}")
print(f"Answer: {response}")
