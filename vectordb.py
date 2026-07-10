from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import os

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

print(f"Printing first documen\n")
print(documents[0])
print(f"Printing first chunk of document\n")
print(splits[0])
print(f"Printing metadata of first chunk of document\n")
print(splits[0].metadata)

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

#Fetching data from vectorstore
query = "Waht is fittrack watch 5?"
search_results = vectorstore.similarity_search(query, k=2)
print(f"\nTop 2 most relevant chunks for the query: '{query}'\n")
for i, result in enumerate(search_results, 1):
    print(f"Result {i}:")
    print(f"Source: {result.metadata.get('source', 'Unknown')}")
    print(f"Content: {result.page_content}")
    print()

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
retriever_results = retriever.invoke("When was GreenGrow Innovations founded?")
print(retriever_results)




