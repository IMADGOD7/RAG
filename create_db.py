#load pdf
#chunking
#embedding
#vector database
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

#loading the documents
data=PyPDFLoader(
    "document_loaders/GRU.pdf"
)
docs=data.load()

#splitting the documents into chunks
splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
chunks=splitter.split_documents(docs)

#embedding the chunks
embeddings=MistralAIEmbeddings(model="mistral-embed")

#creating the vector database
vectorstore=Chroma.from_documents(documents=chunks,embedding=embeddings,persist_directory="chroma_db")

