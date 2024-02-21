from pydantic import HttpUrl

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma

# def load_and_store(url: list[HttpUrl]):


def store_to_db(url: HttpUrl):
    loader = WebBaseLoader(url)
    docs = loader.load()

    textSplitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,
        chunk_overlap=100
    )
    splits = textSplitter.split_documents(docs)

    embedding = GPT4AllEmbeddings()

    vectorstore = Chroma.from_documents(
        documents=splits,
        collection_name="CRAG-Chroma",
        embedding=embedding
    )

    retriever = vectorstore.as_retriever()

    return retriever


if __name__ == "__main__":
    store_to_db("https://docs.pydantic.dev/latest/api/networks/")
