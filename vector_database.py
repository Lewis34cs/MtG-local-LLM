from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import os

folder_path = "data/"
chroma_path = "chroma_db/"

def load_documents(fodler_path: str):
    """Load documents from a directory."""
    loader = PyPDFDirectoryLoader(folder_path)
    return loader.load()

def split_documents(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        # max chars per chunk
        chunk_size=chunk_size, 
        # amount of overlap between chunks
        # useful for ensuring context is preserved between chunks
        chunk_overlap=chunk_overlap, 
        # counts characters, not tokens
        length_function=len, 
        # seperators like (\n, \n\n, etc.) are treated as normal strings, not regex
        # useful for ensuring that the text is split at natural boundaries
        is_separator_regex=False
    )
    # returns a list of Document chunks - each preserving metadata from the original document
    return text_splitter.split_documents(documents)

# contains the pages of the document(s) as Document objects
loaded_docs = load_documents(folder_path)
# split the documents into smaller chunks
chunks = split_documents(loaded_docs)

def create_chunk_ids(chunks: list[Document]):
    """Create unique IDs for each chunk."""
    prev_page_id = None
    curr_chunk_id = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = int(chunk.metadata.get("page"))
        curr_page_id = f"{source}_pg{page}"

        if curr_page_id == prev_page_id:
            curr_chunk_id += 1

        else:
            curr_chunk_id = 0
            
        full_chunk_id = f"{curr_page_id}_chunk{curr_chunk_id}"
        chunk.metadata["id"] = full_chunk_id

        prev_page_id = curr_page_id
    
    return chunks

def add_to_database(chunks: list[Document]):
    if not os.path.exists(chroma_path):
        os.makedirs(chroma_path)

        vector_db = Chroma(
            collection_name="mtg_docs",
            embedding_function=OllamaEmbeddings(model="llama3.2"),
            persist_directory=chroma_path
        )

        updated_chunks = create_chunk_ids(chunks)
        chunk_ids = [chunk.metadata["id"] for chunk in updated_chunks]

        vector_db.add_documents(
            documents=updated_chunks,
            ids=chunk_ids
        )

        print(f"Added {len(updated_chunks)} chunks to the database.")

    else:
        vector_db = Chroma(
            collection_name="mtg_docs",
            embedding_function=OllamaEmbeddings(model="llama3.2"),
            persist_directory=chroma_path
        )
    return vector_db

vector_db = add_to_database(chunks)

retriever = vector_db.as_retriever(
    search_kwargs={
        "k": 5
    }
),