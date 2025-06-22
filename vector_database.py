from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import os

folder_path = "data/"
chroma_path = "chroma_db/"

def load_documents(folder_path: str):
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
    """Create unique IDs for each chunk so we can reference them later."""

    # setting up variables to track the current page and chunk ID
    prev_page_id = None
    curr_chunk_id = 0

    for chunk in chunks:
        # for every chunk, we get the source and page number from the metadata
        source = chunk.metadata.get("source")
        page = int(chunk.metadata.get("page"))
        # and put them together to create a unique ID for the chunk
        curr_page_id = f"{source}_pg{page}"

        # if the current page ID is the same as the previous one, we add to the chunk ID
        if curr_page_id == prev_page_id:
            curr_chunk_id += 1
        # otherwise, we reset the chunk ID to 0
        else:
            curr_chunk_id = 0
        
        # we create a full chunk ID by combining the page ID and chunk ID
        full_chunk_id = f"{curr_page_id}_chunk{curr_chunk_id}"
        # and add it to the chunk's metadata
        chunk.metadata["id"] = full_chunk_id

        # and then set the previous page ID to the current one
        prev_page_id = curr_page_id
    
    # once all chunks have been processed, we return the updated list of chunks
    return chunks

# checking out the chunks to see how they look
# updated_chunks = create_chunk_ids(chunks)

# for chunk in updated_chunks:
#     print(f"{chunk}\n")

def add_to_database(chunks: list[Document]):
    # Check if the chroma database directory exists, if not, create it
    if not os.path.exists(chroma_path):
        os.makedirs(chroma_path)

        # instantiate the Chroma vector database with the Ollama embeddings model
        # and the path to the chroma database directory
        vector_db = Chroma(
            collection_name="mtg_docs",
            embedding_function=OllamaEmbeddings(model="mxbai-embed-large"),
            persist_directory=chroma_path
        )
        # create unique IDs for each chunk
        updated_chunks = create_chunk_ids(chunks)
        chunk_ids = [chunk.metadata["id"] for chunk in updated_chunks]

        # add the chunks to the vector database
        vector_db.add_documents(
            documents=updated_chunks,
            ids=chunk_ids
        )

        print(f"Added {len(updated_chunks)} chunks to the database.")

    else:
        # If the chroma database directory already exists, we load the existing database
        vector_db = Chroma(
            collection_name="mtg_docs",
            embedding_function=OllamaEmbeddings(model="mxbai-embed-large"),
            persist_directory=chroma_path
        )

    # return the vector database object
    return vector_db

# instantiate the vector database and add the chunks to it (if it doesn't already exist)
vector_db = add_to_database(chunks)

# create a retriever from the vector database
# this will allow us to query the database for relevant chunks based on a search query
retriever = vector_db.as_retriever(
    search_kwargs={
        "k": 20
    }
)