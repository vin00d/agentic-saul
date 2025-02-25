from pathlib import Path
from dotenv import load_dotenv
import os, sys
from langchain_community.tools import tool

## imports for tools
from langchain_community.tools.reddit_search.tool import RedditSearchRun
from langchain_community.utilities.reddit_search import RedditSearchAPIWrapper
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from serpapi import GoogleSearch 

## imports for vectore store retriever
from langchain_openai import OpenAIEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from loguru import logger

load_dotenv()

# Add the parent directory to the system path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
semantic_scholar_tool = SemanticScholarQueryRun()
# google_scholar_tool = GoogleScholarQueryRun(api_wrapper=GoogleScholarAPIWrapper())
reddit_tool = RedditSearchRun(
    api_wrapper=RedditSearchAPIWrapper(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT")
    )
)

# Google Scholar tool
@tool
def google_scholar_tool(query: str, top_k: int = 10) -> str:
    """Search Google Scholar Case Law for judicial opinions from numerous federal and state courts in the US about the given query."""
    params = {
        "q": query,
        "api_key": os.getenv("SERP_API_KEY"),
        "engine": "google_scholar",
        "hl": "en",
        "as_sdt": "4",
    }
    logger.debug(f"Google Scholar params: {params}")
    search = GoogleSearch(params)
    results = search.get_dict().get("organic_results", [])
    logger.debug(f"Google Scholar results: {results}")
    if not results:
        return "No good Google Scholar results found."

    # Format the results
    formatted_results = [
        f"Title: {result.get('title', '')}\n"
        f"Snippet: {result.get('snippet', '')}\n"
        f"Summary: {result.get('publication_info', {}).get('summary', '')}"
        for result in results[:top_k]  # Limit to top_k results
    ]
    logger.info(f"Google Scholar results: {formatted_results}")
    return "\n\n".join(formatted_results)


######## Vector Store Retriever ########
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# finuned model embeddings - model name "vin00d/snowflake-arctic-legal-ft-1"
# embeddings = HuggingFaceEmbeddings(model_name="vin00d/snowflake-arctic-legal-ft-1")

# Initialize Qdrant client
client = QdrantClient(":memory:")

client.create_collection(
    collection_name="legal_mumbo_jumbo",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="legal_mumbo_jumbo",
    embedding=embeddings,
)

# Load data
path = Path("saul/rag_data/")
# epub_loader = UnstructuredEPubLoader(path + "BlacksLaw9thEdition.epub")
# read all pdfs in the directory
pdf_loader = DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyPDFLoader)
# Add documents to the vector store

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 750,
    chunk_overlap  = 50,
    length_function = len
)

docs = text_splitter.split_documents(pdf_loader.load())

# Add documents to the vector store
_ = vector_store.add_documents(docs)

# Create a vector store retriever
vector_store_retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)

# Retrieval Augmented Generation (RAG) tool
@tool
def rag_tool(query: str) -> str:
    """A vector database retriever that contains all the legal terms and definitions - a legal glossary."""        
    logger.info(f"Invoking RAG tool query: {query}")
    results = vector_store_retriever.invoke(query)

    return results



# Initialize tools
tools = [
    wikipedia_tool,
    reddit_tool,
    google_scholar_tool,
    rag_tool,
]