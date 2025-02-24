from dotenv import load_dotenv
import os
from langchain_community.tools import tool

from langchain_community.tools.reddit_search.tool import RedditSearchRun
from langchain_community.utilities.reddit_search import RedditSearchAPIWrapper
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
# from langchain_community.tools.google_scholar import GoogleScholarQueryRun
# from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper
from serpapi import GoogleSearch 

from loguru import logger

load_dotenv()

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


# add "rag" tool
@tool
def rag_tool():
    """RAG tool."""
    pass

# Initialize tools
tools = [
    wikipedia_tool,
    reddit_tool,
    google_scholar_tool,
    # rag_tool()
]