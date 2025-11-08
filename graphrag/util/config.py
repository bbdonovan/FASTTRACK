# -*- coding: utf-8 -*-
"""
configures the environment and logging for the application, and initializes
necessary credentials and constants for connections.
- Loads environment variables from a .env file using `dotenv`.
- Configures a logger with colored logs for better readability.
- Retrieves and validates the API keys from environment variables.
- Defines a constant `ANSWER_PROMPT` used for formatting responses based on vector store results.
"""

import logging
import os
import coloredlogs
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set debug mode, `True` will generate graphs in multiple
# formats (dot, png, text) for use in analyzing results
DEBUG_MODE=False

# Configure logger
LOGGER = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=LOGGER)

# Initialize embeddings and LLM using OpenAI
"""
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")
"""

MODEL_NAME = os.getenv("MODEL_NAME")

DB = os.getenv("VECTOR_DB")
if DB == "chroma":
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY")
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME")
    if not all([CHROMA_PERSIST_DIRECTORY, CHROMA_COLLECTION_NAME]):
        raise ValueError("Chroma DB configs must be set.")
    NODE_TABLE = CHROMA_COLLECTION_NAME

ANSWER_PROMPT = (
    "The original question is given below."
    "This question has been used to retrieve information from a vector store."
    "The matching results are shown below."
    "Only use context to answer the question."
    "Do not hallucinate or generate new information."
    "Do not include images or links in your output."
    "If you cannot answer the question based on the context say so."
    "Use the information in the results to answer the original question."
    "Give responses in pretty Markdown."
    "Original Question: {question}\n\n"
    "Vector Store Results:\n{context}\n\n"
    "Response:"
)

BASE_URL = os.getenv("BASE_URL")
BASE_DIR = os.getenv("INPUT_FOLDER")


#SIMILARITY_SEARCH_URL="https://python.langchain.com/v0.2/api_reference/community/graph_vectorstores/langchain_community.graph_vectorstores.cassandra.CassandraGraphVectorStore.html#langchain_community.graph_vectorstores.cassandra.CassandraGraphVectorStore.similarity_search"
#SIMILARITY_MMR_SEARCH_URL="https://python.langchain.com/v0.2/api_reference/community/graph_vectorstores/langchain_community.graph_vectorstores.cassandra.CassandraGraphVectorStore.html#langchain_community.graph_vectorstores.cassandra.CassandraGraphVectorStore.mmr_traversal_search"