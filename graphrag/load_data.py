# -*- coding: utf-8 -*-
"""
This script loads, processes, and visualizes documents from a list of docs.
"""
import os
from pathlib import Path
import json
import warnings
from typing import List, Tuple
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_community.document_loaders import AsyncHtmlLoader

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_chroma import Chroma
import chromadb
from chromadb.config import Settings
#from langchain_community.graph_vectorstores import NetworkxGraphVectorStore
from langchain_community.graphs import NetworkxEntityGraph
#from langchain_community.graph_vectorstores import GraphVectorStore
#from langchain_community.graph_vectorstores import CassandraGraphVectorStore
from langchain_community.graph_vectorstores.extractors import (
    LinkExtractorTransformer,
    HtmlLinkExtractor,
    KeybertLinkExtractor,
    GLiNERLinkExtractor,
    #KnowledgeGraphLinkExtractor,
)
from langchain_community.document_transformers import BeautifulSoupTransformer

from openai import OpenAI
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from cdlib import algorithms

from constants import DOCUMENTS
from util.config import LOGGER, DB, NODE_TABLE, BASE_URL, INPUT_DOCS_DIR, MODEL_NAME
from hybrid_retrieval import (
    build_entity_index_and_graph,
    hybrid_retrieve_and_answer,
    show_subgraph,
    graph_to_d3_tree,
    graph_to_d3_tree_from_subgraph,
)


#########
# Setup #
#########

# Global cache so we don't rebuild everything on every request
_PIPELINE_CACHE = {
    "initialized": False,
    "vectorstore": None,
    "graph": None,
    "entity_index": None,
    "llm": None,
}


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()

# Suppress all of the Langchain beta and other warnings
warnings.filterwarnings("ignore", lineno=0)

# Initialize embeddings and LLM using OpenAI
#embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

BASE_DIR = Path(__file__).resolve().parent

if DB == "chroma":
    from util.config import CHROMA_PERSIST_DIRECTORY, CHROMA_COLLECTION_NAME

    #BASE_DIR = os.getenv("INPUT_FOLDER")
    INPUT_DOCS_DIR = BASE_DIR / os.getenv("INPUT_FOLDER")
    CHROMA_PERSIST_DIRECTORY = str(BASE_DIR / CHROMA_PERSIST_DIRECTORY)

    #vdb_client = chromadb.Client(Settings(persist_directory=CHROMA_PERSIST_DIRECTORY))
    ##collection = client.create_collection("my_vectors")
    #collection = vdb_client.get_or_create_collection(CHROMA_COLLECTION_NAME)
    ##vdb_store = Chroma(collection_name=CHROMA_COLLECTION_NAME, embedding_function=embeddings, persist_directory=CHROMA_PERSIST_DIRECTORY)
    #vdb_store = Chroma(client=vdb_client, collection_name=CHROMA_COLLECTION_NAME, embedding_function=embeddings)


openai_client = OpenAI(api_key=OPENAI_API_KEY) if not BASE_URL else OpenAI(base_url=BASE_URL, api_key=OPENAI_API_KEY)
##llmchat_client = ChatOpenAI(api_key=OPENAI_API_KEY, model=MODEL_NAME) if not BASE_URL else ChatOpenAI(base_url=BASE_URL, api_key=OPENAI_API_KEY, model=MODEL_NAME)
#llmchat_client = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini") if not BASE_URL else ChatOpenAI(base_url=BASE_URL, api_key=OPENAI_API_KEY, model=MODEL_NAME)



#######################
# Utilities & Helpers #
#######################
def _load_input_documents() -> List[Document]:
    docs: List[Document] = []
    if not INPUT_DOCS_DIR.exists():
        return docs

    for path in INPUT_DOCS_DIR.glob("*.txt"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        docs.append(
            Document(
                page_content=text,
                metadata={"source": path.name},
            )
        )
    return docs


def _init_vectorstore_and_graph():
    docs = _load_input_documents()

    # Embeddings + Chroma
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vdb_client = chromadb.Client(
        Settings(persist_directory=CHROMA_PERSIST_DIRECTORY)
    )

    collection = vdb_client.get_or_create_collection(CHROMA_COLLECTION_NAME)

    vdb_store = Chroma(client=vdb_client, collection_name=CHROMA_COLLECTION_NAME, embedding_function=embeddings)

    # Only add if collection is empty
    if not collection.count():
        vdb_store.add_documents(docs)

    # Build entity index + KG (simple doc-entity graph)
    entity_index, G = build_entity_index_and_graph(docs)

    # LLM for answering
    llmchat_client = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0.1) if not BASE_URL else ChatOpenAI(base_url=BASE_URL, api_key=OPENAI_API_KEY, model=MODEL_NAME, temperature=0.1)

    return vdb_store, G, entity_index, llmchat_client


def _ensure_pipeline_initialized():
    if not _PIPELINE_CACHE["initialized"]:
        vdb, G, entity_index, llm = _init_vectorstore_and_graph()
        _PIPELINE_CACHE["vectorstore"] = vdb
        _PIPELINE_CACHE["graph"] = G
        _PIPELINE_CACHE["entity_index"] = entity_index
        _PIPELINE_CACHE["llm"] = llm
        _PIPELINE_CACHE["initialized"] = True



# ensure folder exists
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# Utility: write list of texts to files
def write_list_to_files(text_list, folder, prefix="item"):
    ensure_dir(folder)
    for i, text in enumerate(text_list):
        file_path = os.path.join(folder, f"{prefix}_{i}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

# Utility: read list of texts from folder
def read_list_from_folder(folder):
    if not os.path.exists(folder):
        return []
    files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".txt")])
    contents = []
    for f in files:
        with open(f, "r", encoding="utf-8") as file:
            contents.append(file.read())
    return contents


###########################################
# Split Source Documents Into Text Chunks #
###########################################
def split_documents_into_chunks(documents, chunk_size=600, overlap_size=100):
    folder = os.path.join(INPUT_DOCS_DIR, "1_chunks")
    existing_chunks = read_list_from_folder(folder)
    if existing_chunks:
        print(f"Loaded {len(existing_chunks)} chunks from {folder}")
        return existing_chunks

    chunks = []
    for document in documents:
        for i in range(0, len(document), chunk_size - overlap_size):
            chunk = document[i:i + chunk_size]
            chunks.append(chunk)
    write_list_to_files(chunks, folder, prefix="chunk")
    print(f"Wrote {len(chunks)} chunks to {folder}")
    return chunks

#####################################
# Public API: called from ui/app.py #
#####################################

def run_graphrag_pipeline(query: str) -> Tuple[str, dict, dict]:
    """
    1. Ensures documents are loaded, embedded, and KG is built.
    2. Runs hybrid_retrieve_and_answer() on the given query.
    3. Returns:
       - answer text
       - full KG as D3 tree JSON
       - subgraph (for this query) as D3 tree JSON
    """
    _ensure_pipeline_initialized()

    vdb = _PIPELINE_CACHE["vectorstore"]
    G = _PIPELINE_CACHE["graph"]
    entity_index = _PIPELINE_CACHE["entity_index"]
    llm = _PIPELINE_CACHE["llm"]

    answer, subgraph = hybrid_retrieve_and_answer(
        query,
        vdb,
        llm,
        G,
        entity_index,
        return_subgraph=True,
    )

    full_graph_json = graph_to_d3_tree(G, root_label="Knowledge Graph")
    subgraph_json = graph_to_d3_tree_from_subgraph(
        subgraph, root_label="Answer Subgraph"
    )

    return answer, full_graph_json, subgraph_json




def old_main():
    try:
        print("000")
        #loader = DirectoryLoader(DOCUMENTS, glob="*.txt", loader_cls=TextLoader) #TODO: use this instead!
        #documents = loader.load()
        chunks = split_documents_into_chunks(DOCUMENTS, chunk_size=600, overlap_size=100)
        documents = [
            Document(page_content=chunk, metadata={"source": f"chunk_{i}"})
            for i, chunk in enumerate(chunks)
        ]
        #print(documents)
        
        print("111")
        link_extractor = LinkExtractorTransformer(
            #link_extractors=[KnowledgeGraphLinkExtractor(llm=llmchat_client)] # deprecated
            link_extractors=[GLiNERLinkExtractor(
                labels=["Person", "Organization", "Location", "Event"] #, "Product", "Technology"]
            )]
        )

        print("222")
        document_chunk = link_extractor.transform_documents(documents)
        #print(document_chunk)
        
        print("333")
        """
        for doc in document_chunk:
            print(doc.metadata['links'])
            doc.metadata['links'] = ""
        vdb_store.add_documents(document_chunk)
        """
        
        print("444")
        # Build the graph and index
        #G = NetworkxEntityGraph()
        ##G.add_graph_documents()
        entity_index, G = build_entity_index_and_graph(document_chunk)

        print("555")
        vdb_store = None # added so that the next line doesn't err
        vdb_store.add_documents(document_chunk)

        print("666")
        query = "Who are Xiaomi's primary competitors and what kinds of investments are they making?"
        
        """
        # Vector text retrieval - this section works!
        results = vdb_store.similarity_search(query, k=3)
        #print(results)
        print("\n--- Similarity Search Results ---")
        for r in results:
            print(f"Score: {r.metadata.get('score', 'N/A')}\nContent: {r.page_content}\n")
        """
        
        llmchat_client = None # added so that the next line doesn't err
        answer = hybrid_retrieve_and_answer(query, vdb_store, llmchat_client, G, entity_index)

        print("Answer:\n", answer)

        # Visualize relevant subgraph from KG
        show_subgraph(G, list(entity_index.keys())[:2])
        
        
        """
        #graph_store = GraphVectorStore(
        graph_store = NetworkxGraphVectorStore(
            vectorstore=vdb_store,
            embedding_function=embeddings,
        )
        
        print("444")
        graph_store.add_documents(document_chunk)
        
        
        print("444.111")
        G = graph_store.graph
        
        print("444.222")
        plt.figure(figsize=(8,6))
        pos = nx.spring_layout(G, seed=42)
        
        nx.draw_networkx_nodes(G, pos, node_size=1200, node_color="skyblue", alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=1.5, edge_color="gray", alpha=0.6)
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")
        
        edge_labels = nx.get_edge_attributes(G, "label")
        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")
        
        plt.title("Graph Representation of Documents", fontsize=12)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        
        """
    except Exception as e:
        LOGGER.error("An error occurred: %s", e)
        
    """
    print("")
    print("")
    #retriever = vdb_store.as_retriever(search_type="mmr", search_kwargs={"k": 1, "fetch_k": 5})
    #retriever.invoke("Xiaomi competitors")
    results = vdb_store.similarity_search_by_vector(embedding=embeddings.embed_query("Xiaomi competitors"))
    for doc in results:
        print("")
        print(doc)
    """

"""
if __name__ == "__main__":
    old_main()
"""

if __name__ == "__main__":
    query = "Who are Xiaomi's primary competitors and what kinds of investments are they making?"
    demo_query = "Who are Xiaomi's competitors and what OS do they use?"
    ans, full_g, sub_g = run_graphrag_pipeline(demo_query)
    print(ans)

