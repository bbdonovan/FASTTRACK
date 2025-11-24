# -*- coding: utf-8 -*-
"""
loads, processes, and visualizes documents for the GraphRAG pipeline used by the Flask UI.

- Load .txt docs from INPUT_FOLDER.
- Split docs into overlapping chunks (default: 500 tokens ~ chars, 100 overlap).
- Run GLiNERLinkExtractor to populate metadata["links"] per chunk.
- Build a NetworkX KG with:
    * file nodes
    * chunk nodes
    * entity nodes
    * 'contains' / 'mentions' / 'related' edges
- Store chunk docs in Chroma for vector search.
- Expose run_graphrag_pipeline(query, ...) returning:
    * answer text
    * full document tree JSON ("sources" -> source -> files)
    * same doc tree as "subgraph" placeholder
    * full KG JSON (for #kg-container)
    * KG subgraph JSON for the specific query
- Persist full graph to Kùzu with File/Chunk/Entity nodes and CONTAINS/MENTIONS/RELATED edges.
"""

import os
from pathlib import Path
import warnings
from typing import List, Tuple, Dict, Any, Optional

import networkx as nx
import matplotlib.pyplot as plt
import json

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_chroma import Chroma
import chromadb
from chromadb.config import Settings

from langchain_community.graph_vectorstores.extractors import (
    LinkExtractorTransformer,
    GLiNERLinkExtractor,
)

from openai import OpenAI

from util.config import LOGGER, DB, NODE_TABLE, BASE_URL, MODEL_NAME, INPUT_DOCS_DIR

from constants import DOCUMENTS, DOCUMENT_FILE_NAMES
from hybrid_retrieval import (
    build_entity_index_and_graph,
    hybrid_retrieve_and_answer,
    generate_graph_json,
)

import kuzu
from constants import KUZU_DB_DIRECTORY, KUZU_DB_PATH


#########
# SETUP #
#########

# Suppress all of the Langchain beta and other warnings
warnings.filterwarnings("ignore", lineno=0)
load_dotenv()

# Cache so we don't rebuild between requests
_PIPELINE_CACHE = {
    "initialized": False,
    "vectorstore": None,
    "graph": None,
    "entity_index": None,
    "llm": None,
}

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()

BASE_DIR = Path(__file__).resolve().parent
if DB == "chroma":
    from util.config import CHROMA_PERSIST_DIRECTORY, CHROMA_COLLECTION_NAME
    docs_folder_name = INPUT_DOCS_DIR or os.getenv("INPUT_FOLDER", "input_docs")
    INPUT_DOCS_PATH = BASE_DIR / docs_folder_name

    CHROMA_PERSIST_DIRECTORY = str(BASE_DIR / CHROMA_PERSIST_DIRECTORY)
else:
    INPUT_DOCS_PATH = BASE_DIR  # placeholder to support other DBs

#######################
# Utilities & Helpers #
#######################

def _load_input_documents_old() -> List[Document]:
    # You already prepare DOCUMENTS in constants.py based on INPUT_FOLDER (from .env).  :contentReference[oaicite:2]{index=2}
    docs: List[Document] = []
    for i, text in enumerate(DOCUMENTS):
        docs.append(Document(page_content=text, metadata={"source": f"file_{i}.txt"}))
    LOGGER.debug("Loaded %d raw texts from constants.DOCUMENTS", len(docs))
    return docs
def _load_input_documents() -> Tuple[List[Document], List[str]]:
    """
    Load input .txt documents as langchain Documents.

    Prefer reading from the filesystem (INPUT_DOCS_PATH).
    If that directory isn't available (or empty), fall back to
    DOCUMENTS / DOCUMENT_FILE_NAMES from constants.py.
    """
    docs: List[Document] = []
    file_names: List[str] = []

    if INPUT_DOCS_PATH.exists():
        for path in sorted(INPUT_DOCS_PATH.glob("*.txt")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            docs.append(Document(page_content=text, metadata={"source": path.name}))
            file_names.append(path.name)
    else:
        # Fallback to constants
        for idx, txt in enumerate(DOCUMENTS):
            name = DOCUMENT_FILE_NAMES[idx] if idx < len(DOCUMENT_FILE_NAMES) else f"file_{idx}.txt"
            docs.append(Document(page_content=txt, metadata={"source": name}))
            file_names.append(name)

    return docs, file_names

def _chunk_documents(docs: List[Document], chunk_size: int = 500, overlap_size: int = 100) -> List[Document]:
    """
    Simple character-based chunking of each doc into overlapping segments.

    - chunk_size:    approx tokens-per-chunk (for now treated as chars)
    - overlap_size:  approx overlapping chars between chunks

    Returns a list of chunk-level Documents with metadata:
      - file_id
      - file_name
      - chunk_id / node_id
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap_size < 0 or overlap_size >= chunk_size:
        raise ValueError("overlap_size must be >= 0 and < chunk_size")

    chunk_docs: List[Document] = []
    step = chunk_size - overlap_size

    for file_idx, doc in enumerate(docs):
        text = doc.page_content
        file_name = doc.metadata.get("source", f"file_{file_idx}.txt")
        file_id = f"file_{file_idx}"

        start = 0
        chunk_idx = 0
        while start < len(text):
            chunk_text = text[start : start + chunk_size]
            node_id = f"chunk_{file_idx}_{chunk_idx}"

            chunk_docs.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "source": file_name,
                        "file_id": file_id,
                        "file_name": file_name,
                        "chunk_id": node_id,
                        "node_id": node_id,
                    },
                )
            )

            chunk_idx += 1
            start += step

    LOGGER.debug("Created %d chunk documents", len(chunk_docs))
    return chunk_docs

#TODO: the following will only work for single-source doc corpuses
def _build_document_tree(file_names: List[str], source_label: str = "doc source") -> dict:
    """
    Build simple tree for the 'Project Document Visualization' section:
      sources
        └── <source_label>  (e.g., "CNN")
              ├── file_0.txt
              ├── file_1.txt
              └── ...

    Intentionally stop at the file layer here (no chunk nodes).
    """
    return {
        "name": "sources",
        "children": [
            {
                "name": source_label,
                "children": [{"name": fn} for fn in file_names],
            }
        ],
    }


def _init_vectorstore_and_graph_old():
    docs = _load_input_documents()

    # Embeddings + Chroma
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vdb_client = chromadb.Client(Settings(persist_directory=CHROMA_PERSIST_DIRECTORY))
    collection = vdb_client.get_or_create_collection(CHROMA_COLLECTION_NAME)
    vdb_store = Chroma(client=vdb_client, collection_name=CHROMA_COLLECTION_NAME, embedding_function=embeddings)

    # Only add if collection is empty
    if not collection.count():
        vdb_store.add_documents(docs)

    # Build entity index + KG (simple doc-entity graph)
    entity_index, G = build_entity_index_and_graph(docs)

    # LLM for answering
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=BASE_URL or None,
        model=MODEL_NAME or "gpt-4o-mini",
        temperature=0.1,
    )
    return vdb_store, G, entity_index, llm
def _init_vectorstore_and_graph(chunk_size: int = 500, overlap_size: int = 100) -> Tuple[Chroma, "nx.Graph", Dict[str, List[str]], ChatOpenAI, List[str]]:
    """
    1. Load docs and filenames.
    2. Chunk docs into overlapping segments.
    3. Extract entities with GLiNERLinkExtractor.
    4. Index chunks in Chroma.
    5. Build file→chunk→entity graph.
    6. Initialize ChatOpenAI client.

    Returns:
      - Chroma vectorstore (on chunk docs)
      - NetworkX graph G
      - entity_index (chunk_id -> list of tags)
      - LLM client
      - file_names (for document tree)
    """
    from typing import List as _List, Dict as _Dict  # local alias to avoid confusion

    # 1. Load full documents
    docs, file_names = _load_input_documents()
    LOGGER.info("Loaded %d documents for GraphRAG pipeline", len(docs))

    # 2. Chunk into overlapping segments
    chunk_docs = _chunk_documents(docs, chunk_size=chunk_size, overlap_size=overlap_size)

    # 3. Extract entities / links
    link_extractor = LinkExtractorTransformer(
        link_extractors=[
            GLiNERLinkExtractor(
                labels=[
                    "Person",
                    "Organization",
                    "Location",
                    #"Product", # don't think this is a valid entity type
                    #"Technology", # don't think this is a valid entity type
                    #"Company", # don't think this is a valid entity type
                    "Event",
                ]
            )
        ]
    )
    LOGGER.info("Running GLiNERLinkExtractor over %d chunks...", len(chunk_docs))
    chunk_docs = link_extractor.transform_documents(chunk_docs)
    
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    # Persist chunk + entity graph into Kùzu (with embeddings)
    LOGGER.info("Writing graph to Kùzu (chunks + entities + embeddings)...")
    write_graph_to_kuzu_from_chunks(chunk_docs, embeddings)


    # Normalize metadata["links"] BEFORE adding to Chroma
    from hybrid_retrieval import normalize_links_for_metadata    
    normalized_chunks = []
    for doc in chunk_docs:
        doc, _ = normalize_links_for_metadata(doc)
        normalized_chunks.append(doc)
    chunk_docs = normalized_chunks

    # 4. Embeddings + Chroma
    #embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vdb_client = chromadb.Client(Settings(persist_directory=CHROMA_PERSIST_DIRECTORY))
    collection = vdb_client.get_or_create_collection(CHROMA_COLLECTION_NAME)
    vdb_store = Chroma(
        client=vdb_client,
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
    )

    # To keep things in sync, clear any previous contents then add current chunk docs
    if collection.count():
        LOGGER.info("Clearing existing Chroma collection '%s' before reloading.", CHROMA_COLLECTION_NAME)
        collection.delete(where={})

    LOGGER.info("Adding %d chunk documents to Chroma...", len(chunk_docs))
    vdb_store.add_documents(chunk_docs)

    # 5. Build entity index + KG
    entity_index, G = build_entity_index_and_graph(chunk_docs)
    LOGGER.info(
        "Built KG with %d nodes and %d edges",
        len(G.nodes),
        len(G.edges),
    )

    # 6. LLM for answering
    llmchat_client = (
        ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model="gpt-4o-mini",
            temperature=0.1,
        )
        if not BASE_URL
        else ChatOpenAI(
            base_url=BASE_URL,
            api_key=OPENAI_API_KEY,
            model=MODEL_NAME,
            temperature=0.1,
        )
    )

    return vdb_store, G, entity_index, llmchat_client, file_names


def _ensure_pipeline_initialized_old():
    if not _PIPELINE_CACHE["initialized"]:
        vdb, G, entity_index, llm = _init_vectorstore_and_graph()
        _PIPELINE_CACHE.update(
            {
                "initialized": True,
                "vectorstore": vdb,
                "graph": G,
                "entity_index": entity_index,
                "llm": llm,
            }
        )
def _ensure_pipeline_initialized(chunk_size: int = 500, overlap_size: int = 100):
    """
    Ensure the Chroma vectorstore + KG + LLM are initialized once.

    NOTE: chunk_size / overlap_size are only honored on the first call.
    """
    if not _PIPELINE_CACHE["initialized"]:
        (
            vdb,
            G,
            entity_index,
            llm,
            file_names,
        ) = _init_vectorstore_and_graph(chunk_size=chunk_size, overlap_size=overlap_size)

        _PIPELINE_CACHE["vectorstore"] = vdb
        _PIPELINE_CACHE["graph"] = G
        _PIPELINE_CACHE["entity_index"] = entity_index
        _PIPELINE_CACHE["llm"] = llm
        _PIPELINE_CACHE["file_names"] = file_names
        _PIPELINE_CACHE["initialized"] = True


####################################################
# Kùzu graph persistence – files, chunks, entities #
####################################################
def write_graph_to_kuzu_from_chunks(chunk_docs: List[Document], embeddings_model: OpenAIEmbeddings) -> None:
    """
    Persist the current file + chunk + entity graph into a Kùzu database.

    NODE TABLES:
      - File(
          id   STRING,   -- e.g. "file_0"
          name STRING,   -- e.g. "file_0.txt"
          PRIMARY KEY (id)
        )
      - Chunk(
          id        STRING,
          text      STRING,
          embedding DOUBLE[],
          PRIMARY KEY (id)
        )
      - Entity(
          id    STRING,  -- e.g. "Organization:Xiaomi"
          name  STRING,  -- "Xiaomi"
          label STRING,  -- "Organization"
          PRIMARY KEY (id)
        )

    REL TABLES:
      - CONTAINS(FROM File TO Chunk)
      - MENTIONS(FROM Chunk TO Entity)
      - RELATED(FROM Entity TO Entity)
    """
    #logger = LOGGER.getLogger(__name__)

    # Ensure directory exists
    KUZU_DB_DIRECTORY.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Initializing Kùzu database at %s", KUZU_DB_DIRECTORY)
    db = kuzu.Database(str(KUZU_DB_PATH))
    conn = kuzu.Connection(db)

    # --- Schema: node + rel tables ---
    LOGGER.debug("Ensuring Kùzu schema...")

    conn.execute(
        """
        CREATE NODE TABLE IF NOT EXISTS File (
            id STRING,
            name STRING,
            PRIMARY KEY (id)
        );
        """
    )

    conn.execute(
        """
        CREATE NODE TABLE IF NOT EXISTS Chunk (
            id STRING,
            text STRING,
            embedding DOUBLE[],
            PRIMARY KEY (id)
        );
        """
    )

    conn.execute(
        """
        CREATE NODE TABLE IF NOT EXISTS Entity (
            id STRING,
            name STRING,
            label STRING,
            PRIMARY KEY (id)
        );
        """
    )

    conn.execute(
        """
        CREATE REL TABLE IF NOT EXISTS CONTAINS (
            FROM File TO Chunk
        );
        """
    )

    conn.execute(
        """
        CREATE REL TABLE IF NOT EXISTS MENTIONS (
            FROM Chunk TO Entity
        );
        """
    )

    conn.execute(
        """
        CREATE REL TABLE IF NOT EXISTS RELATED (
            FROM Entity TO Entity
        );
        """
    )

    # For now, clear previous contents for a fresh run
    #TODO: add versioning instead of full wipe
    LOGGER.info("Clearing previous Kùzu graph contents...")
    conn.execute("MATCH ()-[m:MENTIONS]->() DELETE m;")
    conn.execute("MATCH ()-[m:CONTAINS]->() DELETE m;")
    conn.execute("MATCH ()-[m:RELATED]->() DELETE m;")
    conn.execute("MATCH (c:Chunk) DELETE c;")
    conn.execute("MATCH (e:Entity) DELETE e;")
    conn.execute("MATCH (f:File) DELETE f;")

    # Compute embeddings for chunks
    texts = [doc.page_content for doc in chunk_docs]
    LOGGER.info("Computing embeddings for %d chunks to store in Kùzu...", len(texts))
    chunk_embeddings = embeddings_model.embed_documents(texts)

    # Ingest files, chunks, entities, relationships
    LOGGER.info("Writing files + chunks + entities to Kùzu...")
    for idx, (doc, emb) in enumerate(zip(chunk_docs, chunk_embeddings)):
        # Chunk ID: try to reuse existing metadata; fall back to a stable synthetic ID
        chunk_id = str(
            doc.metadata.get("chunk_id")
            or doc.metadata.get("id")
            or doc.metadata.get("source")
            or f"chunk_{idx}"
        )

        file_id = str(doc.metadata.get("file_id") or "file_unknown")
        file_name = str(doc.metadata.get("file_name") or file_id)

        # Create/update file node
        conn.execute(
            """
            MERGE (f:File {id: $id})
            SET f.name = $name
            """,
            parameters={"id": file_id, "name": file_name},
        )

        # Create/update chunk node
        conn.execute(
            """
            MERGE (c:Chunk {id: $id})
            SET c.text = $text,
                c.embedding = $embedding
            """,
            parameters={"id": chunk_id, "text": doc.page_content, "embedding": emb},
        )

        # Create/update CONTAINS relationship File -> Chunk
        conn.execute(
            """
            MATCH (f:File {id: $file_id}),
                  (c:Chunk {id: $chunk_id})
            MERGE (f)-[:CONTAINS]->(c)
            """,
            parameters={"file_id": file_id, "chunk_id": chunk_id},
        )

        # Entities come from GLiNER: doc.metadata["links"] is a list of Link(...)
        raw_links = doc.metadata.get("links") or []
        ent_ids_for_chunk: List[str] = []

        for link in raw_links:
            # Defensive: support both Link objects and plain dicts if needed
            kind = getattr(link, "kind", None) or getattr(link, "type", None)
            tag = getattr(link, "tag", None) or getattr(link, "name", None)

            if not kind or not tag:
                continue

            # kind looks like "entity:Organization" -> label = "Organization"
            if ":" in kind:
                _, label = kind.split(":", 1)
            else:
                label = kind

            entity_name = str(tag)
            # Build a stable entity ID, suitable as PK
            entity_id = f"{label}:{entity_name}"

            ent_ids_for_chunk.append(entity_id)

            # Create/update entity node
            conn.execute(
                """
                MERGE (e:Entity {id: $id})
                SET e.name = $name,
                    e.label = $label
                """,
                parameters={
                    "id": entity_id,
                    "name": entity_name,
                    "label": label,
                },
            )

            # Create/update MENTIONS relationship Chunk -> Entity
            conn.execute(
                """
                MATCH (c:Chunk {id: $chunk_id}),
                      (e:Entity {id: $entity_id})
                MERGE (c)-[:MENTIONS]->(e)
                """,
                parameters={"chunk_id": chunk_id, "entity_id": entity_id},
            )

        # Add RELATED edges between all entity pairs within this chunk
        for i in range(len(ent_ids_for_chunk)):
            for j in range(i + 1, len(ent_ids_for_chunk)):
                e1 = ent_ids_for_chunk[i]
                e2 = ent_ids_for_chunk[j]
                # Make edges in both directions so traversal is easy
                conn.execute(
                    """
                    MATCH (e1:Entity {id: $id1}),
                          (e2:Entity {id: $id2})
                    MERGE (e1)-[:RELATED]->(e2)
                    """,
                    parameters={"id1": e1, "id2": e2},
                )
                conn.execute(
                    """
                    MATCH (e1:Entity {id: $id1}),
                          (e2:Entity {id: $id2})
                    MERGE (e2)-[:RELATED]->(e1)
                    """,
                    parameters={"id1": e1, "id2": e2},
                )

    LOGGER.info("Finished writing graph to Kùzu: %d chunks", len(chunk_docs))



#####################################
# public API: called from ui/app.py #
#####################################
#TODO: only works for single doc source
def run_graphrag_pipeline(query: str, *, source_label: str = "source", chunk_size: int = 500,
    overlap_size: int = 100) -> Tuple[str, dict, dict, dict, dict]:
    """
    Public entry point for the Flask UI.

    1. Ensures documents are loaded, chunked, embedded, and KG is built.
    2. Runs hybrid_retrieve_and_answer() on the given query.
    3. Returns:
       - answer text                      (for "Gemini Response" panel)
       - document_tree_full (JSON)        (for 'Project Document Visualization')
       - document_tree_sub (JSON)         (currently same as full)
       - kg_full_json (JSON)              (for full KG in 'Project Knowledge Graph')
       - kg_subgraph_json (JSON)          (for subgraph relevant to this query)
    """
    _ensure_pipeline_initialized(chunk_size=chunk_size, overlap_size=overlap_size)

    vdb = _PIPELINE_CACHE["vectorstore"]
    G = _PIPELINE_CACHE["graph"]
    entity_index = _PIPELINE_CACHE["entity_index"]
    llm = _PIPELINE_CACHE["llm"]
    file_names = _PIPELINE_CACHE["file_names"] or []

    # 1. Hybrid retrieval over chunks + KG
    answer, subgraph = hybrid_retrieve_and_answer(
        query,
        vdb,
        llm,
        G,
        entity_index,
        return_subgraph=True,
    )

    # 2. Document tree: sources -> <source_label> -> files
    document_tree_full = _build_document_tree(file_names, source_label=source_label)
    document_tree_sub = document_tree_full  # no sub-tree concept for docs (by design)

    # 3. KG JSONs for D3 force-directed graph
    kg_full_json = generate_graph_json(G)
    kg_subgraph_json = generate_graph_json(subgraph)

    return (
        answer,
        document_tree_full,
        document_tree_sub,
        kg_full_json,
        kg_subgraph_json,
    )


if __name__ == "__main__":
    demo_query = "Tell me about Xiaomi cellphones in the context of China, their Chinese competitors and investments in future technology?"
    ans, *_ = run_graphrag_pipeline(demo_query)
    print(ans)

    """
    Some good queries:
        What OS does Xiaomi use in its cellphones? How does this compare to its competitors?
        Tell me about Xiaomi's investments in new technology, especially as it relates to next generation cellphones. Compare these investments to those of Huawei and other competitors.
        Given only the source info supplied, what are some open questions, not explicitly answered in the provided sources, that could potentially be answered with reasonable confidence by inferring likely relationships within and between entities in the provided sources?
    """





