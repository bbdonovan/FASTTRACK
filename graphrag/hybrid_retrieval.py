# -*- coding: utf-8 -*-
"""
Hybrid retrieval utilities:
 - Normalize metadata["links"] from GLiNER / link extractors
 - Build a NetworkX knowledge graph
    - file nodes
    - chunk nodes
    - entity nodes
- Hybrid retrieval: vector search + graph subgraph + LLM answer
- Graph JSON helpers for D3 visualisation
"""

import networkx as nx
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Iterable, Tuple, Any
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

import kuzu
from constants import KUZU_DB_PATH

##############################################################
# Normalize metadata["links"] into tags + Chroma-safe values #
##############################################################
def normalize_links_for_metadata(doc: Document) -> Tuple[Document, List[str]]:
    raw = doc.metadata.get("links")
    tags: List[str] = []

    if raw is None:
        return doc, tags

    # GLiNER + other link extractors often put a list of objects in metadata["links"].
    if isinstance(raw, list):
        tags = [str(getattr(x, "tag", x)) for x in raw]
        doc.metadata["links"] = ",".join(tags)
    elif isinstance(raw, str):
        tags = [t.strip() for t in raw.split(",") if t.strip()]
    else:
        tags = [str(raw)]
        doc.metadata["links"] = str(raw)
    return doc, tags


###################################################
# Build a file -> chunk -> entity index and graph #
###################################################
def build_entity_index_and_graph(docs: Iterable[Document]) -> Tuple[Dict[str, List[str]], nx.Graph]:
    """
    Expect each Document to represent a *chunk* and carry metadata:
      - file_id:   e.g. "file_0"
      - file_name: original filename, e.g. "file_0.txt"
      - node_id or chunk_id: unique chunk node id, e.g. "chunk_0_3"
      - links: entity/link objects or strings (set by GLiNERLinkExtractor)

    Build graph with:
      - file nodes:   kind="file"
      - chunk nodes:  kind="chunk"
      - entity nodes: kind="entity"

    Edges:
      - file --(contains)--> chunk
      - chunk --(mentions)--> entity
      - entity --(related)--> entity (co-occur in same chunk)
    """
    G = nx.Graph()
    entity_index: Dict[str, List[str]] = {}

    for i, doc in enumerate(docs):
        # Identify this chunk and its parent file
        file_id = doc.metadata.get("file_id")
        file_name = doc.metadata.get("file_name", file_id or "unknown_file")

        if file_id and not G.has_node(file_id):
            G.add_node(
                file_id,
                kind="file",
                title=file_name,
            )

        chunk_id = (
            doc.metadata.get("node_id")
            or doc.metadata.get("chunk_id")
            or doc.metadata.get("id")
            or doc.metadata.get("source")
            or f"chunk_{i}"
        )
        chunk_title = doc.metadata.get("chunk_label", chunk_id)

        if not G.has_node(chunk_id):
            G.add_node(
                chunk_id,
                kind="chunk",
                title=chunk_title,
            )

        # file -> chunk edge (contains)
        if file_id:
            G.add_edge(file_id, chunk_id, relation="contains")

        # Normalize links into tags and store in entity_index
        doc, tags = normalize_links_for_metadata(doc)
        entity_index[chunk_id] = tags

        # Connect chunk to entity nodes
        ent_nodes_for_chunk: List[str] = []
        for tag in tags:
            ent_node = f"ent::{tag}"
            if not G.has_node(ent_node):
                G.add_node(ent_node, kind="entity", label=tag)
            # chunk -> entity edge (mentions)
            G.add_edge(chunk_id, ent_node, relation="mentions")
            ent_nodes_for_chunk.append(ent_node)

        # Add entity-entity 'related' edges within this chunk (co-occurrence)
        for idx_a in range(len(ent_nodes_for_chunk)):
            for idx_b in range(idx_a + 1, len(ent_nodes_for_chunk)):
                a = ent_nodes_for_chunk[idx_a]
                b = ent_nodes_for_chunk[idx_b]
                if not G.has_edge(a, b):
                    G.add_edge(a, b, relation="related")

    return entity_index, G
def build_entity_index_and_graph_old(docs: Iterable[Document]) -> Tuple[Dict[str, List[str]], nx.Graph]:
    G = nx.Graph()
    entity_index: Dict[str, List[str]] = {}
    
    for i, doc in enumerate(docs):
        doc_id = doc.metadata.get("id") or doc.metadata.get("source") or f"doc_{i}"
        
        doc, tags = normalize_links_for_metadata(doc)
        entity_index[doc_id] = tags
        
        G.add_node(doc_id, kind="doc", title=doc.metadata.get("source", doc_id))
        
        for tag in tags:
            ent_node = f"ent::{tag}"
            G.add_node(ent_node, kind="entity", label=tag)
            G.add_edge(doc_id, ent_node, relation="mentions")
    
    return entity_index, G


###################################
# Kùzu-powered subgraph expansion #
###################################

# cache a single Kùzu connection per process
_KUZU_CONN = None

def _get_kuzu_conn() -> kuzu.Connection:
    """Get or create a kuzu connection."""
    global _KUZU_CONN
    if _KUZU_CONN is None:
        db = kuzu.Database(str(KUZU_DB_PATH))
        _KUZU_CONN = kuzu.Connection(db)
    return _KUZU_CONN


def kuzu_expand_subgraph_nodes(seed_ids: List[str], G: nx.Graph, hop_depth: int = 1) -> set:
    """
    Use Kuzu as the 'source of truth' to expand from the seed nodes.

    - Rely on the Kuzu schema created in load_data.write_graph_to_kuzu_from_chunks:
        NODE TABLE File   (id, name)
        NODE TABLE Chunk  (id, text, embedding)
        NODE TABLE Entity (id, name, label)

        REL TABLE CONTAINS (FROM File TO Chunk)
        REL TABLE MENTIONS (FROM Chunk TO Entity)
        REL TABLE RELATED  (FROM Entity TO Entity)

    - Use NetworkX G only to look up node "kind" and to build the final
      subgraph once we have the node ID set.
    """
    conn = _get_kuzu_conn()

    visited = set(seed_ids)
    frontier = set(seed_ids)

    for _ in range(hop_depth):
        next_frontier = set()

        for node_id in frontier:
            if node_id not in G:
                # If for some reason Kùzu has more than G, fall back to G neighbors
                for nbr in G.neighbors(node_id):
                    if nbr not in visited:
                        visited.add(nbr)
                        next_frontier.add(nbr)
                continue

            kind = G.nodes[node_id].get("kind")

            # File node neighbors: File -> Chunk via CONTAINS
            if kind == "file":
                q = """
                MATCH (f:File {id: $id})-[:CONTAINS]->(c:Chunk)
                RETURN c.id AS id
                """
                result = conn.execute(q, parameters={"id": node_id})
                while result.has_next():
                    row = result.get_next()
                    neigh_id = row[0]
                    if neigh_id not in visited:
                        visited.add(neigh_id)
                        next_frontier.add(neigh_id)

            # Chunk node neighbors: File via CONTAINS, Entity via MENTIONS
            elif kind == "chunk":
                # Files containing this chunk
                q_files = """
                MATCH (f:File)-[:CONTAINS]->(c:Chunk {id: $id})
                RETURN f.id AS id
                """
                res_f = conn.execute(q_files, parameters={"id": node_id})
                while res_f.has_next():
                    row = res_f.get_next()
                    neigh_id = row[0]
                    if neigh_id not in visited:
                        visited.add(neigh_id)
                        next_frontier.add(neigh_id)

                # Entities mentioned by this chunk
                q_ents = """
                MATCH (c:Chunk {id: $id})-[:MENTIONS]->(e:Entity)
                RETURN e.id AS id
                """
                res_e = conn.execute(q_ents, parameters={"id": node_id})
                while res_e.has_next():
                    row = res_e.get_next()
                    neigh_id = row[0]
                    if neigh_id not in visited:
                        visited.add(neigh_id)
                        next_frontier.add(neigh_id)

            # Entity node neighbors: Chunks via MENTIONS, Entities via RELATED
            elif kind == "entity":
                # Chunks that mention this entity
                q_chunks = """
                MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {id: $id})
                RETURN c.id AS id
                """
                res_c = conn.execute(q_chunks, parameters={"id": node_id})
                while res_c.has_next():
                    row = res_c.get_next()
                    neigh_id = row[0]
                    if neigh_id not in visited:
                        visited.add(neigh_id)
                        next_frontier.add(neigh_id)

                # Related entities
                q_rel = """
                MATCH (e1:Entity {id: $id})-[:RELATED]->(e2:Entity)
                RETURN e2.id AS id
                """
                res_r = conn.execute(q_rel, parameters={"id": node_id})
                while res_r.has_next():
                    row = res_r.get_next()
                    neigh_id = row[0]
                    if neigh_id not in visited:
                        visited.add(neigh_id)
                        next_frontier.add(neigh_id)

            else:
                # Fallback: use NetworkX adjacency if kind is unknown
                for nbr in G.neighbors(node_id):
                    if nbr not in visited:
                        visited.add(nbr)
                        next_frontier.add(nbr)

        frontier = next_frontier

    return visited


###########################################################
# Hybrid retrieval: vector search + subgraph + LLM answer #
###########################################################
def hybrid_retrieve_and_answer(query: str, vectorstore: Chroma, llm: ChatOpenAI,
    G: nx.Graph, entity_index: Dict[str, List[str]], *,
    k_docs: int=4, hop_depth: int=1, return_subgraph: bool=False): # -> str:

    """
    - Seeds for the graph subgraph use chunk-level node_id / chunk_id
      (so the subgraph is rooted at relevant chunks, not whole-doc nodes).
    - Treat 'file' and 'chunk' nodes as doc-like when summarizing the graph.
    - Subgraph expansion is done via Kuzu instead of pure NetworkX BFS.
    """
    
    # 1 - Vector Similarity (returns chunk Documents)
    top_docs: List[Document] = vectorstore.similarity_search(query, k=k_docs)

    seed_doc_ids: List[str] = []
    for i, d in enumerate(top_docs):
        doc_id = (
            d.metadata.get("node_id")
            or d.metadata.get("chunk_id")
            or d.metadata.get("id")
            or d.metadata.get("source")
            or f"chunk_{i}"
        )
        seed_doc_ids.append(doc_id)

    # 2 - Subgraph expansion from these chunk seeds
    sub_nodes = set(seed_doc_ids)
    frontier = set(seed_doc_ids)
    for _ in range(hop_depth):
        next_frontier = set()
        for n in frontier:
            if n in G:
                next_frontier.update(G.neighbors(n))
        sub_nodes.update(next_frontier)
        frontier = next_frontier

    SG = G.subgraph(sub_nodes).copy()

    # 3 - Build a lightweight textual summary of the subgraph
    graph_summary_lines: List[str] = []
    for n, data in SG.nodes(data=True):
        kind = data.get("kind")
        if kind in ("file", "chunk"):
            graph_summary_lines.append(
                f"- {kind.upper()} {n}: connected to {len(list(SG.neighbors(n)))} neighbors"
            )
        elif kind == "entity":
            label = data.get("label", n)
            nbr_docs = [
                x
                for x in SG.neighbors(n)
                if SG.nodes[x].get("kind") in ("file", "chunk")
            ]
            graph_summary_lines.append(
                f"- ENTITY '{label}': linked to {len(nbr_docs)} file/chunk nodes -> {nbr_docs}"
            )

    graph_text = "\n".join(graph_summary_lines)

    # 4 - Compose prompt for the LLM
    passage_block = "\n\n".join(
        [f"[Chunk {i+1}] {d.page_content}" for i, d in enumerate(top_docs)]
    )

    prompt = f"""You are answering a user query using:
1) Top {k_docs} CHUNKS from a vector search over a document collection.
2) A small knowledge subgraph of files, chunks, and entities.

User Query:
{query}

Vector Passages (chunks):
{passage_block}

Knowledge Subgraph:
{graph_text}

Provide a clear, concise answer based ONLY on this information.
If the context is insufficient, say so explicitly.
"""

    response = llm.invoke(prompt)
    answer_text = getattr(response, "content", str(response))

    if return_subgraph:
        return answer_text, SG

    return answer_text


##########################
# Subgraph Visualisation #
##########################
def show_subgraph(G: nx.Graph, seed_doc_ids: List[str], hop_depth: int = 1):
    
    sub_nodes = set(seed_doc_ids)
    frontier = set(seed_doc_ids)
    for _ in range(hop_depth):
        nxt = set()
        for n in frontier:
            nxt.update(G.neighbors(n))
        sub_nodes.update(nxt)
        frontier = nxt
    SG = G.subgraph(sub_nodes).copy()
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(SG, seed=42)
    nx.draw_networkx_nodes(SG, pos, node_size=800, node_color="skyblue")
    nx.draw_networkx_edges(SG, pos, alpha=0.5)
    nx.draw_networkx_labels(SG, pos, font_size=8)
    plt.title("Subgraph for Hybrid Retrieval")
    plt.axis("off")
    plt.show()

#################################################
# Convert NetworkX graph to force-directed JSON #
#################################################
def generate_graph_json(G: nx.Graph) -> Dict[str, Any]:
    """
    Convert a NetworkX graph to a D3-friendly force-directed format:
      {
        "nodes": [
          {"id": "file_0", "group": 1, "label": "file_0.txt", "kind": "file"},
          {"id": "chunk_0_0", "group": 2, "label": "chunk_0_0", "kind": "chunk"},
          {"id": "ent::Xiaomi", "group": 3, "label": "Xiaomi", "kind": "entity"},
          ...
        ],
        "links": [
          {"source": "file_0", "target": "chunk_0_0", "value": 1, "relation": "contains"},
          {"source": "chunk_0_0", "target": "ent::Xiaomi", "value": 1, "relation": "mentions"},
          {"source": "ent::Xiaomi", "target": "ent::Android", "value": 1, "relation": "related"},
          ...
        ]
      }

    The frontend can then set different link distances for:
      - relation === "contains"  (short)
      - relation === "mentions"  (medium)
      - relation === "related"   (longer)
    """
    nodes = []
    for n, data in G.nodes(data=True):
        kind = data.get("kind", "other")
        if kind == "file":
            group = 1
        elif kind == "chunk":
            group = 2
        elif kind == "entity":
            group = 3
        else:
            group = 4

        nodes.append(
            {
                "id": n,
                "group": group,
                "label": data.get("title") or data.get("label", n),
                "kind": kind,
            }
        )

    links = []
    for u, v, data in G.edges(data=True):
        relation = data.get("relation", "related")
        links.append(
            {
                "source": u,
                "target": v,
                "value": 1,
                "relation": relation,
            }
        )

    return {"nodes": nodes, "links": links}


############################################################
# Convert a NetworkX graph to a D3-friendly tree structure #
############################################################

def graph_to_d3_tree(G: nx.Graph, *, root_label: str = "Knowledge Graph") -> dict:
    """
    Simple hierarchical view:
      root -> doc nodes -> entity nodes
    """
    root = {"name": root_label, "children": []}
    doc_nodes = [
        (n, d) for n, d in G.nodes(data=True) if d.get("kind") == "doc"
    ]

    for doc_id, data in doc_nodes:
        doc_entry = {
            "name": data.get("title", doc_id),
            "children": [],
        }
        for nbr in G.neighbors(doc_id):
            nbr_data = G.nodes[nbr]
            if nbr_data.get("kind") == "entity":
                doc_entry["children"].append(
                    {"name": nbr_data.get("label", nbr)}
                )
        root["children"].append(doc_entry)

    return root


def graph_to_d3_tree_from_subgraph(SG: nx.Graph, *, root_label: str = "Answer Subgraph") -> dict:
    return graph_to_d3_tree(SG, root_label=root_label)

