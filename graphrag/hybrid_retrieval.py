# -*- coding: utf-8 -*-
"""
"""

import networkx as nx
from typing import Dict, List, Iterable, Tuple
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI


######################
# Normalize metadata #
######################
def normalize_links_for_metadata(doc: Document) -> Tuple[Document, List[str]]:
    raw = doc.metadata.get("links")
    tags: List[str] = []
    if raw is None:
        return doc, tags
    if isinstance(raw, list):
        tags = [str(getattr(x, "tag", x)) for x in raw]
        doc.metadata["links"] = ",".join(tags)
    elif isinstance(raw, str):
        tags = [t.strip() for t in raw.split(",") if t.strip()]
    else:
        tags = [str(raw)]
        doc.metadata["links"] = str(raw)
    return doc, tags


#########################################
# Build a doc -> entity index and graph #
#########################################
def build_entity_index_and_graph(docs: Iterable[Document]) -> Tuple[Dict[str, List[str]], nx.Graph]:
    G = nx.Graph()
    entity_index: Dict[str, List[str]] = {}
    for i, doc in enumerate(docs):
        doc_id = doc.metadata.get("id") or doc.metadata.get("source") or f"doc_{i}"
        _, tags = normalize_links_for_metadata(doc)
        entity_index[doc_id] = tags
        G.add_node(doc_id, kind="doc", title=doc.metadata.get("source", doc_id))
        for tag in tags:
            ent_node = f"ent::{tag}"
            G.add_node(ent_node, kind="entity", label=tag)
            G.add_edge(doc_id, ent_node, relation="mentions")
    return entity_index, G


####################
# Hybrid retrieval #
####################
def hybrid_retrieve_and_answer(query: str, vectorstore: Chroma, llm: ChatOpenAI,
    G: nx.Graph, entity_index: Dict[str, List[str]], *,
    k_docs: int = 4, hop_depth: int = 1) -> str:

    top_docs = vectorstore.similarity_search(query, k=k_docs)
    seed_ids = [d.metadata.get("id") or d.metadata.get("source") or f"doc_{i}" for i, d in enumerate(top_docs)]

    sub_nodes = set(seed_ids)
    frontier = set(seed_ids)
    for _ in range(hop_depth):
        next_frontier = set()
        for n in frontier:
            next_frontier.update(G.neighbors(n))
        sub_nodes.update(next_frontier)
        frontier = next_frontier

    SG = G.subgraph(sub_nodes).copy()

    graph_summary = []
    for n, data in SG.nodes(data=True):
        if data.get("kind") == "doc":
            graph_summary.append(f"- DOC {n}: connected to {len(list(SG.neighbors(n)))} entities")
        elif data.get("kind") == "entity":
            label = data.get("label", n)
            nbr_docs = [x for x in SG.neighbors(n) if SG.nodes[x].get("kind") == "doc"]
            graph_summary.append(f"- ENTITY '{label}': mentioned in {len(nbr_docs)} docs -> {nbr_docs}")

    prompt = f"""
Answer the following user query using both:
1. The top {k_docs} similar documents, and
2. The following subgraph of entities and document relationships.

Query:
{query}

Documents:
{[d.page_content for d in top_docs]}

Graph:
{chr(10).join(graph_summary)}
"""

    response = llm.invoke(prompt)
    return getattr(response, "content", str(response))


##########################
# Subgraph Visualisation #
##########################
def show_subgraph(G: nx.Graph, seed_doc_ids: List[str], hop_depth: int = 1):
    import matplotlib.pyplot as plt
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
