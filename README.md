# FASTTRACK
FASTTRACK AI Research Assistant


To run the GraphRAG component:
> conda activate myenv  
> pip install -r requirements.txt  
> python load_data.py  

This will do the following:  
    - read the documents specified in .env > INPUT_FOLDER (currently set to `input_docs_cnnxiaomi`)  
    - chunk the documents up into small, overlapping pieces  
    - extract entities and links from the document chunks  
    - build a simple knowledge graph (KG) from the extracted entities and links  
    - store the document chunks and extracted metadata in a Chroma vector database  
    - launch a hybrid graph/vector search, based on a query  
    - use an LLM to answer the query, based on the retrieved entities and document chunks  
    - visualize the subgraph of the KG relevant to the query  
