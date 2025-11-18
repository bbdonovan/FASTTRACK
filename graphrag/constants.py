# -*- coding: utf-8 -*-
"""
Constants and simple helpers for reading input documents
"""

import os
from pathlib import Path
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent

load_dotenv()
#BASE_DIR = os.getenv("INPUT_FOLDER")

INPUT_FOLDER = os.getenv("INPUT_FOLDER", "input_docs")

INPUT_DOCS_DIR = BASE_DIR / INPUT_FOLDER

KUZU_DB_DIRECTORY = BASE_DIR / os.getenv("KUZU_PERSIST_DIRECTORY", "kuzu_db")

# Function to read the content of each document from the example_text directory
def read_documents_from_files_old():
    documents = []
    directory = INPUT_DOCS_DIR
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                documents.append(file.read())
    return documents

def read_documents_from_files():
    """
    Return both the document texts and their filenames.
    """
    texts = []
    file_names = []

    if not INPUT_DOCS_DIR.exists():
        return texts, file_names

    for path in sorted(INPUT_DOCS_DIR.glob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        texts.append(text)
        file_names.append(path.name)

    return texts, file_names



# Read documents and store them in the DOCUMENTS list
#DOCUMENTS = read_documents_from_files()
DOCUMENTS, DOCUMENT_FILE_NAMES = read_documents_from_files()

