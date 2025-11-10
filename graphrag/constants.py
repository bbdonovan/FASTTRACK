# -*- coding: utf-8 -*-
"""
"""

import os
from pathlib import Path
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent

load_dotenv()
#BASE_DIR = os.getenv("INPUT_FOLDER")
INPUT_DOCS_DIR = BASE_DIR / os.getenv("INPUT_FOLDER")


# Function to read the content of each document from the example_text directory
def read_documents_from_files():
    documents = []
    directory = INPUT_DOCS_DIR
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                documents.append(file.read())
    return documents


# Read documents and store them in the DOCUMENTS list
DOCUMENTS = read_documents_from_files()