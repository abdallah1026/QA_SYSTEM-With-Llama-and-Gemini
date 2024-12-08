from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.gemini import GeminiEmbedding

from QAWithPDF.data_ingestion import load_data
from QAWithPDF.model_api import load_model

import sys
from exception import customexception
from logger import logging

def download_gemini_embedding(model,document):
    """
    Downloads and initializes a Gemini Embedding model for vector embeddings.

    Returns:
    - VectorStoreIndex: An index of vector embeddings for efficient similarity queries.
    """
    try:
        logging.info("")
        gemini_embed_model = GeminiEmbedding(model_name="models/embedding-001")
        llm = Gemini(models="gemini-pro", api_key="AIzaSyD6tWu64ToJIkNzn9PjPQkiu-VJyeq9CIo")

        # Initialize embedding model (e.g., OpenAIEmbedding)
        embed_model = GeminiEmbedding(model="models/embedding-001")

        # Configure the settings with the models
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.chunk_size = 800
        Settings.chunk_overlap = 20

        # You can also configure other settings such as number of output tokens, etc.
        Settings.num_output = 512  # Example of additional configuration
        Settings.context_window = 3900 
        logging.info("")
        index = VectorStoreIndex.from_documents(document)
        index.storage_context.persist()
        
        logging.info("")
        query_engine = index.as_query_engine()
        return query_engine
    except Exception as e:
        raise customexception(e,sys)