# coding=utf-8
import json
import sqlite3
import time
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma
from config import SQLITE_DB_DIR, SQLITE_DB_NAME, CHROMA_DB_DIR, CHROMA_COLLECTION_NAME, OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL_NAME, BATCH_SIZE


class DocumentEmbedder:

    def __init__(self, sqlite_db_path=f"{SQLITE_DB_DIR}/{SQLITE_DB_NAME}", chroma_db_dir=CHROMA_DB_DIR, embedding_model_name=OPENAI_EMBEDDING_MODEL_NAME, collection_name=CHROMA_COLLECTION_NAME):
        self.sqlite_db_path = sqlite_db_path
        # Initialize OpenAI embeddings with the specified model
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model_name,
            openai_api_key=OPENAI_API_KEY
        )
        self.chroma = self._create_chroma_instance(chroma_db_dir, self.embeddings, collection_name)

    @staticmethod
    def _create_chroma_instance(db_file_dir, embeddings, collection_name):
        # Create Chroma instance with the embedding function from embeddings
        embedding_db_instance = Chroma(persist_directory=db_file_dir, embedding_function=embeddings, collection_name=collection_name)
        return embedding_db_instance

    def fetch_preprocessed_data(self, offset, limit):
        # Fetch data from SQLite in batches
        conn = sqlite3.connect(self.sqlite_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT raw_id, url, json_data FROM t_preprocess_tab LIMIT ? OFFSET ?", (limit, offset))
        data = cursor.fetchall()
        conn.close()
        return data

    def compute_and_store_embeddings(self, data):
        # Initialize a list to hold all documents for batch insertion
        documents_to_add = []
        for row in data:
            raw_id, url, json_data = row
            contents = json.loads(json_data)['summary']
            for content in contents:
                # Construct the metadata for each document
                metadata = {"source": url, "id": raw_id}
                # Create a Document object with page_content and metadata
                doc = Document(page_content=content['content'], metadata=metadata)
                # Add the document metadata to the list
                documents_to_add.append(doc)

        # Check if there are any documents to add
        if documents_to_add:
            # Add all documents to Chroma in a single batch
            self.chroma.add_documents(documents_to_add)
            print(f"Added {len(documents_to_add)} documents to Chroma.")
            print("\n==================================begin=======================")
            print(documents_to_add)
            print("==================================end=========================\n")

    def run(self, batch_size=BATCH_SIZE):
        # Main method to process and store embeddings
        offset = 0
        while True:
            data = self.fetch_preprocessed_data(offset, batch_size)
            if not data:
                break
            self.compute_and_store_embeddings(data)
            offset += batch_size
        print("Embedding computation and storage completed.")

if __name__ == "__main__":
    sqlite_db_path = f"{SQLITE_DB_DIR}/{SQLITE_DB_NAME}"
    chroma_db_dir = CHROMA_DB_DIR

    begin = int(time.time())
    embedder = DocumentEmbedder(sqlite_db_path, chroma_db_dir)
    embedder.run(batch_size=10)  # Adjust the batch_size as per your requirement
    end = int(time.time())

    time_cost = end - begin
    print(f"\ntime_cost={time_cost}")

