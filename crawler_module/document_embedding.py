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
        cursor.execute("SELECT id, url, content FROM t_raw_tab LIMIT ? OFFSET ?", (limit, offset))
        data = cursor.fetchall()
        conn.close()
        return data

    def compute_and_store_embeddings(self, data):
        """
        Compute embeddings for the provided data and store them in Chroma.
        """
        records_to_add = []
        records_to_update = []
        for row in data:
            documents_to_add = []  # Initialize a list to hold all document parts for batch processing.
            timestamp = int(time.time())
            doc_id, url, content = row
            text_vec = json.loads(content)
            part_index = 0
            for part_content in text_vec:
                # Process each part of the content.
                # Construct metadata for each document part, ensuring each part has a unique ID.
                metadata = {"source": url, "id": f"{doc_id}-part{part_index}"}
                # Create a Document object with the part content and metadata.
                doc = Document(page_content=part_content, metadata=metadata)
                # Add the document part to the list for batch addition.
                documents_to_add.append(doc)
                part_index += 1

            # Check if there are document parts to add.
            if documents_to_add:
                # Add all document parts to Chroma in a single batch operation.
                embedding_id_vec = self.chroma.add_documents(documents_to_add)
                print(f"doc_id={doc_id}, url={url} added {len(documents_to_add)} document parts to Chroma., embedding_id_vec={embedding_id_vec}")

                records_to_add.append((doc_id, json.dumps(embedding_id_vec), timestamp, timestamp))

                #`doc_status` meanings:
                #  1 - 'Web page recorded',
                #  2 - 'Web page crawling',
                #  3 - 'Web page crawled',
                #  4 - 'Web text Embedding stored in Chroma'
                records_to_update.append((4, timestamp, doc_id))

        conn = sqlite3.connect(self.sqlite_db_path)
        cursor = conn.cursor()

        # insert records into t_doc_embedding_map_tab
        if records_to_add:
            cursor.executemany('''
            INSERT INTO t_doc_embedding_map_tab (doc_id, embedding_id_list, ctime, mtime)
            VALUES (?, ?, ?, ?)
            ''', records_to_add)

        # update doc_status in t_raw_tab
        if records_to_update:
            cursor.executemany('''
            UPDATE t_raw_tab SET doc_status = ?, mtime = ? WHERE id = ?
            ''', records_to_update)

        conn.commit()
        conn.close()

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
    embedder.run(batch_size=1)  # Adjust the batch_size as per your requirement
    end = int(time.time())

    time_cost = end - begin
    print(f"\ntime_cost={time_cost}")

