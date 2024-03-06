# coding=utf-8
import concurrent.futures
import json
import os
import re
import sqlite3
import time
from openai import OpenAI
from config import SQLITE_DB_DIR, SQLITE_DB_NAME, OPENAI_API_KEY, GPT_MODEL_NAME, BATCH_SIZE


class DocumentProcessor:

    def __init__(self, sqlite_db_path=f"{SQLITE_DB_DIR}/{SQLITE_DB_NAME}"):
        self.sqlite_db_path = sqlite_db_path
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = GPT_MODEL_NAME
        self.prompt_template = """
        Given the document provided in its original language, please provide a concise summary divided into a few distinct parts, formatted as a JSON list. Each part should adhere to the following constraints:
        - The content of each part should be rich in information and aim to be around 500 characters, but flexibility is allowed to ensure no critical information is lost. The goal is to minimize the number of parts while maximizing the information content of each.
        - Each part should capture a distinct aspect or section of the document, ensuring no information is lost from the original document.
        - Summaries should collectively cover the entire content of the document without overlapping information.

        Please format the summaries as a JSON list of objects, where each summary part is an object with a "content" key containing the text summary. Do not add any Markdown or special formatting characters like backticks or JSON notation. Avoid changing the original language of the document.
        Example of expected output format:
        [
            {"content": "Summary part 1."},
            {"content": "Summary part 2."},
            {"content": "Summary part 3."}
        ]

        Document:
        {formatted_documents}

        Based on the document above, please provide the summaries in the specified JSON list format, each containing a 'content' key with the text summary, and should retain the original language of the document.
        """

    def fetch_batch(self, offset, limit):
        conn = sqlite3.connect(self.sqlite_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, url, content FROM t_raw_tab ORDER BY id LIMIT ? OFFSET ?",
            (limit, offset)
        )
        batch_data = cursor.fetchall()
        conn.close()
        return batch_data

    def custom_format(self, template, **data):
        def replace(match):
            key = match.group(1)
            return data.get(key, match.group(0))

        pattern = r"\{(\w+)\}"
        return re.sub(pattern, replace, template)

    def process_document_with_gpt(self, document_content):
        prompt = self.custom_format(self.prompt_template, formatted_documents=document_content)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt}],
                temperature=0
            )
            ret = response.choices[0].message.content
            if ret:
                return json.loads(ret)
            else:
                print("Received an empty response or invalid JSON")
                return None
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def store_processed_data(self, raw_id, url, processed_data):
        conn = sqlite3.connect(self.sqlite_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO t_preprocess_tab (raw_id, url, json_data, data_length, process_time) VALUES (?, ?, ?, ?, ?)",
            (raw_id, url, json.dumps({"summary": processed_data}), len(processed_data), int(time.time()))
        )
        conn.commit()
        conn.close()

    def process_documents_concurrently(self, documents):
        """
        Process documents concurrently using a thread pool.
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Map futures to documents
            future_to_doc = {executor.submit(self.process_document_with_gpt, doc['content']): doc for doc in documents}

            for future in concurrent.futures.as_completed(future_to_doc):
                doc = future_to_doc[future]
                try:
                    print(f"raw_id={doc['raw_id']}, url={doc['url']}")
                    processed_data = future.result()
                    if processed_data:
                        self.store_processed_data(doc['raw_id'], doc['url'], processed_data)
                except Exception as exc:
                    print(f"Document {doc['raw_id']} generated an exception: {exc}")

    def run(self, batch_size=BATCH_SIZE):
        """
        Main processing workflow to batch process records from the database.
        """
        offset = 0
        while True:
            batch_data = self.fetch_batch(offset, batch_size)
            if not batch_data:
                break

            # Convert batch data to a list of dictionaries for passing more information
            documents = [{'raw_id': row[0], 'url': row[1], 'content': row[2]} for row in batch_data]

            # Process documents concurrently
            self.process_documents_concurrently(documents)

            offset += batch_size


if __name__ == "__main__":
    sqlite_db_path = f"{SQLITE_DB_DIR}/{SQLITE_DB_NAME}"  # Adjust the database path as needed

    begin = int(time.time())
    processor = DocumentProcessor(sqlite_db_path)
    processor.run(batch_size=10)  # Adjust the batch_size as per your requirement
    end = int(time.time())

    time_cost = end - begin
    print(f"\ntime_cost={time_cost}")
