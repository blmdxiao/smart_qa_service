# coding=utf-8
import asyncio
from functools import wraps
import json
import os
import sqlite3
from threading import Thread
import time
from urllib.parse import urljoin, urlparse
import uuid
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from langchain.embeddings.openai import OpenAIEmbeddings
from openai import OpenAI
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from utils.redis_config import redis_client
from utils.redis_lock import RedisLock
from utils.token_helper import TokenHelper
from utils.logger_config import my_logger as logger
from crawler_module.web_link_crawler import AsyncCrawlerSiteLink
from crawler_module.web_content_crawler import AsyncCrawlerSiteContent
from crawler_module.document_embedding import DocumentEmbedder


# Load environment variables from .env file
load_dotenv()

# Retrieve environment variables or set default values
SITE_TITLE = os.getenv('SITE_TITLE', 'your_site_title')
SQLITE_DB_DIR = os.getenv('SQLITE_DB_DIR', 'your_sqlite_db_directory')
SQLITE_DB_NAME = os.getenv('SQLITE_DB_NAME', 'your_sqlite_db_name')
MAX_CRAWL_PARALLEL_REQUEST = int(os.getenv('MAX_CRAWL_PARALLEL_REQUEST', '5'))
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your_openai_api_key')
CHROMA_DB_DIR = os.getenv('CHROMA_DB_DIR', 'your_chroma_db_directory')
CHROMA_COLLECTION_NAME = os.getenv('CHROMA_COLLECTION_NAME', 'your_collection_name')
GPT_MODEL_NAME = os.getenv('GPT_MODEL_NAME', 'gpt-3.5-turbo')
OPENAI_EMBEDDING_MODEL_NAME = os.getenv('OPENAI_EMBEDDING_MODEL_NAME', 'text-embedding-3-small')
MAX_EMBEDDING_INPUT = int(os.getenv('MAX_EMBEDDING_INPUT', '2048'))
MAX_QUERY_SIZE = int(os.getenv('MAX_QUERY_SIZE', '200'))
RECALL_TOP_K = int(os.getenv('RECALL_TOP_K', '3'))
MAX_HISTORY_QUERY_SIZE = int(os.getenv('MAX_HISTORY_QUERY_SIZE', '5'))
HISTORY_EXPIRE_TIME = int(os.getenv('HISTORY_EXPIRE_TIME', '10800'))
STATIC_DIR = os.getenv('STATIC_DIR', 'your_static_dir')
URL_PREFIX = os.getenv('URL_PREFIX', 'your_url_prefix')
MEDIA_DIR = os.getenv('MEDIA_DIR', 'your_media_dir')


app = Flask(__name__)


# Initialize Redis distributed lock
g_redis_lock = RedisLock(redis_client, 'open_kf:distributed_lock')

# Set OpenAI GPT API key
g_client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize OpenAI embeddings with the specified model
g_embeddings = OpenAIEmbeddings(
    model=OPENAI_EMBEDDING_MODEL_NAME,
    openai_api_key=OPENAI_API_KEY
)

g_document_embedder = DocumentEmbedder(
    persist_directory=CHROMA_DB_DIR,
    embedding_function=g_embeddings,
    collection_name=CHROMA_COLLECTION_NAME
)


def token_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(" ")[1]
        if not token:
            logger.error("Token is missing!")
            return jsonify({'retcode': -10000, 'message': 'Token is missing!', 'data': {}})
        try:
            user_payload = TokenHelper.verify_token(token)
            if user_payload == 'Token expired':
                logger.error(f"Token: '{token}' is expired!")
                return jsonify({'retcode': -10001, 'message': 'Token is expired!', 'data': {}})
            elif user_payload == 'Invalid token':
                logger.error(f"Token: '{token}' is invalid")
                return jsonify({'retcode': -10001, 'message': 'Token is invalid!', 'data': {}})
            request.user_payload = user_payload  # Store payload in request for further use
        except Exception as e:
            logger.error(f"Token: '{token}' is invalid, the exception is {e}")
            return jsonify({'retcode': -10001, 'message': 'Token is invalid!', 'data': {}})
        return f(*args, **kwargs)
    return decorated_function

def get_db_connection():
    conn = sqlite3.connect(f"{SQLITE_DB_DIR}/{SQLITE_DB_NAME}")
    conn.row_factory = sqlite3.Row  # Set row factory to access columns by name
    return conn

def get_user_query_history(user_id):
    history_key = f"open_kf:query_history:{user_id}"
    history_items = redis_client.lrange(history_key, 0, -1)
    history = [json.loads(item) for item in history_items]
    return history

def search_and_answer(query, user_id, k=RECALL_TOP_K):
    # Perform similarity search
    t1 = time.time()
    #results = g_chroma.similarity_search_with_relevance_scores(query, k=k)
    #results = g_chroma.similarity_search_with_score(query, k=k)
    results = g_document_embedder.search_document(query, k)
    time_cost1 = time.time() - t1
    logger.info(f"for the query:'{query}' and user_id:'{user_id}, the top {k} results are: {results} \n*******\nthe time_cost of similarity_search_with_score is {time_cost1}\n*******\n")

    # Build the prompt for GPT
    context = "\n\n".join([f"Document URL: {result[0].metadata['source']}\nContent: {result[0].page_content}" for result in results])

    # Get user history from Redis
    user_history = get_user_query_history(user_id)
    # Include user history in the prompt
    history_context = "\n--------------------\n".join([f"Previous Query: {item['query']}\nPrevious Answer: {item['answer']}" for item in user_history])
    logger.info(f"for the query:'{query}' and user_id:'{user_id}', the history_context is {history_context}")

    site_title = SITE_TITLE
    prompt = f"""
    This is a smart customer service bot designed to assist users by providing information based on the content of the '{site_title}' website and its documentation. The system uses a combination of Language Model Generative Pre-trained Transformer (GPT) and Retriever-Augmented Generation (RAG) with Chroma as the vector database to find the most relevant documents in response to user queries.

    Given the user's previous interactions as described above, consider how their past queries might inform their current needs. This historical context can help tailor the response to be more aligned with their likely interests or unresolved questions from previous interactions.

    When a query is received, it first performs a similarity search to recall the top {k} documents from Chroma. These documents then serve as the context for generating an answer. The aim is to provide users with precise information related to the '{site_title}' website, enhancing their understanding and usage of the site.

    For general greetings or queries not directly related to the website's content (e.g., "hello", "who are you"), the system should provide a friendly response and guide the user towards making inquiries related to the services or information available on the '{site_title}' website.

    The goal is to assist users in retrieving information specific to the '{site_title}' website's offerings and documentation. Therefore, when generating a response, consider the user's actual application scenario and the intent behind their query, as well as any relevant history from their previous queries. Ensure that the response is informative, directly related to the query, and based on the documents provided as context.

    **It is crucial to provide responses that are as detailed and comprehensive as possible.** When the query indicates a need for specific information, such as URLs, steps, or example code, **the response should aim to include all such details**. Use the context from the top recalled documents and any relevant history to form a thorough answer, leveraging any available specifics to enhance the relevance and usefulness of the response.

    Additionally, **if the query's true intent seems to extend beyond the literal question asked, strive to address the underlying need or interest,** possibly hinted at in their query history. This means not only answering the direct question but also providing additional information that could be helpful to the user based on the context and intent of the query, as well as their past interactions.

    When generating code snippets or examples, **it is essential to ensure the code is not only correct but also well-formatted, with proper indentation and spacing for enhanced readability and adherence to coding standards.** This helps users easily understand and apply the code within their own projects.

    Given the information from the documents listed below and user's query history, please formulate a detailed and specific answer to the query in the same language as the query. Your response should be in JSON format, containing 'answer' and 'source' fields. The 'answer' field must include a precise and informative response based on the document contents, matching the language of the query, and considering any relevant user history. The 'source' field should list the URLs of the documents that directly support your answer. If the documents do not provide sufficient information for a definitive answer, please indicate that the answer is unknown in the 'answer' field.

    **Documents:**
    {context}

    **User History:**
    {history_context}

    **Query:**
    '{query}'

    Instructions for response:
    - Ensure your answer is relevant to the '{site_title}' website's content and the user's query history.
    - Provide a detailed and specific answer based on the information found in the documents and the user's past queries, including URLs, steps, example code, and any other specifics requested in the query.
    - For general inquiries or unrelated questions, offer a standard response that encourages users to ask more specific questions related to the website and their previous interactions.
    - Respond in a manner that considers the user's intent and the practical application of the query, addressing not just the literal question but also the broader context, potential needs, and past queries.
    - When providing code examples, ensure the code is correct and follows best practices for formatting and indentation to promote readability and maintainability.
    - Avoid speculative or general responses not supported by the document contents or the user's query history.
    - **Respond in a manner that aligns with the query's language (e.g., if the query is in Chinese, respond in Chinese; if in English, respond in English; and so on for other languages)**.

    Please format your response as follows:
    {{
      "answer": "Provide a detailed and specific answer here, in the same language as the query, including any requested URLs, steps, example code, or other specifics.",
      "source": ["URL(s) of the document(s) supporting your answer"]
    }}

    Please format `answer` as follows:
    The `answer` must be fully formatted using Markdown syntax to ensure proper rendering on web interfaces. This includes:
    - **Bold** (`**bold**`) and *italic* (`*italic*`) text for emphasis.
    - Unordered lists (`- item`) for itemization and ordered lists (`1. item`) for sequencing.
    - `Inline code` (`` `Inline code` ``) for brief code snippets and (` ``` `) for longer examples, specifying the programming language for syntax highlighting when possible.
    - [Hyperlinks](URL) (`[Hyperlinks](URL)`) to reference external sources.
    - Headings (`# Heading 1`, `## Heading 2`, ...) to structure the answer effectively.

    Ensure each Markdown element is used appropriately for its intended purpose. Avoid common formatting errors such as inconsistent use of list symbols, improper nesting of Markdown elements, or broken link syntax.
    """

    # Call GPT model to generate an answer
    response = g_client.chat.completions.create(
        model=GPT_MODEL_NAME,
        messages=[{"role": "system", "content": prompt}],
        temperature=0
    )
    ret = response.choices[0].message.content
    return ret


@app.route('/open_kf_api/get_token', methods=['POST'])
def get_token():
    data = request.json
    user_id = data.get('user_id')
    if not user_id:
        return jsonify({'retcode': -10000, 'message': 'user_id is required', 'data': {}})

    try:
        # generate token
        token = TokenHelper.generate_token(user_id)
        logger.success(f"generate token:'{token}' with user_id:'{user_id}'")
        return jsonify({"retcode": 0, "message": "success", "data": {"token": token}})
    except Exception as e:
        logger.error(f"generate token with user_id:'{user_id}' is failed, the exception is {e}")
        return jsonify({'retcode': -20000, 'message': str(e), 'data': {}})


@app.route('/open_kf_api/smart_query', methods=['POST'])
@token_required
def smart_query():
    data = request.json
    user_id = data.get('user_id')
    query = data.get('query')
    if not user_id or not query:
        logger.error(f"user_id and query are required")
        return jsonify({'retcode': -20000, 'message': 'user_id and query are required', 'data': {}})
    
    token_user_id = request.user_payload['user_id']
    if token_user_id != user_id:
        logger.error(f"user_id:'{user_id}' does not match with token_user_id:'{token_user_id}'")
        return jsonify({'retcode': -10001, 'message': 'Token is invalid!', 'data': {}})

    try:
        # Check if the query is in Redis
        redis_key = f"open_kf:intervene:{query}"
        intervene_data = redis_client.get(redis_key)
        if intervene_data:
            # If found in Redis, parse the JSON data and return it
            intervene_data = json.loads(intervene_data)
            logger.info(f"user_id:'{user_id}', query:'{query}' is hit in Redis, the intervene_data is {intervene_data}")
            return jsonify({"retcode": 0, "message": "success", "data": intervene_data})
    except Exception as e:
        logger.error(f"Redis exception {e} for user_id:'{user_id}' and query:'{query}'")
        # Just ignore Redis error
        #return jsonify({'retcode': -40000, 'message': f'Redis exception {e}', 'data': {}})

    if len(query) > MAX_QUERY_SIZE:
        query = query[:MAX_QUERY_SIZE]

    last_character = query[-1]
    if last_character != "？" and last_character != "?":
        query += "?"

    try:
        beg = time.time()
        answer = search_and_answer(query, user_id)
        time_cost = time.time() - beg
        answer_json = json.loads(answer)
        logger.success(f"query:'{query}' and user_id:'{user_id}' is processed successfully, the answer is {answer} \n-------\nthe total time_cost is {time_cost}\n-------\n")
    except Exception as e:
        logger.error(f"query:'{query}' and user_id:'{user_id}' is processed failed, the exception is {e}")
        return jsonify({'retcode': -20000, 'message': str(e), 'data': {}})

    try:
        # After generating the response from GPT
        # Store user query and GPT response in Redis
        history_key = f"open_kf:query_history:{user_id}"
        history_data = {'query': query, 'answer': answer_json}
        redis_client.lpush(history_key, json.dumps(history_data))
        redis_client.ltrim(history_key, 0, MAX_HISTORY_QUERY_SIZE - 1)  # Keep only the latest N entries
        # Set the expiry time for the history key
        redis_client.expire(history_key, HISTORY_EXPIRE_TIME)
    except Exception as e:
        logger.error(f"query:'{query}' and user_id:'{user_id}' is processed failed with Redis, the exception is {e}")
        # just ignore Redis error
        #return jsonify({'retcode': -30000, 'message': str(e), 'data': {}})

    timestamp = int(time.time())
    conn = None
    try:
        # Store user query and GPT resposne in DB
        conn = get_db_connection()
        if g_redis_lock.acquire_lock():
            try:
                conn.execute('INSERT INTO t_user_qa_record_tab (user_id, query, answer, source, ctime, mtime) VALUES (?, ?, ?, ?, ?, ?)',
                             (user_id, query, answer_json["answer"], json.dumps(answer_json["source"]), timestamp, timestamp))
                conn.commit()
            finally:
                g_redis_lock.release_lock()
    except Exception as e:
        logger.error(f"query:'{query}' and user_id:'{user_id}' is processed failed with Database, the exception is {e}")
        # just ignore DB error
        #return jsonify({'retcode': -30000, 'message': str(e), 'data': {}})
    finally:
        if conn:
            conn.close()

    return jsonify({"retcode": 0, "message": "success", "data": answer_json})


@app.route('/open_kf_api/get_user_conversation_list', methods=['POST'])
@token_required
def get_user_conversation_list():
    """Retrieve a list of user conversations within a specified time range, with pagination and total count."""
    data = request.json
    start_timestamp = data.get('start_timestamp')
    end_timestamp = data.get('end_timestamp')
    page = data.get('page')
    page_size = data.get('page_size')

    if None in ([start_timestamp, end_timestamp, page, page_size]):
        return jsonify({'retcode': -10002, 'message': 'Missing required parameters'})

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        offset = (page - 1) * page_size

        # First, get the total count of distinct user_ids within the time range for pagination
        cur.execute("""
            SELECT COUNT(DISTINCT user_id) AS total_count FROM t_user_qa_record_tab
            WHERE ctime BETWEEN ? AND ?
        """, (start_timestamp, end_timestamp))
        total_count = cur.fetchone()['total_count']

        # Then, fetch the most recent conversation record for each distinct user within the time range
        cur.execute(f"""
            WITH RankedConversations AS (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY ctime DESC) AS rn
                FROM t_user_qa_record_tab
                WHERE ctime BETWEEN ? AND ?
            )
            SELECT * FROM RankedConversations WHERE rn = 1
            ORDER BY ctime DESC
            LIMIT ? OFFSET ?
        """, (start_timestamp, end_timestamp, page_size, offset))

        conversation_list = [{
            "user_id": row["user_id"],
            "latest_query": {
                "id": row["id"],
                "query": row["query"],
                "answer": row["answer"],
                "source": json.loads(row["source"]),
                "ctime": row["ctime"],
                "mtime": row["mtime"]
            }
        } for row in cur.fetchall()]

        return jsonify({'retcode': 0, 'message': 'Success', 'data': {'total_count': total_count, 'conversation_list': conversation_list}})
    except Exception as e:
        logger.error(f"Failed to retrieve user conversation list: {e}")
        return jsonify({'retcode': -10003, 'message': 'Internal server error'})
    finally:
        if conn:
            conn.close()


@app.route('/open_kf_api/get_user_query_history_list', methods=['POST'])
@token_required
def get_user_query_history_list():
    data = request.json
    page = data.get('page')
    page_size = data.get('page_size')
    user_id = data.get('user_id')

    # Check for mandatory parameters
    if None in (page, page_size, user_id):
        logger.error("page, page_size and user_id are required")
        return jsonify({'retcode': -20000, 'message': 'page, page_size and user_id are required', 'data': {}})

    try:
        # Convert timestamps and pagination parameters to integers
        page = int(page)
        page_size = int(page_size)
    except ValueError as e:
        logger.error(f"Parameter conversion error: {e}")
        return jsonify({'retcode': -20001, 'message': 'Invalid parameters', 'data': {}})

    # Build query conditions
    query_conditions = "WHERE user_id = ?"
    params = [user_id]

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")

        # First, query the total count of records under the given conditions
        cur.execute(f'SELECT COUNT(*) FROM t_user_qa_record_tab {query_conditions}', params)
        total_count = cur.fetchone()[0]

        # Then, query the paginated records
        cur.execute(f'SELECT * FROM t_user_qa_record_tab {query_conditions} ORDER BY ctime DESC LIMIT ? OFFSET ?',
                    params + [page_size, (page-1) * page_size])
        rows = cur.fetchall()

        record_list = [dict(row) for row in rows]  # Convert rows to dictionaries
        # Apply json.loads on the 'source' field of each record
        for record in record_list:
            if 'source' in record:  # Ensure the 'source' key exists
                try:
                    record['source'] = json.loads(record['source'])  # Convert JSON string to Python list
                except json.JSONDecodeError:
                    record['source'] = []  # If decoding fails, set to an empty list or other default value

        return jsonify({
            "retcode": 0,
            "message": "success",
            "data": {
                "total_count": total_count,
                "query_list": record_list
            }
        })
    except Exception as e:
        logger.error(f"Database exception: {e}")
        return jsonify({'retcode': -30000, 'message': 'Database exception', 'data': {}})
    finally:
        if conn:
            conn.close()


@app.route('/open_kf_api/add_intervene_record', methods=['POST'])
@token_required
def add_intervene_record():
    data = request.json
    query = data.get('query')
    intervene_answer = data.get('intervene_answer')
    source = data.get('source', [])

    if None in (query, intervene_answer, source):
        return jsonify({'retcode': -20000, 'message': 'Missing mandatory parameters', 'data': {}})

    conn = None
    try:
        # Check if query already exists in the database
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")

        cur.execute('SELECT COUNT(*) FROM t_user_qa_intervene_tab WHERE query = ?', (query,))
        result = cur.fetchone()
        if result and result[0] > 0:
            logger.error(f"intervene query:'{query}' is already exists in the database")
            return jsonify({'retcode': -30001, 'message': 'Query already exists in the database', 'data': {}})

        # Insert the intervene record into DB
        timestamp = int(time.time())
        source_str = json.dumps(source)

        cur = conn.cursor()
        if g_redis_lock.acquire_lock():
            try:
                cur.execute('INSERT INTO t_user_qa_intervene_tab (query, intervene_answer, source, ctime, mtime) VALUES (?, ?, ?, ?, ?)',
                            (query, intervene_answer, source_str, timestamp, timestamp))
                conn.commit()
            finally:
                g_redis_lock.release_lock()

        # Update Redis using simple string with the query as the key (prefixed)
        redis_key = f"open_kf:intervene:{query}"
        redis_value = json.dumps({"answer": intervene_answer, "source": source})
        redis_client.set(redis_key, redis_value)

        return jsonify({"retcode": 0, "message": "success", 'data': {}})
    except Exception as e:
        return jsonify({'retcode': -30000, 'message': 'Database or Redis error', 'data': {}})
    finally:
        if conn:
            conn.close()


@app.route('/open_kf_api/delete_intervene_record', methods=['POST'])
@token_required
def delete_intervene_record():
    data = request.json
    record_id = data.get('id')

    if not record_id:
        return jsonify({'retcode': -20000, 'message': 'id is required', 'data': {}})

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")

        # First, find the query string for the given id to delete it from Redis
        cur.execute('SELECT query FROM t_user_qa_intervene_tab WHERE id = ?', (record_id,))
        row = cur.fetchone()

        if row:
            query = row['query']
            # Delete the record from DB
            if g_redis_lock.acquire_lock():
                try:
                    cur.execute('DELETE FROM t_user_qa_intervene_tab WHERE id = ?', (record_id,))
                    conn.commit()
                finally:
                    g_redis_lock.release_lock()

            # Now, delete the corresponding record from Redis
            redis_key = f"open_kf:intervene:{query}"
            redis_client.delete(redis_key)

            return jsonify({"retcode": 0, "message": "success", 'data': {}})
        else:
            return jsonify({'retcode': -20001, 'message': 'Record not found', 'data': {}})
    except Exception as e:
        return jsonify({'retcode': -30000, 'message': 'Database error', 'data': {}})
    finally:
        if conn:
            conn.close()


@app.route('/open_kf_api/batch_delete_intervene_record', methods=['POST'])
@token_required
def batch_delete_intervene_record():
    data = request.json
    id_list = data.get('id_list')

    if not id_list or not isinstance(id_list, list) or len(id_list) == 0:
        return jsonify({'retcode': -20000, 'message': 'Missing or invalid mandatory parameter: id_list', 'data': {}})

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")

        # Retrieve the queries to delete their corresponding Redis entries
        cur.execute(f'SELECT query FROM t_user_qa_intervene_tab WHERE id IN ({",".join(["?"]*len(id_list))})', id_list)
        rows = cur.fetchall()

        for row in rows:
            query = row['query']
            redis_key = f"open_kf:intervene:{query}"
            redis_client.delete(redis_key)  # Delete from Redis

        # Then, batch delete from DB
        if g_redis_lock.acquire_lock():
            try:
                cur.execute(f'DELETE FROM t_user_qa_intervene_tab WHERE id IN ({",".join(["?"]*len(id_list))})', id_list)
                conn.commit()
            finally:
                g_redis_lock.release_lock()

        return jsonify({"retcode": 0, "message": "success", 'data': {}})
    except Exception as e:
        return jsonify({'retcode': -30000, 'message': 'Database error', 'data': {}})
    finally:
        if conn:
            conn.close()


@app.route('/open_kf_api/update_intervene_record', methods=['POST'])
@token_required
def update_intervene_record():
    data = request.json
    record_id = data.get('id')
    intervene_answer = data.get('intervene_answer')
    source = data.get('source', [])

    if None in (record_id, intervene_answer, source):
        return jsonify({'retcode': -20000, 'message': 'Missing or invalid mandatory parameters', 'data': {}})

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")

        source_json = json.dumps(source)  # Convert the source list to a JSON string for storing in DB
        timestamp = int(time.time())
        # Update the DB record
        if g_redis_lock.acquire_lock():
            try:
                cur.execute('UPDATE t_user_qa_intervene_tab SET intervene_answer = ?, source = ?, mtime = ? WHERE id = ?',
                            (intervene_answer, source_json, timestamp, record_id))
                conn.commit()
            finally:
                g_redis_lock.release_lock()

        # Retrieve the query text to update the corresponding Redis entry
        cur.execute('SELECT query FROM t_user_qa_intervene_tab WHERE id = ?', (record_id,))
        row = cur.fetchone()
        if row:
            query = row['query']
            redis_key = f"open_kf:intervene:{query}"
            redis_value = json.dumps({"answer": intervene_answer, "source": source})
            redis_client.set(redis_key, redis_value)  # Update Redis
        else:
            return jsonify({'retcode': -20001, 'message': 'Record not found', 'data': {}})

        return jsonify({"retcode": 0, "message": "success", 'data': {}})
    except Exception as e:
        return jsonify({'retcode': -30000, 'message': 'Database error', 'data': {}})
    finally:
        if conn:
            conn.close()


@app.route('/open_kf_api/get_intervene_query_list', methods=['POST'])
@token_required
def get_intervene_query_list():
    data = request.json
    start_timestamp = data.get('start_timestamp')
    end_timestamp = data.get('end_timestamp')
    page = data.get('page')
    page_size = data.get('page_size')

    # Validate mandatory parameters
    if None in (start_timestamp, end_timestamp, page, page_size):
        return jsonify({'retcode': -20000, 'message': 'Missing mandatory parameters', 'data': {}})

    try:
        # Convert timestamps and pagination parameters to integers
        start_timestamp = int(start_timestamp)
        end_timestamp = int(end_timestamp)
        page = int(page)
        page_size = int(page_size)
    except ValueError:
        return jsonify({'retcode': -20001, 'message': 'Invalid parameters', 'data': {}})

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")

        # Calculate total count
        cur.execute('SELECT COUNT(*) FROM t_user_qa_intervene_tab WHERE ctime BETWEEN ? AND ?', (start_timestamp, end_timestamp))
        total_count = cur.fetchone()[0]

        # Calculate the starting point for the query
        start = (page - 1) * page_size

        # Retrieve the specified page of records
        cur.execute('''
            SELECT id, query, intervene_answer, source, ctime, mtime
            FROM t_user_qa_intervene_tab
            WHERE ctime BETWEEN ? AND ?
            ORDER BY ctime DESC
            LIMIT ? OFFSET ?''', (start_timestamp, end_timestamp, page_size, start))

        rows = cur.fetchall()
        record_list = [dict(row) for row in rows]  # Convert rows to dictionaries
        # Apply json.loads on the 'source' field of each record
        for record in record_list:
            if 'source' in record:  # Ensure the 'source' key exists
                try:
                    record['source'] = json.loads(record['source'])  # Convert JSON string to Python list
                except json.JSONDecodeError:
                    record['source'] = []  # If decoding fails, set to an empty list or other default value

        return jsonify({
            "retcode": 0,
            "message": "success",
            "data": {
                "total_count": total_count,
                "intervene_list": record_list
            }
        })
    except Exception as e:
        return jsonify({'retcode': -30000, 'message': 'Database error', 'data': {}})
    finally:
        if conn:
            conn.close()


@app.route('/open_kf_api/login', methods=['POST'])
def login():
    data = request.json
    account_name = data.get('account_name')
    password = data.get('password')

    if not account_name or not password:
        return jsonify({'retcode': -10002, 'message': 'Account name and password are required', 'data': {}})

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")

        # Check if the account exists and verify the password
        cur.execute('SELECT id, password_hash FROM t_account_tab WHERE account_name = ?', (account_name,))
        account = cur.fetchone()

        if account and check_password_hash(account['password_hash'], password):
            # Generate token with account_name in the payload
            token = TokenHelper.generate_token(account_name)
            logger.info(f"Generate token:'{token}'")
            
            # Set is_login to 1 and update mtime to the current Unix timestamp
            if g_redis_lock.acquire_lock():
                try:
                    cur.execute('UPDATE t_account_tab SET is_login = 1, mtime = ? WHERE account_name = ?', (int(time.time()), account_name,))
                    conn.commit()
                finally:
                    g_redis_lock.release_lock()

            return jsonify({'retcode': 0, 'message': 'Login successful', 'data': {'token': token}})
        else:
            return jsonify({'retcode': -10003, 'message': 'Invalid credentials', 'data': {}})
    except Exception as e:
        return jsonify({'retcode': -30001, 'message': f'An error occurred during login, exception:{e}', 'data': {}})
    finally:
        if conn:
            conn.close()


@app.route('/open_kf_api/update_password', methods=['POST'])
@token_required
def update_password():
    data = request.json
    account_name = data.get('account_name')
    current_password = data.get('current_password')
    new_password = data.get('new_password')

    if None in (account_name, current_password, new_password):
        return jsonify({'retcode': -10004, 'message': 'Account name, current password, and new password are required', 'data': {}})

    token_user_id = request.user_payload['user_id']
    if token_user_id != account_name:
        logger.error(f"account_name:'{account_name}' does not match with token_user_id:'{token_user_id}'")
        return jsonify({'retcode': -10001, 'message': 'Token is invalid!', 'data': {}})

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")

        # Check if the account exists and verify the current password
        cur.execute('SELECT id, password_hash FROM t_account_tab WHERE account_name = ?', (account_name,))
        account = cur.fetchone()

        if not account or not check_password_hash(account['password_hash'], current_password):
            logger.error(f"Invalid account_name:'{account_name}' or current_password:'{current_password}'")
            return jsonify({'retcode': -10005, 'message': 'Invalid account name or password', 'data': {}})

        # Update the password
        new_password_hash = generate_password_hash(new_password, method='pbkdf2:sha256', salt_length=10)
        if g_redis_lock.acquire_lock():
            try:
                cur.execute('UPDATE t_account_tab SET password_hash = ?, mtime = ? WHERE account_name = ?', (new_password_hash, int(time.time()), account_name,))
                conn.commit()
            finally:
                g_redis_lock.release_lock()

        return jsonify({'retcode': 0, 'message': 'Password updated successfully', 'data': {}})
    except Exception as e:
        return jsonify({'retcode': -10006, 'message': f'An error occurred: {str(e)}', 'data': {}})
    finally:
        if conn:
            conn.close()


@app.route('/open_kf_api/get_bot_setting', methods=['POST'])
def get_bot_setting():
    """Retrieve bot setting, first trying Redis and falling back to DB if not found."""
    try:
        # Attempt to retrieve the setting from Redis
        redis_key = "open_kf:bot_setting"
        setting_redis = redis_client.get(redis_key)
        if setting_redis:
            setting_data = json.loads(setting_redis)
            return jsonify({'retcode': 0, 'message': 'Success', 'data': {'config': setting_data}})
        else:
            logger.warning(f"could not find '{redis_key}' in Redis!")
    except Exception as e:
        logger.error(f"Error retrieving setting from Redis, excpetion:{e}")
        # Just ignore Redis error
        #return jsonify({'retcode': -10006, 'message': f'An error occurred: {str(e)}', 'data': {}})

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")

        cur.execute('SELECT * FROM t_bot_setting_tab LIMIT 1')
        setting = cur.fetchone()
        if setting:
            setting = dict(setting)
            # Process and return the setting details
            setting_data = {k: json.loads(v) if k in ['initial_messages', 'suggested_messages'] else v for k, v in setting.items()}
            return jsonify({'retcode': 0, 'message': 'Success', 'data': {'config': setting_data}})
        else:
            logger.warning(f"No setting found")
            return jsonify({'retcode': -10008, 'message': 'No setting found', 'data': {}})
    except Exception as e:
        logger.error(f"Error retrieving setting: {e}")
        return jsonify({'retcode': -10006, 'message': f'An error occurred: {str(e)}', 'data': {}})
    finally:
        if conn:
            conn.close()


@app.route('/open_kf_api/update_bot_setting', methods=['POST'])
@token_required
def update_bot_setting():
    # Retrieve request data
    data = request.json

    # Extract and validate all required fields
    setting_id = data.get('id')
    initial_messages = data.get('initial_messages')
    suggested_messages = data.get('suggested_messages')
    bot_name = data.get('bot_name')
    bot_avatar = data.get('bot_avatar')
    chat_icon = data.get('chat_icon')
    placeholder = data.get('placeholder')
    model = data.get('model')

    # Check for the presence of all required fields
    if None in (setting_id, initial_messages, suggested_messages, bot_name, bot_avatar, chat_icon, placeholder, model):
        return jsonify({'retcode': -10004, 'message': 'All fields are required', 'data': {}})

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")

        # Check if the setting with provided ID exists
        cur.execute('SELECT id FROM t_bot_setting_tab WHERE id = ?', (setting_id,))
        if not cur.fetchone():
            logger.error(f"No setting found")
            return jsonify({'retcode': -10009, 'message': 'Setting not found', 'data': {}})

        # Convert lists to JSON strings for storage
        initial_messages_json = json.dumps(initial_messages)
        suggested_messages_json = json.dumps(suggested_messages)

        # Update the setting
        if g_redis_lock.acquire_lock():
            try:
                cur.execute('''
                    UPDATE t_bot_setting_tab
                    SET initial_messages = ?, suggested_messages = ?, bot_name = ?, bot_avatar = ?, chat_icon = ?, placeholder = ?, model = ?, mtime = ?
                    WHERE id = ?
                ''', (initial_messages_json, suggested_messages_json, bot_name, bot_avatar, chat_icon, placeholder, model, int(time.time()), setting_id))
                conn.commit()
            finally:
                g_redis_lock.release_lock()
        return jsonify({'retcode': 0, 'message': 'Settings updated successfully', 'data': {}})
    except Exception as e:
        logger.error(f"Error updating setting: {str(e)}")
        return jsonify({'retcode': -10006, 'message': f'An error occurred: {str(e)}', 'data': {}})
    finally:
        if conn:
            conn.close()


def is_valid_url(url):
    """Check if the provided string is a valid URL."""
    parsed_url = urlparse(url)
    return bool(parsed_url.scheme) and bool(parsed_url.netloc)

def async_crawl_link_task(site, version):
    """Start the crawl link task in an asyncio event loop."""
    logger.info(f"create crawler_link")
    crawler_link = AsyncCrawlerSiteLink(
        base_url=site,
        sqlite_db_path=f"{SQLITE_DB_DIR}/{SQLITE_DB_NAME}",
        max_requests=MAX_CRAWL_PARALLEL_REQUEST,
        version=version,
        redis_lock=g_redis_lock
    )
    logger.info(f"async_crawl_link_task begin!, site:'{site}', version:{version}")
    asyncio.run(crawler_link.run())
    logger.info(f"async_crawl_link_task end!, site:'{site}', version:{version}")

@app.route('/open_kf_api/submit_crawl_site', methods=['POST'])
@token_required
def submit_crawl_site():
    """Submit a site for crawling."""
    data = request.json
    site = data.get('site')
    timestamp = data.get('timestamp')

    if not site or not timestamp:
        return jsonify({'retcode': -10004, 'message': 'site and timestamp are required', 'data': {}})

    if not is_valid_url(site):
        logger.error(f"site:'{site} is not a valid URL!")
        return jsonify({'retcode': -10007, 'message': f"site:'{site}' is not a valid URL", 'data': {}})

    domain = urlparse(site).netloc
    logger.info(f"domain is '{domain}'")
    conn = None
    try:
        timestamp = int(timestamp)
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")

        # Check if the domain exists in the database
        cur.execute("SELECT id, version FROM t_domain_tab WHERE domain = ?", (domain,))
        domain_info = cur.fetchone()

        if domain_info and timestamp <= domain_info["version"]:
            return jsonify({'retcode': -10005, 'message': f'New timestamp:{timestamp} must be greater than the current version:{domain_info["version"]}.', 'data': {}})

        if g_redis_lock.acquire_lock():
            try:
                if domain_info:
                    domain_id, version = domain_info
                    # Update domain record
                    cur.execute("UPDATE t_domain_tab SET version = ?, domain_status = 1, mtime=? WHERE id = ?", (timestamp, int(time.time()), domain_id))
                else:
                    # Insert new domain record
                    cur.execute("INSERT INTO t_domain_tab (domain, domain_status, version, ctime, mtime) VALUES (?, 1, ?, ?, ?)", (domain, timestamp, int(time.time()), int(time.time())))

                conn.commit()
            finally:
                g_redis_lock.release_lock()

        # Start the asynchronous crawl task
        Thread(target=async_crawl_link_task, args=(site, timestamp)).start()

        return jsonify({'retcode': 0, 'message': 'Site submitted successfully for crawling.', 'data': {}})
    except Exception as e:
        return jsonify({'retcode': -10006, 'message': f'An error occurred: {str(e)}', 'data': {}})
    finally:
        if conn:
            conn.close()


@app.route('/open_kf_api/get_crawl_site_info', methods=['POST'])
@token_required
def get_crawl_site_info():
    """Fetch the crawl status information for a given site or all sites if site parameter is not provided."""
    data = request.json
    site = data.get('site', None)  # Make site an optional parameter

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")

        if site:
            if not is_valid_url(site):
                logger.error(f"site:'{site}' is not a valid URL!")
                return jsonify({'retcode': -10007, 'message': f"site:'{site}' is not a valid URL", 'data': {}})
            domain = urlparse(site).netloc
            logger.info(f"Searching for domain: '{domain}'")
            cur.execute("SELECT * FROM t_domain_tab WHERE domain = ?", (domain,))
        else:
            logger.info("Fetching information for all sites.")
            cur.execute("SELECT * FROM t_domain_tab")

        rows = cur.fetchall()
        if rows:
            sites_info = [dict(row) for row in rows]
            return jsonify({'retcode': 0, 'message': 'Success', 'data': {'sites_info': sites_info}})
        else:
            return jsonify({'retcode': -10008, 'message': 'No site information found', 'data': {}})
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return jsonify({'retcode': -10006, 'message': f'An error occurred: {str(e)}', 'data': {}})
    finally:
        if conn:
            conn.close()


@app.route('/open_kf_api/get_crawl_url_list', methods=['POST'])
@token_required
def get_crawl_url_list():
    """Fetch the list of URLs and their status information. If the site is specified and valid, returns information for that site. Returns an error if the site is specified but invalid."""
    data = request.json
    site = data.get('site', None)  # Make site an optional parameter

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")

        response_data = {
            'url_list': []
        }

        if site is not None:
            if not is_valid_url(site):
                logger.error(f"Provided site: '{site}' is not a valid URL.")
                return jsonify({'retcode': -10007, 'message': f"Provided site: '{site}' is not a valid URL.", 'data': {}})

            domain = urlparse(site).netloc
            logger.info(f"Fetching URL list for domain: '{domain}'")
            cur.execute("SELECT domain_status FROM t_domain_tab WHERE domain = ?", (domain,))
            domain_status_row = cur.fetchone()
            if domain_status_row:
                response_data['domain_status'] = domain_status_row['domain_status']
            cur.execute("SELECT id, url, content_length, doc_status, version, ctime, mtime FROM t_raw_tab WHERE domain = ?", (domain,))
        else:
            logger.info("Fetching URL list for all domains.")
            cur.execute("SELECT id, url, content_length, doc_status, version, ctime, mtime FROM t_raw_tab")

        rows = cur.fetchall()
        response_data['url_list'] = [dict(row) for row in rows]
        return jsonify({'retcode': 0, 'message': 'Success', 'data': response_data})
    except Exception as e:
        logger.error(f"An error occurred while fetching URL list: {str(e)}")
        return jsonify({'retcode': -10006, 'message': f'An error occurred: {str(e)}', 'data': {}})
    finally:
        if conn:
            conn.close()


def async_crawl_content_task(domain, url_dict, task_type):
    """
    Starts the asynchronous crawl and embedding process for a list of document IDs.

    task_type:
      1 - add_content
      2 - delete_content
      3 - update_content
    """

    """Start the crawl content task in an asyncio event loop."""
    logger.info(f"async_crawl_content_task begin! domain:'{domain}', url_dict:{url_dict}, task_type:{task_type}")
    crawler_content = AsyncCrawlerSiteContent(
        domain=domain,
        sqlite_db_path=f"{SQLITE_DB_DIR}/{SQLITE_DB_NAME}",
        max_requests=MAX_CRAWL_PARALLEL_REQUEST,
        max_embedding_input=MAX_EMBEDDING_INPUT,
        document_embedder_obj=g_document_embedder,
        redis_lock=g_redis_lock
    )

    # Run the crawler
    if task_type == 1:
        asyncio.run(crawler_content.add_content(url_dict))
    elif task_type == 2:
        asyncio.run(crawler_content.delete_content(url_dict))
    elif task_type == 3:
        asyncio.run(crawler_content.update_content(url_dict))
    logger.info(f"async_crawl_content_task end!, domain:'{domain}', task_type:{task_type}")

def check_crawl_content_task(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        data = request.json
        id_list = data.get('id_list')

        if not id_list or not isinstance(id_list, list) or len(id_list) == 0:
            return jsonify({'retcode': -10000, 'message': 'Invalid or missing id_list parameter'})

        conn = None
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")

            placeholders = ', '.join(['?'] * len(id_list))
            cur.execute(f"SELECT id, url FROM t_raw_tab WHERE id IN ({placeholders})", id_list)
            rows = cur.fetchall()

            if len(rows) != len(id_list):
                missing_ids = set(id_list) - set(row[0] for row in rows)
                return jsonify({'retcode': -10009, 'message': f'The following ids do not exist: {missing_ids}', 'data': {}})

            domain = urlparse(rows[0]["url"]).netloc
            logger.info(f"domain is '{domain}'")
            url_dict = {row["id"]: row["url"] for row in rows}

            # Store domain and url_dict in request for further use
            request.domain = domain
            request.url_dict = url_dict

            # Check and update domain_status in t_domain_tab
            cur.execute("SELECT domain_status FROM t_domain_tab WHERE domain = ?", (domain,))
            domain_info = cur.fetchone()
            if domain_info and domain_info["domain_status"] < 3:
                if g_redis_lock.acquire_lock():
                    try:
                        cur.execute("UPDATE t_domain_tab SET domain_status = 3, mtime = ? WHERE domain = ?", (int(time.time()), domain))
                        conn.commit()
                        logger.info(f"Updated domain_status to 3 for domain: '{domain}'")
                    finally:
                        g_redis_lock.release_lock()
        except Exception as e:
            return jsonify({'retcode': -10010, 'message': f'An error occurred: {str(e)}', 'data': {}})
        finally:
            if conn:
                conn.close()
        return f(*args, **kwargs)
    return decorated_function


@app.route('/open_kf_api/add_crawl_url_list', methods=['POST'])
@token_required
@check_crawl_content_task
def add_crawl_url_list():
    domain = request.domain
    url_dict = request.url_dict
    # Use threading to avoid blocking the Flask application
    Thread(target=async_crawl_content_task, args=(domain, url_dict, 1)).start()
    return jsonify({'retcode': 0, 'message': 'Started processing the URL list.', 'data': {}})


@app.route('/open_kf_api/delete_crawl_url_list', methods=['POST'])
@token_required
@check_crawl_content_task
def delete_crawl_url_list():
    domain = request.domain
    url_dict = request.url_dict
    # Use threading to avoid blocking the Flask application
    Thread(target=async_crawl_content_task, args=(domain, url_dict, 2)).start()
    return jsonify({'retcode': 0, 'message': 'Started deleting the URL list embeddings.'})


@app.route('/open_kf_api/update_crawl_url_list', methods=['POST'])
@token_required
@check_crawl_content_task
def update_crawl_url_list():
    domain = request.domain
    url_dict = request.url_dict
    # Use threading to avoid blocking the Flask application
    Thread(target=async_crawl_content_task, args=(domain, url_dict, 3)).start()
    return jsonify({'retcode': 0, 'message': 'Started updating the URL list embeddings.'})


@app.route('/open_kf_api/upload_picture', methods=['POST'])
@token_required
def upload_picture():
    picture_file = request.files.get('picture_file')
    if not picture_file:
        logger.error("Missing required parameters picture_file")
        return jsonify({'retcode': -10001, 'message': 'Missing required parameters picture_file', data:{}})

    try:
        original_filename = secure_filename(picture_file.filename)
        extension = os.path.splitext(original_filename)[1]
        new_filename = f"open_kf_{uuid.uuid4()}{extension}"

        save_directory = f"{STATIC_DIR}/{MEDIA_DIR}/"
        if not os.path.exists(save_directory):
            logger.error(f"save_directory:'{save_directory} does not exist!")
            return jsonify({'retcode': -20001, 'message': 'save_directory does not exitst!', 'data':{}})
        image_path = os.path.join(save_directory, new_filename)
        picture_file.save(image_path)

        picture_url  = f"{URL_PREFIX}{MEDIA_DIR}/{new_filename}"
        return jsonify({'retcode': 0, 'message': 'upload picture success', 'data': {'picture_url': picture_url}})
    except Exception as e:
        logger.error(f"An error occureed: {str(e)}")
        return jsonify({'retcode': -20002, 'message': f'An error occurred: {str(e)}', 'data': {}})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7000)
