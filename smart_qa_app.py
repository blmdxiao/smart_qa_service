# coding=utf-8
import asyncio
import json
import logging
import os
import sys
import threading
import time

from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI
from loguru import logger
from sqlalchemy.exc import SQLAlchemyError

from admin.models.db import db
from app_config import SQLITE_DB_DIR, SQLITE_DB_NAME
from crawler_module.document_crawler import AsyncCrawler
from crawler_module.document_embedding import DocumentEmbedder
from util.redis_config import redis_client

from admin.token_helper import TokenHelper

# Configure loguru logger
logger.add("error.log", rotation="10 MB")

# Load environment variables from .env file
load_dotenv()

# Retrieve environment variables or set default values
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your_openai_api_key')
CHROMA_DB_DIR = os.getenv('CHROMA_DB_DIR', 'your_chroma_db_directory')
CHROMA_COLLECTION_NAME = os.getenv('CHROMA_COLLECTION_NAME', 'your_collection_name')
GPT_MODEL_NAME = os.getenv('GPT_MODEL_NAME', 'gpt-3.5-turbo')
OPENAI_EMBEDDING_MODEL_NAME = os.getenv('OPENAI_EMBEDDING_MODEL_NAME', 'text-embedding-3-small')
MAX_QUERY_SIZE = int(os.getenv('MAX_QUERY_SIZE', '200'))
RECALL_TOP_K = int(os.getenv('RECALL_TOP_K', '3'))
MAX_HISTORY_QUERY_SIZE = int(os.getenv('MAX_HISTORY_QUERY_SIZE', '5'))
HISTORY_EXPIRE_TIME = int(os.getenv('HISTORY_EXPIRE_TIME', '259200'))
# SQLite URI compatible
WIN = sys.platform.startswith('win')
if WIN:
    prefix = 'sqlite:///'
else:
    prefix = 'sqlite:////'

app = Flask(__name__)

# db orm
basedir = os.path.join(os.getcwd(), f'/{SQLITE_DB_DIR}/')

app.config['SQLALCHEMY_DATABASE_URI'] = prefix + os.path.join(basedir, SQLITE_DB_NAME)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

with app.app_context():
    db.create_all()

# Set OpenAI GPT API key
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize OpenAI embeddings with the specified model
embeddings = OpenAIEmbeddings(
    model=OPENAI_EMBEDDING_MODEL_NAME,
    openai_api_key=OPENAI_API_KEY
)

# Initialize Chroma instance
chroma = Chroma(persist_directory=CHROMA_DB_DIR,
                embedding_function=embeddings,
                collection_name=CHROMA_COLLECTION_NAME
                )
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from admin.models.account import Account
from admin.models.bot_setting import BotSetting
from admin.models.crawl_content_task import CrawlUrlContentTask
from admin.models.crawl_url_task import CrawlUrlsTask
from admin.models.doc_embedding_map import DocEmbeddingMap
from admin.models.response.get_crawl_url_status import CommonResponse, UrlStatusData


def get_user_history(user_id):
    history_key = f"user:{user_id}:history"
    history_items = redis_client.lrange(history_key, 0, -1)
    history = [json.loads(item) for item in history_items]
    return history


def search_and_answer(query, user_id, k=RECALL_TOP_K):
    # Perform similarity search
    # results = chroma.similarity_search_with_relevance_scores(query, k=k)
    t1 = time.time()
    results = chroma.similarity_search_with_score(query, k=k)
    time_cost1 = time.time() - t1
    logger.info(
        f"for the query:'{query}' and user_id:'{user_id}, the top {k} results are: {results} \n*******\nthe time_cost of similarity_search_with_score is {time_cost1}\n*******\n")

    # Build the prompt for GPT
    context = "\n\n".join(
        [f"Document URL: {result[0].metadata['source']}\nContent: {result[0].page_content}" for result in results])

    # Get user history from Redis
    user_history = get_user_history(user_id)
    # Include user history in the prompt
    history_context = "\n--------------------\n".join(
        [f"Previous Query: {item['query']}\nPrevious Answer: {item['answer']}" for item in user_history])
    logger.info(f"for the query:'{query}' and user_id:'{user_id}', the history_context is {history_context}")

    prompt = f"""
    This is a smart customer service bot designed to assist users by providing information based on the content of the OpenIM website and its documentation, encompassing 6 web pages from the main site and 716 documentation pages. The system uses a combination of Language Model Generative Pre-trained Transformer (GPT) and Retriever-Augmented Generation (RAG) with Chroma as the vector database to find the most relevant documents in response to user queries.

    Given the user's previous interactions as described above, consider how their past queries might inform their current needs. This historical context can help tailor the response to be more aligned with their likely interests or unresolved questions from previous interactions.

    When a query is received, it first performs a similarity search to recall the top {k} documents from Chroma. These documents then serve as the context for generating an answer. The aim is to provide users with precise information related to the OpenIM website, enhancing their understanding and usage of the site.

    For general greetings or queries not directly related to the website's content (e.g., "hello", "who are you"), the system should provide a friendly response and guide the user towards making inquiries related to the services or information available on the OpenIM website.

    The goal is to assist users in retrieving information specific to the OpenIM website's offerings and documentation. Therefore, when generating a response, consider the user's actual application scenario and the intent behind their query, as well as any relevant history from their previous queries. Ensure that the response is informative, directly related to the query, and based on the documents provided as context.

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
    - Ensure your answer is relevant to the OpenIM website's content and the user's query history.
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
    response = client.chat.completions.create(
        model=GPT_MODEL_NAME,
        messages=[{"role": "system", "content": prompt}],
        temperature=0
    )
    ret = response.choices[0].message.content
    return ret


@app.route('/open_kf_api/smart_qa', methods=['POST'])
def smart_qa():
    data = request.json
    query = data.get('query', '')
    user_id = data.get('user_id', '')
    if not query or not user_id:
        return jsonify({'retcode': -1, 'message': 'query and user_id are required', 'data': {}}), 400

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
        logger.success(
            f"query:'{query}' and user_id:'{user_id}' is processed successfully, the answer is {answer} \n-------\nthe total time_cost is {time_cost}\n-------\n")

        # After generating the response from GPT
        # Store user query and GPT response in Redis
        history_key = f"user:{user_id}:history"
        history_data = {'query': query, 'answer': answer_json}
        redis_client.lpush(history_key, json.dumps(history_data))
        redis_client.ltrim(history_key, 0, MAX_HISTORY_QUERY_SIZE - 1)  # Keep only the latest N entries
        # Set the expiry time for the history key
        redis_client.expire(history_key, HISTORY_EXPIRE_TIME)

        result = {
            'retcode': 0,
            'message': 'success',
            'data': answer_json
        }
        return jsonify(result)
    except Exception as e:
        logger.error(f"query:'{query}' and user_id:'{user_id}' is processed failed, the exception is {e}")
        return jsonify({'retcode': -2, 'message': str(e), 'data': {}}), 200


@app.route('/open_kf_api/get_token', methods=['POST'])
def get_token():
    data = request.json
    user_id = data.get('user_id')

    # Check if user_id is provided
    if not user_id:
        logger.error("user_id is required but not provided.")
        return jsonify({'retcode': -1, 'message': 'user_id is required', 'data': {}}), 200  # Use 400 for bad request

    try:
        # Generate token
        token = TokenHelper.generate_token(user_id)  # Assuming generate_token is a @staticmethod
        logger.info(f"user_id: '{user_id}' get token successfully, the token is {token}")
        return jsonify({
            'retcode': 0,
            'message': 'success',
            'data': {'token': token}
        }), 200  # Success
    except Exception as e:
        logger.error(f"user_id: '{user_id}' get token failed, the exception is {e}")
        return jsonify(
            {'retcode': -2, 'message': 'Failed to generate token', 'data': {}}), 200  # Use 500 for server error


@app.route('/open_kf_api/login', methods=['POST'])
def login():
    data = request.json
    account_name = data.get('account_name')
    password = data.get('password')

    if not account_name or not password:
        return jsonify({'retcode': -1, 'message': 'Login failed due to missing credentials.', 'data': {}}), 200

    try:
        account_got = db.session.query(Account).filter_by(account_name=account_name).first()
        if account_got and account_got.validate_password(password):
            account_got.login = True
            token = TokenHelper.generate_token(account_name)
            db.session.commit()
            return jsonify({
                'retcode': 0,
                'message': 'success',
                'data': {'token': token}
            }), 200
    except SQLAlchemyError as e:
        logger.error(f"Database error during login attempt for account_name: '{account_name}', exception: {e}")
        return jsonify({'retcode': -2, 'message': 'Login failed due to a server error.', 'data': {}}), 200

    return jsonify({'retcode': -1, 'message': 'Login failed due to incorrect username or password.', 'data': {}}), 200


@app.route('/open_kf_api/logout', methods=['POST'])
def logout():
    try:
        login_response = verify_login()
        if login_response is not None:
            return login_response

        account_name = get_request_account_name_by_token()
        account_got = get_account_by_name(account_name)

        if account_got and account_got.login:
            account_got.login = False
            account_got.token = ''
            db.session.commit()
        else:
            return jsonify({'retcode': -1, 'message': 'login first'}), 200
    except Exception as e:
        logger.error(f"logout failed, the exception is {e}")
        return jsonify({'retcode': -2, 'message': str(e), 'data': {}}), 200

    return jsonify({'retcode': 0, 'message': 'success'}), 200


def verify_login():
    token = get_request_token()
    if not token:
        return jsonify({"retcode": -1, "message": "token is none", "data": {}}), 400
    verify_token_payload = TokenHelper.verify_token(token)
    name = verify_token_payload["user_id"]
    account_by_name = get_account_by_name(name)
    if not account_by_name or not account_by_name.is_login():
        return jsonify({'retcode': -1, 'message': 'login first'}), 400
    return None


def get_request_account_name_by_token():
    token = get_request_token()
    if not token:
        return jsonify({"retcode": -1, "message": "token is none", "data": {}}), 400
    verify_token_payload = TokenHelper.verify_token(token)
    return verify_token_payload["user_id"]


def get_account_by_name(account_name):
    try:
        account_got = Account.query.filter_by(username=account_name).first()
        return account_got
    except Exception as e:
        logger.error(f"An error occurred while fetching account by name '{account_name}': {e}")
        return None


def get_request_token():
    auth_header = request.headers.get('Authorization')
    if not auth_header or 'Bearer ' not in auth_header:
        logger.error("Authorization header missing or invalid")
        return None
    token = auth_header.split(' ')[1]
    return token


@app.route('/open_kf_api/update_password', methods=['POST'])
def update_password():
    try:
        login = verify_login()
        if not login:
            return login
        data = request.get_json()
        name = get_account_by_name(data["name"])
        npd = data.get("new_password")
        cpd = data.get('current_password')
        account_got = Account.query.filter_by(account_name=name).first()

        if not account_got or not account_got.validate_password(cpd):
            return jsonify({
                'retcode': -1,
                'message': 'Account not found or current password is incorrect.',
            }), 200

        account_got.password(npd)
        db.session.commit()
        return jsonify({
            'retcode': 0,
            'message': 'success',
        }), 200

    except Exception as e:
        logger.error(f"update password failed, the exception is {e}")
        return jsonify({'retcode': -2, 'message': str(e), 'data': {}}), 200


@app.route('/open_kf_api/crawl_urls', methods=['POST'])
def crawl_urls():
    login_verify_message = verify_login()
    if login_verify_message:
        return login_verify_message

    data = request.get_json()
    site_url = data.get('site')

    try:
        task_got = db.session.query(CrawlUrlsTask).filter_by(site_url=site_url).first()

        if not task_got:
            new_task = CrawlUrlsTask(site_url=site_url, version=1, status=1)
            db.session.add(new_task)
            db.session.commit()
        else:
            if task_got.status != 2:
                return jsonify({'retcode': -1, 'message': 'The site was submitted already'}), 200
            task_got.version += 1
            db.session.commit()

        crawler = AsyncCrawler(site_url)
        thread = threading.Thread(target=start_async_crawl_and_wait, args=(crawler,))
        thread.start()

    except Exception as e:
        logger.error(f"Add site task failed, the exception is {e}")
        return jsonify({'retcode': -2, 'message': str(e), 'data': {}}), 200
    return jsonify({'retcode': 0, 'message': 'Success'}), 200


def start_async_crawl_and_wait(crawler):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    future = asyncio.ensure_future(crawler.run())
    loop.run_until_complete(future)
    # update status=2
    task = db.session.query(CrawlUrlsTask).filter_by(site_url=crawler.base_url).first()
    if task:
        task.status = 2
        db.session.commit()
    loop.close()


@app.route('/open_kf_api/embedding_urls_content', methods=['POST'])
def embedding_urls_content():
    login_response = verify_login()
    if login_response:
        return login_response

    data = request.get_json()
    url_ids = data.get("urls", [])

    if not url_ids:
        return jsonify({'retcode': -1, 'message': "No URL IDs provided.", 'data': {}}), 200

    try:
        embedder = DocumentEmbedder()
        preprocessed_data = embedder.fetch_preprocessed_data_by_ids(url_ids)

        if not preprocessed_data:
            return jsonify({'retcode': -1, 'message': "No data found for provided IDs.", 'data': {}}), 200

        embedder.compute_and_store_embeddings(preprocessed_data)
        return jsonify({'retcode': 0, 'message': "Success", 'data': {}}), 200

    except Exception as e:
        logger.error(f"Processing site task failed, the exception is {e}")
        return jsonify({'retcode': -2, 'message': "Processing failed due to an error.", 'data': {}}), 200


@app.route('/open_kf_api/get_crawl_url_status', methods=['post'])
def get_crawl_url_status():
    try:
        login_response = verify_login()
        if login_response:
            return login_response

        data = request.get_json()
        site = data.get("site")

        urls_task = CrawlUrlsTask.query.filter_by(site=site).first()
        if not urls_task:
            return jsonify(CommonResponse(retcode=0, message="success", data=None)), 200

        if urls_task.status == 1:
            response_data = UrlStatusData(status=1, urls=None)
            return jsonify(CommonResponse(retcode=0, message="success", data=response_data)), 200

        elif urls_task.status == 2:
            site_content_all = CrawlUrlContentTask.query.filter_by(site=site, base_url_id=urls_task.id).all()
            urls = [
                {"url_id": content.id, "status": content.doc_status, "url": content.url,
                 "length": content.content_length}
                for content in site_content_all
            ]
            status_data = UrlStatusData(status=2, urls=urls)
            return jsonify(CommonResponse(retcode=0, message="success", data=status_data)), 200

    except Exception as e:
        logger.error(f"Fetching crawl url status failed, exception: {e}")
        return jsonify(CommonResponse(retcode=-2, message=str(e), data=None)), 200

    return jsonify(CommonResponse(retcode=-1, message="Unhandled condition", data=None)), 200


@app.route('/open_kf_api/get_all_embedding_url', methods=['POST'])
def get_all_embedding_url():
    try:
        login_verify = verify_login()
        if not login_verify:
            return login_verify

        page = request.json.get('page', 1)
        size = request.json.get('size', 10)

        total = DocEmbeddingMap.query.count()

        embedding_map_query_all = DocEmbeddingMap.query.paginate(page, size, False)

        if not embedding_map_query_all.items:
            return jsonify(CommonResponse(retcode=0, message="No embedding URLs found.", data=None, total=total)), 200

        ids = [embedding_map.id for embedding_map in embedding_map_query_all.items]
        urls_all = CrawlUrlContentTask.query.filter(CrawlUrlContentTask.id.in_(ids)).all()
        if not urls_all:
            return jsonify(CommonResponse(retcode=0, message="No content URLs found.", data=None, total=total)), 200

        urls = [url.url for url in urls_all]
        response_data = {'urls': urls, 'total': total}
        return jsonify(CommonResponse(retcode=0, message="Success", data=response_data)), 200

    except Exception as e:
        logger.error(f"Fetching embedding URLs failed: {e}")
        return jsonify({'retcode': -2, 'message': str(e), 'data': {}}), 200


@app.route('/open_kf_api/delete_url', methods=['post'])
def delete_url():
    try:
        login_verify = verify_login()
        if not login_verify:
            return login_verify

        data = request.get_json()
        urls = data.get("urls")
        if not urls:
            return jsonify({'retcode': -1, 'message': 'No URLs provided', 'data': {}}), 200

        urls_all = CrawlUrlContentTask.query.filter(CrawlUrlContentTask.url.in_(urls)).all()
        ids = [url.id for url in urls_all]

        if ids:
            DocumentEmbedder().delete_documents(ids)
            return jsonify({'retcode': 0, 'message': 'Success', 'data': {}}), 200
        else:
            return jsonify({'retcode': -1, 'message': 'No matching URLs found', 'data': {}}), 200
    except Exception as e:
        logger.error(f"Delete URLs failed: {e}")
        return jsonify({'retcode': -2, 'message': str(e), 'data': {}}), 200


@app.route('/open_kf_api/get_settings', methods=['post'])
def get_settings():
    try:
        login_msg = verify_login()
        if not login_msg:
            return login_msg
        first = BotSetting.query.filter().first()
        return jsonify(CommonResponse(data=first, retcode=0, message="success")), 200
    except Exception as e:
        logger.error(f"add site task failed, the exception is {e}")
        return jsonify({'retcode': -2, 'message': str(e), 'data': {}}), 200


@app.route('/open_kf_api/update_setting', methods=['post'])
def update_setting():
    try:
        login_msg = verify_login()
        if not login_msg:
            return login_msg
        data = request.get_json()

        first = BotSetting.query.filter().first()
        first.init_message = data['initMessage']
        first.suggested_messages = data['suggested_messages']
        first.bot_name = data['bot_name']
        first.bot_avatar = data['bot_avatar']
        first.chat_icon = data['chat_icon']
        first.placeholder = data['placeholder']
        first.model = data['model']
        first.mtime = int(time.time())
        return jsonify(CommonResponse(data=first, retcode=0, message="success")), 200
    except Exception as e:
        logger.error(f"add site task failed, the exception is {e}")
        return jsonify({'retcode': -2, 'message': str(e), 'data': {}}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
