# coding=utf-8
import json
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI
from loguru import logger


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
RECALL_TOP_K = int(os.getenv('RECALL_TOP_K', '4'))

app = Flask(__name__)

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

def search_and_answer(query, k=RECALL_TOP_K):
    # Perform similarity search
    results = chroma.similarity_search_with_relevance_scores(query, k=k)
    logger.info(f"The top {k} result is ${results}")

    # Build the prompt for GPT
    context = "\n\n".join([f"Document URL: {result[0].metadata['source']}\nContent: {result[0].page_content}" for result in results])
    prompt = f"""
    Based on the following documents, please provide an answer to the query in JSON format, including 'answer' and 'source' fields. 'answer' should be a string containing the response to the query. 'source' should be a list of URLs from the documents that most directly provide the answer. If the documents do not provide enough information to answer, the 'answer' field should state that the answer is unknown. Be sure to respond in the same language as the query.

    Documents:
    {context}

    Query: '{query}'

    Please format your response as follows:
    {{
      "answer": "Your answer here.",
      "source": ["URL of the document that provides the answer"]
    }}
    """

    # Call GPT model to generate an answer
    response = client.chat.completions.create(
            model=GPT_MODEL_NAME,
            messages=[{"role": "system", "content": prompt}],
            temperature=0
        )
    ret = response.choices[0].message.content
    return ret


@app.route('/smart_qa', methods=['POST'])
def smart_qa():
    data = request.json
    query = data.get('query', '')
    if not query:
        return jsonify({'retcode': -1, 'message': 'Query is required', 'data': {}}), 400

    if len(query) > MAX_QUERY_SIZE:
        query = query[:MAX_QUERY_SIZE]
    
    try:
        answer = search_and_answer(query)
        answer_json = json.loads(answer)
        logger.success(f"query:'{query}' is processed successfully, the answer is {answer}")
        result = {
            'retcode': 0,
            'messge': 'success',
            'data': answer_json
        }
        return jsonify(result)
    except Exception as e:
        logger.error(f"query:'{query}' is processed failed, the exception is {e}")
        return jsonify({'retcode': -2, 'message': str(e), 'data': {}}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

