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
    #results = chroma.similarity_search_with_relevance_scores(query, k=k)
    results = chroma.similarity_search_with_score(query, k=k)
    logger.info(f"for the query:'{query}', the top {k} results are: {results}")

    # Build the prompt for GPT
    context = "\n\n".join([f"Document URL: {result[0].metadata['source']}\nContent: {result[0].page_content}" for result in results])
    prompt = f"""
    Given the information from the documents listed below, please formulate a detailed and specific answer to the query in the same language as the query. Your response should be in JSON format, containing 'answer' and 'source' fields. The 'answer' field must include a precise and informative response based on the document contents, matching the language of the query. The 'source' field should list the URLs of the documents that directly support your answer. If the documents do not provide sufficient information for a definitive answer, please indicate that the answer is unknown in the 'answer' field.

    Documents:
    {context}

    Query: '{query}'

    Instructions for response:
    - Ensure your answer is in the same language as the query.
    - Provide a detailed and specific answer based on the information found in the documents.
    - Avoid speculative or general responses not supported by the document contents.
    - Respond in a manner that aligns with the query's language (e.g., if the query is in Chinese, respond in Chinese; if in English, respond in English; and so on for other languages).

    Please format your response as follows:
    {{
      "answer": "Provide a detailed and specific answer here, in the same language as the query.",
      "source": ["URL(s) of the document(s) supporting your answer"]
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

