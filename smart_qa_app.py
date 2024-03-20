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
    This is a smart customer service bot designed to assist users by providing information based on the content of the OpenIM website and its documentation, encompassing 6 web pages from the main site and 716 documentation pages. The system uses a combination of Language Model Generative Pre-trained Transformer (GPT) and Retriever-Augmented Generation (RAG) with Chroma as the vector database to find the most relevant documents in response to user queries.

    When a query is received, it first performs a similarity search to recall the top {k} documents from Chroma. These documents then serve as the context for generating an answer. The aim is to provide users with precise information related to the OpenIM website, enhancing their understanding and usage of the site.

    For general greetings or queries not directly related to the website's content (e.g., "hello", "who are you"), the system should provide a friendly response and guide the user towards making inquiries related to the services or information available on the OpenIM website.

    The goal is to assist users in retrieving information specific to the OpenIM website's offerings and documentation. Therefore, when generating a response, consider the user's actual application scenario and the intent behind their query. Ensure that the response is informative, directly related to the query, and based on the documents provided as context.

    **It is crucial to provide responses that are as detailed and comprehensive as possible.** When the query indicates a need for specific information, such as URLs, steps, or example code, **the response should aim to include all such details**. Use the context from the top recalled documents to form a thorough answer, leveraging any available specifics to enhance the relevance and usefulness of the response.

    Additionally, **if the query's true intent seems to extend beyond the literal question asked, strive to address the underlying need or interest.** This means not only answering the direct question but also providing additional information that could be helpful to the user based on the context and intent of the query.

    When generating code snippets or examples, **it is essential to ensure the code is not only correct but also well-formatted, with proper indentation and spacing for enhanced readability and adherence to coding standards.** This helps users easily understand and apply the code within their own projects.

    Given the information from the documents listed below, please formulate a detailed and specific answer to the query in the same language as the query. Your response should be in JSON format, containing 'answer' and 'source' fields. The 'answer' field must include a precise and informative response based on the document contents, matching the language of the query. The 'source' field should list the URLs of the documents that directly support your answer. If the documents do not provide sufficient information for a definitive answer, please indicate that the answer is unknown in the 'answer' field.

    Documents:
    {context}

    Query: '{query}'

    Instructions for response:
    - Ensure your answer is relevant to the OpenIM website's content.
    - Provide a detailed and specific answer based on the information found in the documents, including URLs, steps, example code, and any other specifics requested in the query.
    - For general inquiries or unrelated questions, offer a standard response that encourages users to ask more specific questions related to the website.
    - Respond in a manner that considers the user's intent and the practical application of the query, addressing not just the literal question but also the broader context and potential needs.
    - When providing code examples, ensure the code is correct and follows best practices for formatting and indentation to promote readability and maintainability.
    - Avoid speculative or general responses not supported by the document contents.
    - Respond in a manner that aligns with the query's language (e.g., if the query is in Chinese, respond in Chinese; if in English, respond in English; and so on for other languages).

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

