# coding=utf-8
import json
import os
import time
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL_NAME, GPT_MODEL_NAME, CHROMA_DB_DIR, CHROMA_COLLECTION_NAME


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
                collection_name=CHROMA_COLLECTION_NAME)


def search_and_answer(query, k=3):
    # Perform similarity search
    results = chroma.similarity_search_with_relevance_scores(query, k=k)

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
    return json.loads(ret)


if __name__ == "__main__":
    vec = ["实时音视频通讯能力的介绍", "Introduction to real-time audio and video communication capabilities", "OpenIM 介绍", "OpenIM introduction", "Introduction to OpenIM", "世界杯的决赛时间", "The final match time of the World Cup", "晚上去哪约会吃饭", "Where to go for a dinner date tonight"]
    for query in vec:
        begin = time.time()
        answer = search_and_answer(query, k=3)
        end = time.time()
        time_cost = end - begin
        print(f"query={query}")
        print(f"answer={answer}")
        print(f"time_cost={time_cost}\n\n")

