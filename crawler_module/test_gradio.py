# coding=utf-8
import json
import os
import time
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI
import gradio as gr
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
    return json.loads(ret)

def gradio_interface(query):
    answer = search_and_answer(query, k=3)
    formatted_answer = f"Answer: {answer['answer']}\nSource: {', '.join(answer['source'])}"
    return formatted_answer

iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(lines=2, placeholder="Enter your query here..."),
    outputs="text",
    title="Document Search and Answer System",
    description="This system searches documents and provides answers based on the content of the documents. Please enter your query below."
)

iface.launch(share=True)
