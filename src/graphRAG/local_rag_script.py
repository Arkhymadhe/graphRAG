# -*- coding: utf-8 -*-

import os
import asyncio
import json

from colorama import Fore

from neo4j import GraphDatabase
from neo4j_graphrag.llm import OllamaLLM, OpenAILLM, VertexAILLM
from neo4j_graphrag.embeddings import OllamaEmbeddings, OpenAIEmbeddings
from neo4j_graphrag.generation import RagTemplate, GraphRAG

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

from decouple import config
from rich import print

from preprocessing import (
    generate_entities_and_relationships,
    generate_vector_retriever,
    generate_knowledge_graph,
)

NEO4J_CONNECTION_URI = config("NEO4J_CONNECTION_URI")
NEO4J_USERNAME = config("NEO4J_USERNAME")
NEO4J_PASSWORD = config("NEO4J_PASSWORD")
CREATE_KG_GRAPH = False
INIT_INDEX = False

neo4j_driver = GraphDatabase.driver(
    NEO4J_CONNECTION_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)

try:
    print(neo4j_driver.verify_connectivity())
    print("Connection successful!")
except Exception as e:
    print(f"Failed to connect to Neo4j: {e}")

model_names = [
    "deepseek-r1:7b",
    "deepseek-r1:1.5b",
    "deepseek-chat",
    "gemini-1.5-flash",
    "llama3.2:1b",
]

model_name = model_names[0]
document_path = "../../data/nist_cybersecurity_documents"
vector_index_name = "vector_index"
# vector_index_name = "vector"
chunk_size = 200
chunk_overlap = 20
TOP_P = 0.9
TEMPERATURE = 0.5


class SuperOllamaLLM(OllamaLLM):
    def invoke(self, input, message_history=None, system_instruction=None):
        response = super().invoke(
            input,
            message_history=message_history,
            system_instruction=system_instruction,
        )
        response_text = response.content.split("</think>")[-1].strip()
        resp = response.__class__
        resp.content = response_text[7:-3]
        return resp


# TODO: Instantiate embedder_llm

embedder_llm = OpenAILLM(
    model_name="deepseek/deepseek-chat-v3-0324:free",
    api_key=config("OPEN_ROUTER_DEEPSEEK_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model_params={
        "response_format": {"type": "json_object"},
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    },
)

# TODO: Instantiate embedder

embedder = OllamaEmbeddings(model=model_name)

query = "Who is Ozymandias?"
# print(embedder_llm.invoke(query).content)

num_dimensions = len(embedder.embed_query(query))

# TODO: Create knowledge graph, if needed

if CREATE_KG_GRAPH:
    kg_graph = generate_knowledge_graph(
        path=document_path,
        embedder=embedder,
        llm=embedder_llm,
        driver=neo4j_driver,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    kg_graph = asyncio.run(kg_graph)

# TODO: Instantiate vector retriever

vector_retriever = generate_vector_retriever(
    driver=neo4j_driver,
    index_name=vector_index_name,
    embedder=embedder,
    dimensions=num_dimensions,
    init_index=INIT_INDEX,
)

# TODO: Instantiate llm for RAG
# 3. GraphRAG Class
llm = OllamaLLM(
    model_name=model_name,
    model_params={
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    },
)

# graph_llm = OpenAILLM(
#     model_name="deepseek/deepseek-chat-v3-0324:free",
#     api_key=config("OPEN_ROUTER_DEEPSEEK_API_KEY"),
#     base_url="https://openrouter.ai/api/v1",
#     model_params={
#         "temperature": TEMPERATURE,
#         "top_p": TOP_P,
#     },
# )

# TODO: Instantiate RAG text template

rag_template_text = """
These are your instructions. Be sure to follow them at all times:

- Answer the Question provided using the following Context, and the Context alone.
- Think through the problem sequentially and critically. Leave no logical stone unturned.
- Only respond with information mentioned in the Context.
- Do not inject any speculative information not mentioned.
- Be sure that the Question provided is relevant to the Context. Ignore all irrelevant Questions.

# Question:
{query_text}

# Context:
{context}

# Answer:
"""
rag_template = RagTemplate(
    template=rag_template_text, expected_inputs=["query_text", "context"]
)

# TODO: Instantiate GraphRAG instance
rag = GraphRAG(llm=llm, retriever=vector_retriever, prompt_template=rag_template)


def extract_json_from_content(content):
    start_index = content.index("{")
    end_index = content.index("}")
    extracted_dict = json.loads(content[start_index : end_index + 1])
    return extracted_dict


def postprocess_rag_completion(completion):
    completion = completion.answer.split("</think>")[-1].strip()
    return completion


# 4. Run

if __name__ == "__main__":
    # user_prompt = '''
    # Return a numbered list of all individual authors/contributors to the NIST cybersecurity documents.
    # '''

    # user_prompt = '''
    # Could you help with a highlight of what cybersecurity topics you know? Also, provide direct quotes and references for your sources.
    # '''

    # user_prompt = '''
    # Provide me with a numbered list of articles The MITRE Corporation is associated with.
    # '''

    user_prompt = """
    Do you know anything about disco music?
    """

    vector = embedder.embed_query(user_prompt)
    print(f"Length of vector: {len(vector)}")

    # exit()
    response = rag.search(user_prompt)
    response = postprocess_rag_completion(response)

    print(Fore.BLUE + ">>> User")
    print(Fore.BLUE + f"{user_prompt}\n\n")

    print(Fore.MAGENTA + ">>> Response")
    print(Fore.MAGENTA + f"{response}\n\n" + Fore.RESET)

    # TODO: Not showing retrieved results. Fix bug.

    # vector_res = vector_retriever.get_search_results(
    #     query_text=user_prompt,
    #     top_k=3
    # )
    # for i in vector_res.records:
    #     print("===="*20 + "\n")
    #     print("[bold red]" + json.dumps(i.data(), indent=4) + "[/bold red]")
