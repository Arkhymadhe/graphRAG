# -*- coding: utf-8 -*-

import json
import asyncio

from neo4j import GraphDatabase
from neo4j_graphrag.llm import OllamaLLM, OpenAILLM, LLMResponse
from neo4j_graphrag.embeddings import OllamaEmbeddings
from neo4j_graphrag.generation import RagTemplate, GraphRAG

from decouple import config
from rich import print

from preprocessing import (
    generate_vector_retriever,
    generate_knowledge_graph,
    postprocess_rag_completion,
    extract_json_from_content,
)

NEO4J_CONNECTION_URI = config("AURA_NEO4J_CONNECTION_URI") # TODO: Should be AURA URI. Be sure to restore this once done.
NEO4J_USERNAME = config("NEO4J_USERNAME")
NEO4J_PASSWORD = config("NEO4J_PASSWORD")
OPENROUTER_DEEPSEEK_API_KEY = config("OPENROUTER_DEEPSEEK_API_KEY")
OPENROUTER_BASE_URL = config("OPENROUTER_BASE_URL")

CREATE_KG_GRAPH = False
GENERATE_SCHEMA = False
INIT_VECTOR_INDEX = False

neo4j_driver = GraphDatabase.driver(
    NEO4J_CONNECTION_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)

try:
    neo4j_driver.verify_connectivity()
    print("[green]Connection successful![/green]")
except Exception as e:
    print(f"[red]Failed to connect to Neo4j: {e}[/red]")

model_names = [
    "deepseek-r1:7b",
    "deepseek-r1:1.5b",
    "deepseek-chat",
    "gemini-1.5-flash",
    "llama3.2:1b",
]
open_router_model_names = [
    "deepseek/deepseek-r1-zero:free",
    "deepseek/deepseek-chat:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "deepseek/deepseek-v3-base:free",
    "deepseek/deepseek-r1-distill-qwen-32b:free",
    "deepseek/deepseek-r1-distill-qwen-14b:free",
    "deepseek/deepseek-r1-distill-llama-70b:free",
    "deepseek/deepseek-r1:free",

]

knowledge_graph_model_name = model_names[0]
# TODO: Use a model that matches the graph embedding dimensions online (i.e., 384)
embedder_model_name = "tazarov/all-minilm-l6-v2-f32"

open_router_model_name = open_router_model_names[1]

document_path = "../../data/nist_cybersecurity_documents"
vector_index_name = "vector" # TODO: Use in-built vector index on Neo4J graph builder
chunk_size = 200
chunk_overlap = 20

TOP_P = 0.9
TEMPERATURE = 0.5


class GraphRAGOllamaLLM(OllamaLLM):
    @staticmethod
    def postprocess_response(response):
        response_text = response.content.split("</think>")[-1].strip()

        if "```json" in response_text:
            response_text = json.dumps(extract_json_from_content(response_text))

        new_response = LLMResponse(content=response_text)

        return new_response

    def invoke(self, input, message_history=None, system_instruction=None):
        response = super().invoke(
            input,
            message_history=message_history,
            system_instruction=system_instruction,
        )

        return self.postprocess_response(response)

    async def ainvoke(self, input, message_history=None, system_instruction=None):
        response = await super().ainvoke(
            input,
            message_history=message_history,
            system_instruction=system_instruction,
        )

        return self.postprocess_response(response)


# TODO: Instantiate knowledge_graph_llm

knowledge_graph_llm = GraphRAGOllamaLLM(
    model_name=knowledge_graph_model_name,
    model_params={
        "response_format": {"type": "json_object"},
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    },
)

# TODO: Instantiate embedder

embedder = OllamaEmbeddings(model=embedder_model_name)

query = "Who is Ozymandias?"

num_dimensions = len(embedder.embed_query(query))
print(f"Number of dimensions: {num_dimensions}")

# TODO: Create knowledge graph, if needed

if CREATE_KG_GRAPH:
    kg_graph = generate_knowledge_graph(
        path=document_path,
        embedder=embedder,
        llm=knowledge_graph_llm,
        driver=neo4j_driver,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        generate_schema=GENERATE_SCHEMA
    )

    kg_graph = asyncio.run(kg_graph)

# TODO: Instantiate vector retriever

vector_retriever = generate_vector_retriever(
    driver=neo4j_driver,
    index_name=vector_index_name,
    embedder=embedder,
    dimensions=num_dimensions,
    init_vector_index=INIT_VECTOR_INDEX,
)

# TODO: Instantiate llm for RAG. Try to use local Ollama model for the task
# 3. GraphRAG Class
graph_llm = GraphRAGOllamaLLM(
    model_name=knowledge_graph_model_name,
    model_params={
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    },
)

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
rag = GraphRAG(llm=graph_llm, retriever=vector_retriever, prompt_template=rag_template)


# 4. Run

if __name__ == "__main__":
    user_prompt = '''
    Return a numbered list of all individual authors/contributors to the NIST cybersecurity documents. Provide references from your context.
    '''

    response = rag.search(user_prompt, return_context=True)
    retriever_results = response.retriever_result
    print(retriever_results)
    retriever_results = [r.content for r in retriever_results.items]

    context = [f"\nContext #{i}\n{context}" for i, context in enumerate(retriever_results, 1)]
    context = "\n".join(context)

    response = postprocess_rag_completion(response)

    separator = "\n\n" + "+"*100 + "\n\n"

    print("[blue]>>> User:[/blue]")
    print(f"[blue]{user_prompt}[/blue]\n")

    print("[magenta]>>>> Response:[/magenta]")
    print(f"[magenta]{response}{separator}[green]{context}[/green]\n\n[/magenta]")

    # TODO: Not showing retrieved results. Fix bug. UPDATE: Bug fixed!

    # vector_res = vector_retriever.get_search_results(
    #     query_text=user_prompt,
    #     top_k=3
    # )
    # for i in vector_res.records:
    #     print("===="*20 + "\n")
    #     print("[bold red]" + json.dumps(i.data(), indent=4) + "[/bold red]")
