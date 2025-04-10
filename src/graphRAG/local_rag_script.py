# -*- coding: utf-8 -*-

import asyncio
from colorama import Fore

from neo4j import GraphDatabase
from neo4j_graphrag.llm import OllamaLLM, LLMResponse
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

NEO4J_CONNECTION_URI = config("LOCAL_NEO4J_CONNECTION_URI")
NEO4J_USERNAME = config("NEO4J_USERNAME")
NEO4J_PASSWORD = config("NEO4J_PASSWORD")

CREATE_KG_GRAPH = True
GENERATE_SCHEMA = False
INIT_VECTOR_INDEX = True

print("="*100 + "\n" + "Connection Details:\n")

print(f"NEO4J_CONNECTION_URI: {NEO4J_CONNECTION_URI}")
print(f"NEO4J_USERNAME: {NEO4J_USERNAME}")
print(f"NEO4J_PASSWORD: {NEO4J_PASSWORD}")

print("="*100 + "\n")

neo4j_driver = GraphDatabase.driver(
    NEO4J_CONNECTION_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)

try:
    neo4j_driver.verify_connectivity()
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

model_name = model_names[1]
document_path = "../../data/nist_cybersecurity_documents"
vector_index_name = "vector_index"

chunk_size = 200
chunk_overlap = 20

TOP_P = 0.9
TEMPERATURE = 0.1


class GraphRAGOllamaLLM(OllamaLLM):
    @staticmethod
    def postprocess_response(response):
        response_text = response.content.split("</think>")[-1].strip()
        new_response = LLMResponse(content=response_text)

        print("\n" + "+"*100)
        print("[blue]" + response_text + "[/blue]")
        print("+"*100 + "\n")
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


# TODO: Instantiate embedder_llm

knowledge_graph_llm = GraphRAGOllamaLLM(
    model_name=model_name,
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
        llm=knowledge_graph_llm,
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
    init_vector_index=INIT_VECTOR_INDEX,
)

# TODO: Instantiate llm for RAG
# 3. GraphRAG Class
graph_llm = GraphRAGOllamaLLM(
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
rag = GraphRAG(llm=graph_llm, retriever=vector_retriever, prompt_template=rag_template)


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

    # user_prompt  = """
    # Generate a JSON object detailing the biometrics of Napoleon Bonaparte. The expected keys are: date_of_birth,
    # data_of_death, battles_won, and battles_lost. Take note of the following:
    #
    # - The returned JSON object should have all internal keys and strings surrounded by double quotes.
    # - Eliminate the outer backticks (if any). Return the pure JSON.
    # """
    #
    # response = asyncio.run(knowledge_graph_llm.ainvoke(user_prompt))
    #
    # print(response.content)
    # import json
    # json_response = json.loads(response.content.split("</think>")[1].strip())
    # print(json_response)
