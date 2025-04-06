# -*- coding: utf-8 -*-

import os
import asyncio
import json
import neo4j

from colorama import Fore

from neo4j_graphrag.llm import OllamaLLM, OpenAILLM, VertexAILLM
from neo4j_graphrag.embeddings import OllamaEmbeddings, OpenAIEmbeddings, VertexAIEmbeddings
from neo4j_graphrag.generation import RagTemplate, GraphRAG

from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.retrievers import VectorRetriever

from rich import print

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

from decouple import config
from rich import print

NEO4J_URI = config("NEO4J_URI")
NEO4J_USERNAME = config("NEO4J_USERNAME")
NEO4J_PASSWORD = config("NEO4J_PASSWORD")
CREATE_KG_GRAPH = False
INIT_INDEX = False

neo4j_driver = neo4j.GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)

model_names = [
    "deepseek-r1:7b",
    "deepseek-r1:1.5b",
    "deepseek-chat",
    "gemini-1.5-flash",
    "llama3.2:1b"
]

model_name = model_names[1]
document_path = "../../data/nist_cybersecurity_documents"
vector_index_name = "vector_index"
# vector_index_name = "vector"
chunk_size = 200
chunk_overlap = 20
TOP_P = 0.9
TEMPERATURE = 0.5


def generate_entities_and_relationships():
    # define node labels
    basic_node_labels = ["Object", "Entity", "Group", "Person", "OrganizationOrInstitution", "IdeaOrConcept", "GeographicLocation"]

    academic_node_labels = ["ArticleOrPaper", "PublicationOrJournal"]

    node_labels = basic_node_labels + academic_node_labels

    # define relationship types
    relation_types = [
        "WORKS_AT", "AFFECTS", "ASSESSES", "ASSOCIATED_WITH", "AUTHORED",
        "CAUSES", "CONTRIBUTES_TO", "DESCRIBES", "EXPRESSES",
        "INCLUDES", "INTERACTS_WITH", "OUTLINES", "PRODUCES", "RECEIVED", "USED_FOR"
    ]

    return node_labels, relation_types


async def generate_knowledge_graph(
    path,
    driver,
    embedder,
    llm,
    chunk_size=500,
    chunk_overlap=100,
    generate_schema=False
):
    prompt_template = '''
    You are a cybersecurity researcher with an IQ of 142, tasked with extracting information from documents 
    and structuring it in a property graph to inform further cybersecurity education, research and Q&A.

    Extract the entities (nodes) and specify their type from the following Input text.
    Also extract the relationships between these nodes. the relationship direction goes from the start node to the end node. 

    Return result as JSON using the following format:
    {{"nodes": [ {{"id": "0", "label": "the type of entity", "properties": {{"name": "name of entity" }} }}],
      "relationships": [{{"type": "TYPE_OF_RELATIONSHIP", "start_node_id": "0", "end_node_id": "1", "properties": {{"details": "Description of the relationship"}} }}] }}

    - Use only the information from the Input text. Do not add any additional information.  
    - If the input text is empty, return an empty JSON. 
    - Make sure to create as many nodes and relationships as needed to offer rich medical context for further research.
    - An AI knowledge assistant must be able to read this graph and immediately understand the context to inform detailed research questions. 
    - Multiple documents may be ingested from different sources and we are using this property graph to connect information, so make sure entity types are fairly general. 

    Use only fhe following nodes and relationships (if provided):
    {schema}

    Assign a unique ID (string) to each node, and reuse it to define relationships.
    Do respect the source and target node types for relationship and
    the relationship direction.

    Do not return any additional information other than the JSON in it.
    '''
    if generate_schema:
        entities, relations = generate_entities_and_relationships()
    else:
        entities, relations = None, None

    kg_graph = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        prompt_template=prompt_template,
        entities=entities,
        relations=relations,
        text_splitter=FixedSizeSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        from_pdf=True
    )
    print(os.getcwd())
    # files = os.listdir(path)
    files = [
        "di-nist-sp1800-11a-draft.pdf",
        "di-nist-sp1800-11b-draft.pdf",
        "cr-mfa-nist-sp1800-17.pdf",
        "derived-piv-nist-sp1800-12-v2.pdf",
        "abac-nist-sp1800-3-draft.pdf"
    ]
    fpaths = [os.path.join(path, f) for f in files]
    for fpath in fpaths:
        print(f"Processing {fpath}")
        pdf_result = await kg_graph.run_async(file_path=fpath)
        print(f"PDF result: {pdf_result}\n\n")

    return kg_graph

def generate_vector_retriever(driver, embedder=None, dimensions=3584, index_name="text_embeddings", init_index=False):
    if init_index:
        create_vector_index(
            driver,
            name=index_name,
            label="Chunk",
            embedding_property="embedding",
            dimensions=dimensions,
            similarity_fn="cosine"
        )
    vector_retriever = VectorRetriever(
        driver=driver,
        index_name=index_name,
        embedder=embedder,
        return_properties=["text"]
    )
    return vector_retriever

class SuperOllamaLLM(OllamaLLM):
    def invoke(self, input, message_history=None, system_instruction=None):
        response = super().invoke(input, message_history=message_history, system_instruction=system_instruction)
        response_text = response.content.split("</think>")[-1].strip()
        resp = response.__class__
        resp.content = response_text[7:-3]
        return resp

# TODO: Instantiate embedder_llm
embedder_llm=OllamaLLM(
   model_name=model_name,
   model_params={
       "response_format": {"type": "json_object"},
       "temperature": TEMPERATURE,
       "top_p": TOP_P,
   }
)

# embedder_llm = OpenAILLM(
#     model_name=model_name,
#     api_key= config("DEEPSEEK_API_KEY"),
#     model_params={
#         "response_format": {"type": "json_object"},
#         "temperature": 0
#     }
# )

# embedder_llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash",
#     api_key=config("GOOGLE_API_KEY"),
#     temperature=0,
#     top_p=.5,
#     safety_settings={
#         HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
#     },
# )

# embedder_llm = VertexAILLM(
#     model_name="gemini-1.5-flash",
#     # api_key=config("GOOGLE_API_KEY"),
#     model_params={
#         "response_format": {"type": "json_object"},
#         "temperature": TEMPERATURE,
#         "top_p": TOP_P
#     },
# )

# TODO: Instantiate embedder

# embedder = OpenAIEmbeddings(
#     model=model_name,
#     api_key=config("DEEPSEEK_API_KEY"),
# )
embedder = OllamaEmbeddings(model=model_name)

# embedder = GoogleGenerativeAIEmbeddings(
#     model="models/embedding-001",
#     google_api_key=config("GOOGLE_API_KEY"),
# )

# embedder = VertexAIEmbeddings(
#     model="gemini-1.5-flash",
# )

num_dimensions = len(embedder.embed_query("Who is Ozymandias?"))
# num_dimensions = 384 # Obtained from vector index online

# TODO: Create knowledge graph, if needed

if CREATE_KG_GRAPH:
    kg_graph = generate_knowledge_graph(
        path=document_path,
        embedder=embedder,
        llm=embedder_llm,
        driver=neo4j_driver,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    kg_graph = asyncio.run(kg_graph)

# TODO: Instantiate vector retriever

vector_retriever = generate_vector_retriever(
    driver=neo4j_driver,
    index_name=vector_index_name,
    embedder=embedder,
    dimensions=num_dimensions,
    init_index=INIT_INDEX
)

# TODO: Instantiate llm for RAG
# 3. GraphRAG Class
llm = OllamaLLM(
    model_name=model_name,
    model_params={
       "temperature": TEMPERATURE,
       "top_p": TOP_P,
   }
)

# llm = OpenAILLM(
#     model_name=model_name,
#     api_key= config("DEEPSEEK_API_KEY"),
#     model_params={
#         "temperature": 0
#     }
# )

# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash",
#     api_key=config("GOOGLE_API_KEY"),
#     temperature=TEMPERATURE,
#     top_p=TOP_P,
#     safety_settings={
#         HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
#     },
# )

# llm = VertexAILLM(
#     model_name="gemini-1.5-flash",
#     # api_key=config("GOOGLE_API_KEY"),
#     model_params={
#         "temperature": TEMPERATURE,
#         "top_p": TOP_P,
#         "api_key": config("GOOGLE_API_KEY")
#     },
# )

# TODO: Instantiate RAG text template

rag_template_text = '''
These are your instructions. Be sure to follow them at all times:

- Answer the Question provided using the following Context, and the Context alone.
- Think through the problem sequentially and critically. Leave no stone unturned.
- Only respond with information mentioned in the Context.
- Do not inject any speculative information not mentioned.
- Be sure that the Question provided is relevant to the Context. Ignore all irrelevant Questions.

# Question:
{query_text}

# Context:
{context}

# Answer:
'''
rag_template = RagTemplate(template=rag_template_text, expected_inputs=['query_text', 'context'])

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

    user_prompt = '''
    Do you know anything about disco music?
    '''

    # vector = embedder.embed_query(user_prompt)
    # print(f"Length of vector: {len(vector)}")

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
