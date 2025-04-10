# -*- coding: utf-8-*-

import os
import json

import pandas as pd

from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.components.pdf_loader import PdfLoader
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.retrievers import VectorRetriever

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import TextSplitter

from rich import print

__all__ = [
    "extract_json_from_content",
    "postprocess_rag_completion",
    "generate_knowledge_graph",
    "generate_vector_retriever",
    "generate_entities_and_relationships"
]

def extract_json_from_content(content: str):
    start_index = content.index("```json")
    end_index = len(content) - content[::-1].index("```")
    new_content = content[start_index : end_index+1].strip().replace("```json", "").replace("```", "")

    print(new_content)

    extracted_dict = json.loads(new_content)
    return extracted_dict


def postprocess_rag_completion(completion):
    try:
        completion = completion.answer.split("</think>")[-1].strip()
    except AttributeError:
        completion = completion.content.split("</think>")[-1].strip()
    return completion


def generate_entities_and_relationships():
    # define node labels
    basic_node_labels = [
        "Object",
        "Entity",
        "Group",
        "Person",
        "OrganizationOrInstitution",
        "IdeaOrConcept",
        "GeographicLocation",
    ]

    academic_node_labels = ["ArticleOrPaper", "PublicationOrJournal"]

    node_labels = basic_node_labels + academic_node_labels

    # define relationship types
    relation_types = [
        "WORKS_AT",
        "AFFECTS",
        "ASSESSES",
        "ASSOCIATED_WITH",
        "AUTHORED",
        "CAUSES",
        "CONTRIBUTES_TO",
        "DESCRIBES",
        "EXPRESSES",
        "INCLUDES",
        "INTERACTS_WITH",
        "OUTLINES",
        "PRODUCES",
        "RECEIVED",
        "USED_FOR",
    ]

    return node_labels, relation_types

def load_entities_and_relationships():
    try:
        entities, relationships = load_entities_and_relationships_from_file()
    except:
        entities, relationships = generate_entities_and_relationships()

    return entities, relationships


def load_entities_and_relationships_from_file():
    graph_schema_dir = "../../data/graph_schema/"

    entity_path = os.path.join(
        graph_schema_dir,
        [f for f in os.listdir(graph_schema_dir) if "labels" in f][0]
    )
    entity_df = pd.read_csv(entity_path, header=0)
    entities = entity_df.iloc[:, 0].values.tolist()
    skip_entities = ["__entity__", "chunk", "document", "__kgbuilder__"]
    entities = list(filter(lambda x: x.lower() not in skip_entities, entities))


    relationship_path = os.path.join(
        graph_schema_dir,
        [f for f in os.listdir(graph_schema_dir) if "relationshipTypes" in f][0]
    )
    relationship_df = pd.read_csv(relationship_path, header=0)
    relationships = relationship_df.iloc[:, 0].values.tolist()
    relationships = list(filter(lambda x: "chunk" not in x.lower(), relationships))

    return entities, relationships


async def generate_knowledge_graph(
    path,
    driver,
    embedder,
    llm,
    chunk_size=500,
    chunk_overlap=100,
    generate_schema=False,
):
    prompt_template = """
    You are a cybersecurity researcher with an IQ of 142, tasked with extracting information from documents 
    and structuring it in a property graph to inform further cybersecurity education, research and Q&A.

    Extract the entities (nodes) and specify their type from the following Input text.
    Also extract the relationships between these nodes. the relationship direction goes from the start node to the end node. 

    Return result as a JSON object using the following format:
    {{"nodes": [ {{"id": "0", "label": "the type of entity", "properties": {{"name": "name of entity" }} }}],
      "relationships": [{{"type": "TYPE_OF_RELATIONSHIP", "start_node_id": "0", "end_node_id": "1", "properties": {{"details": "Description of the relationship"}} }}] }}

    - Use only the information from the Input text. Do not add any additional information.  
    - If the input text is empty, return an empty JSON.
    - The JSON returned must be valid JSON.
    - Omit any backticks around the JSON - simply output the JSON on its own.
    - The JSON object must not wrapped into a list - it is its own JSON object.
    - The returned JSON object should have all internal keys and strings surrounded by double quotes.
    - Eliminate the outer backticks (if any). Return the pure JSON. 
    - Make sure to create as many nodes and relationships as needed to offer rich medical context for further research.
    - An AI knowledge assistant must be able to read this graph and immediately understand the context to inform detailed research questions. 
    - Multiple documents may be ingested from different sources and we are using this property graph to connect information, so make sure entity types are fairly general.

    Use only the following nodes and relationships (if provided):
    {schema}

    Assign a unique ID (string) to each node, and reuse it to define relationships.
    Do respect the source and target node types for relationship and
    the relationship direction.

    I repeat: Do not return any additional information other than the JSON in it. Return the JSON and the JSON alone!
    """
    if generate_schema:
        entities, relations = load_entities_and_relationships()
    else:
        entities, relations = None, None

    kg_graph = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        prompt_template=prompt_template,
        entities=entities,
        relations=relations,
        text_splitter=FixedSizeSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        ),
        from_pdf=True,
    )

    pdf_file_names = [f for f in os.listdir(path) if f.endswith(".pdf")]

    # TODO: Limit the number of processed files to speed up process. This wil select the first 5 files as below:
    # pdf_file_names = [
    #     "di-nist-sp1800-11a-draft.pdf",
    #     "di-nist-sp1800-11b-draft.pdf",
    #     "cr-mfa-nist-sp1800-17.pdf",
    #     "derived-piv-nist-sp1800-12-v2.pdf",
    #     "abac-nist-sp1800-3-draft.pdf",
    # ]

    # TODO: Comment out the line below to use all files.
    pdf_file_names = pdf_file_names[:5]

    print(f"Number of PDF files: {len(pdf_file_names)}")

    file_paths = [os.path.join(path, f) for f in pdf_file_names]

    for file_path in file_paths:
        print(f"Processing {file_path}")
        pdf_result = await kg_graph.run_async(file_path=file_path)
        print(f"PDF result: {pdf_result}\n\n")

    return kg_graph


def generate_vector_retriever(
    driver,
    embedder=None,
    dimensions=3584,
    index_name="text_embeddings",
    init_vector_index=False,
):
    if init_vector_index:
        create_vector_index(
            driver,
            name=index_name,
            label="Chunk",
            embedding_property="embedding",
            dimensions=dimensions,
            similarity_fn="cosine",
        )
    vector_retriever = VectorRetriever(
        driver=driver,
        index_name=index_name,
        embedder=embedder,
        return_properties=["text"],
    )
    return vector_retriever

def generate_text_embeddings(
    model_name=None,
    persist_directory=None,
    file_directory=None,
    chunk_size=500,
    chunk_overlap=100,
    generate_vectors=False,
):

    if persist_directory is None:
        persist_directory = "../../data/chromadb"

    if model_name is None:
        model_name = "llama3.2:1b"

    if file_directory is None:
        file_directory = "../../data/nist_cybersecurity_documents"

    # weaviate_client = weaviate.connect_to_local()
    embedding_generator = OllamaEmbeddings(model = model_name)

    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_generator,
    )

    if generate_vectors:

        document_loader = DirectoryLoader(path = file_directory, glob = "*.pdf")
        # splitter = FixedSizeSplitter(
        #     chunk_size=chunk_size,
        #     chunk_overlap=chunk_overlap
        # )

        splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        documents = document_loader.load()
        documents = splitter.split_documents(documents)

        vectordb.add_documents(documents=documents)
        print(documents)
        print(f"Number of documents: {len(documents)}")
    return vectordb.as_retriever()


if __name__ == "__main__":
    a, b = load_entities_and_relationships_from_file()

    print(a)
    print(b)

    print(len(a))
    print(len(b))

    text = """
    Here is a structured approach to assign unique IDs and define relationships in the graph:

### Node Assignments:
1. ACCESS_GRANTED_BASED_ON
2. AUTHORIZES
3. DEFAULT_REQUEST_ACCESS_TO
4. DEPLOYED_TO
5. EquationalLaw
6. ACCESSGranted (Assuming 'ACCESS_GRANTED' was a typo)
7. AUTHORIZED_TO
8. GAINS ACCESS TO
9. ACCESS_GRANTED
10. DEFAULT_REQUEST_ACCESS_TO
11. DEPLOYED_TO
12. EquationalLaw

### Relationships:
- **ACCESS_GRANTED_BASED_ON** → **AUTHORIZES**
- **AUTHORIZES** → **GAINS ACCESS TO**
- **DEPLOYED_TO** → **DEFAULT_REQUEST_ACCESS_TO**
- **EquationalLaw** ↔ **EquationalLaw** (Identity node)
- **ACCESSGranted** → **ACCESS_GRANTED**
- **AUTHORIZED_TO** → **GAINS ACCESS TO**
- **DEFAULT_REQUEST_ACCESS_TO** → **AUTHORIZES**
- **DEPLOYED_TO** → **DEFAULT_REQUEST_ACCESS_TO**

### Summary:
Each node is assigned a unique ID, and relationships are defined based on their labels. This ensures clarity and accuracy in the graph structure.

```json
{
  "nodes": [
    {
      "id": 1,
      "label": "ACCESS_GRANTED_BASED_ON",
      "description": "A node with the label ACCESS_GRANTED_BASED_ON."
    },
    {
      "id": 2,
      "label": "AUTHORIZES",
      "description": "A node with the label AUTHORIZES."
    },
    {
      "id": 3,
      "label": "DEFAULT_REQUEST_ACCESS_TO",
      "description": "A node with the label DEFAULT_REQUEST_ACCESS_TO."
    },
    {
      "id": 4,
      "label": "DEPLOYED_TO",
      "description": "A node with the label DEPLOYED_TO."
    },
    {
      "id": 5,
      "label": "EquationalLaw",
      "description": "A node with the label EquationalLaw."
    },
    {
      "id": 6,
      "label": "ACCESSGranted",
      "description": "A node with the label ACCESSGranted (assuming a typo for ACCESS_GRANTED)."
    },
    {
      "id": 7,
      "label": "AUTHORIZES",
      "description": "Another node with the label AUTHORIZES."
    },
    {
      "id": 8,
      "label": "GAINS ACCESS TO",
      "description": "A node with the label GAINS ACCESS TO."
    },
    {
      "id": 9,
      "label": "ACCESS_GRANTED",
      "description": "Another node with the label ACCESS_GRANTED."
    },
    {
      "id": 10,
      "label": "DEFAULT_REQUEST_ACCESS_TO",
      "description": "Another node with the label DEFAULT_REQUEST_ACCESS_TO."
    },
    {
      "id": 11,
      "label": "DEPLOYED_TO",
      "description": "Another node with the label DEPLOYED_TO."
    },
    {
      "id": 12,
      "label": "EquationalLaw",
      "description": "Another node with the label EquationalLaw."
    }
  ]
}
```
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

LLM response has improper format for chunk_index=3
    """

    print(extract_json_from_content(text))