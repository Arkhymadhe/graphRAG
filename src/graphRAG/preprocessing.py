# -*- coding: utf-8-*-

import os

from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.retrievers import VectorRetriever

from rich import print


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
    """
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
        text_splitter=FixedSizeSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        ),
        from_pdf=True,
    )
    print(os.getcwd())
    # files = os.listdir(path)
    files = [
        "di-nist-sp1800-11a-draft.pdf",
        "di-nist-sp1800-11b-draft.pdf",
        "cr-mfa-nist-sp1800-17.pdf",
        "derived-piv-nist-sp1800-12-v2.pdf",
        "abac-nist-sp1800-3-draft.pdf",
    ]
    fpaths = [os.path.join(path, f) for f in files]
    for fpath in fpaths:
        print(f"Processing {fpath}")
        pdf_result = await kg_graph.run_async(file_path=fpath)
        print(f"PDF result: {pdf_result}\n\n")

    return kg_graph


def generate_vector_retriever(
    driver,
    embedder=None,
    dimensions=3584,
    index_name="text_embeddings",
    init_index=False,
):
    if init_index:
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
