**1.rag-system-workflow**
**Project Description**

This project focuses on building a semantic search and response generation pipeline using transformer-based models for embedding and natural language generation.

**Objective:**

To encode a large collection of documents into dense vector embeddings, index them for fast similarity search with FAISS, and generate natural language answers from retrieved relevant documents based on user queries.

**Key Steps:**

Embedding Generation:
Use the pre-trained SentenceTransformer model (paraphrase-MiniLM-L6-v2) to convert raw document texts into numerical vector embeddings that capture semantic meaning.

Indexing with FAISS:
Build a FAISS index on the document embeddings for efficient nearest neighbor search using L2 distance.

Semantic Search:
Encode user queries into embeddings, then retrieve the top-k most relevant documents by searching the FAISS index.

Response Generation:
Use a GPT-2 based text generation pipeline to create natural language responses grounded on the retrieved document content.

**Outcome:**

The system enables scalable semantic retrieval and answer generation, allowing users to input natural language queries and receive informative, context-aware responses drawn from a large corpus of documents.

**2.knowledge_graph.ipynb**
**Project Description**

This project demonstrates building and querying a knowledge graph using Neo4j, and generating natural language responses with transformer-based models like GPT-2.

**Objective:**

To create a knowledge graph with entities and relationships using Neo4j, query the graph using Cypher queries, and generate detailed, human-like text answers by feeding retrieved graph information into a GPT-2 text generation pipeline.

**Key Steps:**

Knowledge Graph Construction:
Use the Neo4j Python driver to connect to a Neo4j Aura database and create nodes (e.g., Person, Company) and relationships (e.g., CEO_OF) representing real-world entities and their connections.

Graph Querying:
Execute Cypher queries to retrieve specific information from the graph, such as finding the CEO of a given company.

Text Generation:
Use the Hugging Face Transformers pipeline with GPT-2 to generate natural language responses based on query results from the knowledge graph.

**Outcome:**

**3.hybrid-retrieval-for-rag.ipynb**

The project provides an end-to-end example of integrating a graph database with language models, enabling semantic retrieval of structured data and generation of fluent, context-aware textual responses to user queries.

**Project Description**

This project implements a hybrid document retrieval and response generation pipeline combining dense vector search using SentenceTransformers and FAISS with sparse retrieval using BM25 via Whoosh. It further fine-tunes a BART model for conditional text generation based on retrieved knowledge.

**Objective:**

To efficiently retrieve relevant documents from a large corpus by leveraging both dense embeddings and sparse keyword matching, and generate coherent, context-aware answers by fine-tuning and using the BART transformer model.

**Key Steps:**

Loading Models and Dataset:
Load the Dense Passage Retriever (DPR) question encoder and tokenizer, and download a subset of the Wikimedia Wikipedia dataset to serve as the document corpus.

Embedding Creation and Indexing:
Use SentenceTransformers to encode documents into dense vector embeddings. Build a FAISS index for fast nearest neighbor search in embedding space.

Sparse Retrieval with BM25:
Use Whoosh to create a BM25-based inverted index on the corpus, enabling keyword-based search alongside dense retrieval.

Hybrid Retrieval Function:
Define a function that accepts a user query, computes its dense embedding for FAISS search, and performs BM25 retrieval via Whoosh. Return both dense and sparse retrieval results.

Fine-tuning BART Model:
Load a pre-trained BART model and tokenizer, prepare a simple fine-tuning dataset, and train the model using Hugging Face’s Trainer API to generate text conditioned on retrieved documents.

Generating Responses:
Use the hybrid retrieval to get relevant documents, combine the query with top retrieved documents as input, and generate detailed textual responses with the fine-tuned BART model.

**Outcome:**

This pipeline achieves efficient and accurate retrieval by blending semantic search with keyword-based retrieval. The fine-tuned BART model can generate informed, fluent answers based on the retrieved context, making it suitable for open-domain question answering and knowledge-grounded text generation tasks.
