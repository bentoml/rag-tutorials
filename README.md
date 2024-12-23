<div align="center">
    <h1 align="center">Self-Hosted RAG Web Service with BentoML</h1>
</div>

This is a BentoML example project, containing a series of tutorials where we build a complete self-hosted Retrieval-Augmented Generation (RAG) application, step-by-step.

This project will guide you through setting up a RAG service that uses vector-based search and large language models (LLMs) to answer queries using documents as a knowledge base. Our ultimate goal is to create a system that can scale efficiently and handle complex queries with high performance.

See [here](https://docs.bentoml.com/en/latest/examples/overview.html) for a full list of BentoML example projects.

## Project overview

This repository contains a series of five tutorials designed to progressively build a RAG system with custom embedding and language models as well as a vector database.

0. [Building a Simple RAG System using LlamaIndex](00-simple-local-rag/): Set up a basic RAG system that runs locally on your machine using LlamaIndex. This serves as a foundational step, familiarizing you with the basic components of a RAG system.
1. [Transforming a Local RAG into a BentoML Web Service](01-simple-rag/): Convert the local script into a web service by setting up a basic API service using BentoML.
2. [Integrating a Custom Embedding Service](02-custom-embedding/): Replace the default OpenAI embedding model used in the RAG system with a custom model.
3. [Integrating a Custom LLM](03-custom-llm/): Replace the default OpenAI question-answering part in the RAG system with a custom LLM.
4. [Integrating Milvus Vector Database](04a-vector-store-milvus/): Implement Milvus to manage the documentation index for better scalability and performance.

## Set up the environment

To begin, clone the entire project.

```bash
git clone https://github.com/bentoml/rag-tutorials.git
cd rag-tutorials
```

Next, set up the Python environment required for running the tutorials:

```bash
# Recommend Python 3.11
python3 -m venv rag-bentoml && . rag-bentoml/bin/activate && pip install -r requirement.txt
```

## Get started

Each tutorial is self-contained and includes instructions on setting up and running the components discussed. Start with the [first tutorial](00-simple-local-rag/) and proceed through each to build upon the previous steps. By the end of the series, you will have a better understanding of how to build a RAG system using modern technologies and custom integrations.

Have fun!
