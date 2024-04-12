In this repository, we have a series of tutorials to implement a complete self-hosted RAG application step by step:

- [simple local rag script](00-simple-local-rag/)
- [simple RAG web service](01-simple-rag/)
- [RAG with custom embedding model](02-custom-embedding/)
- [RAG with custom LLM model](03-custom-llm/)

Then we can use the following vector database to host our documentation index:

- [Milvus](04a-vector-store-milvus/)

To set up the Python environment, run:

```bash
python3 -m venv venv && . venv/bin/activate && pip install -r requirements.txt
```
