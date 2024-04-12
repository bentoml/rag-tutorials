# Rag Web Service with Custom Embedding Service

In [last section](../01-simple-rag/) we made a RAG web service. In this section, we want to use our own model to do text embedding. To make LlamaIndex use a custom embedding model, we need to do two things:

1. create our own embedding web service.
2. wrap the embedding service in a class to make LlamaIndex recognize it (reference: <https://docs.llamaindex.ai/en/stable/examples/embeddings/custom_embeddings/>).

BentoML has [an example](https://github.com/bentoml/BentoSentenceTransformers/) showing how to make a text embedding service using [SentenceTransformers](https://sbert.net). Because BentoML's services are composable, we can copy the Python source code as `embedding.py`, add a wrapper class similar to LlamaIndex's [example codes](https://docs.llamaindex.ai/en/stable/examples/embeddings/custom_embeddings/) and import this embedding services directly in our `service.py`. The modifications in `service.py` only need the following lines:

```diff
+ # import embedding service and wrapper class in service.py
+ from embedding import SentenceTransformers, BentoMLEmbeddings

...

 class RAGService:

+    embedding_service = bentoml.depends(SentenceTransformers)
+
     def __init__(self):
         openai.api_key = os.environ.get("OPENAI_API_KEY")
+        self.embed_model = BentoMLEmbeddings(self.embedding_service)

         from llama_index.core import Settings
         self.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
         Settings.node_parser = self.text_splitter
+        Settings.embed_model = self.embed_model

```

We still need the OpenAI key because we use ChatGPT for the question-answering part. In [next section](../03-custom-llm/) we will try to replace this part with our own LLM model.
