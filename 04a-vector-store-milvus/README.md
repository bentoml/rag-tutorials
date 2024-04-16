# Integrating Milvus Vector Database

This is the final tutorial of this BentoML example project.

In the [last tutorial](../03-custom-llm/), we replaced the OpenAI question-answering service with a custom LLM. In this tutorial, we will use [Milvus](https://milvus.io/), an open-source vector database, as our storage of documentation index for better performance and scalability.

## Set up Milvus

Before integrating Milvus into our service, we need to start a Milvus database server. The easiest way to do this is with Docker Compose. Run the following commands (with docker-compose installed):

```bash
wget https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh
bash standalone_embed.sh start
```

You can stop the server any time by running:

```bash
bash standalone_embed.sh stop
```

## Integrate Milvus into the BentoML Service

Once the Milvus server is up, modify the `__init__` method in our `RAGService` class to use Milvus as the vector store backend (and set the index to `None` if no index yet). Here’s how to adjust the initialization process:

```diff

+from llama_index.vector_stores.milvus import MilvusVectorStore

 class RAGService:
     llm_service = bentoml.depends(VLLM)

     def __init__(self):

         ...

-        index = VectorStoreIndex.from_documents([])
-        index.storage_context.persist(persist_dir=PERSIST_DIR)
-        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
-        self.index = load_index_from_storage(storage_context)
+        # Initialize Milvus vector store with configuration options. See more options at https://milvus.io/docs/integrate_with_llamaindex.md
+        vector_store = MilvusVectorStore(dim=384, overwrite=False)
+        self.storage_context = StorageContext.from_defaults(vector_store=vector_store)
+        try:
+            self.index = VectorStoreIndex.from_vector_store(vector_store)
+        except ValueError:
+            self.index = None

         ...
```

With Milvus integrated, both the `ingest_pdf` and `ingest_text` APIs need to be updated to insert documents into the Milvus database:

```diff
     def ingest_text(self, txt: Annotated[Path, bentoml.validators.ContentType("text/plain")]) -> str:
         ...

-        self.index.insert(doc)
-        self.index.storage_context.persist(persist_dir=PERSIST_DIR)
+        if self.index is None:
+            self.index = VectorStoreIndex.from_documents(
+                [doc], storage_context=self.storage_context
+            )
+        else:
+            self.index.insert(doc)
+
+        self.index.storage_context.persist()
```

You can now serve the Service using `bentoml serve .` and interact with it at http://localhost:3000.

By migrating to Milvus for vector storage, our RAG web service gains significant improvements in scalability and search efficiency. This allows us to handle larger datasets and more complex queries more efficiently.