# Rag Web Service with Milvus Vector Database

In this section, we want to use a [Milvus] vector database as our storage of documentation index. We need to first start a Milvus db server first. The simplest way to do that is by running the following commands (with docker-compose installed):

```bash
wget https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh
bash standalone_embed.sh start
```

You can stop the server anytime by running:

```bash
bash standalone_embed.sh stop
```

With Milvus server ready, we can modify `__init__` method of our service to get the index from Milvus db (and set the index to None if no index yet):

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
+        # more options at https://milvus.io/docs/integrate_with_llamaindex.md
+        vector_store = MilvusVectorStore(dim=384, overwrite=False)
+        self.storage_context = StorageContext.from_defaults(vector_store=vector_store)
+        try:
+            self.index = VectorStoreIndex.from_vector_store(vector_store)
+        except ValueError:
+            self.index = None

         ...
```

For both `ingress_pdf` and `ingress_text` endpoints, we will insert the document into Milvus db:

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
