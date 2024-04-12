# Rag Web Service with custom LLM service

In [last section](../02-custom-embedding/) we replace the default OpenAI embedding model with our own model. In this section, we want to use our own LLM model to do question-answering. LlamaIndex supports any LLM model service that has OpenAI-compatible API via [OpenAI like LLM](https://docs.llamaindex.ai/en/stable/api_reference/llms/openai_like/).

BentoML has [an example](https://github.com/bentoml/BentoVLLM) showing how to make an LLM service using [vLLM](https://github.com/vllm-project/vllm). This example includes utility codes to add OpenAI-compatible endpoints to the LLM service. We can just copy the `service.py` from the example to `llm.py` in our code base, then also copy the utility codes `bentovllm_openai/` to our code base. Then we can import the LLM service in our `service.py`. The modifications needed for importing and `__init__` method is:

```diff
+from llama_index.llms.openai_like import OpenAILike
+from llm import VLLM, LLM_MODEL_ID, LLM_MAX_TOKENS
+from bentovllm_openai.utils import _make_httpx_client

 ...

 class RAGService:

     embedding_service = bentoml.depends(SentenceTransformers)
+    llm_service = bentoml.depends(VLLM)

     def __init__(self):
         openai.api_key = os.environ.get("OPENAI_API_KEY")
         Settings.node_parser = self.text_splitter
         Settings.embed_model = self.embed_model

+        from transformers import AutoTokenizer
+        Settings.num_output = 256
+        Settings.context_window = LLM_MAX_TOKENS
+        Settings.tokenizer = AutoTokenizer.from_pretrained(
+            LLM_MODEL_ID
+        )
+
         index = VectorStoreIndex.from_documents([])
         index.storage_context.persist(persist_dir=PERSIST_DIR)
         storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
         self.index = load_index_from_storage(storage_context)

+        from bentoml._internal.container import BentoMLContainer
+        self.vllm_url = BentoMLContainer.remote_runner_mapping.get()["VLLM_OpenAI"]
+

```

Then when do the query we need to change LlamaIndex's LLM to `OpenAILike` pointing to our LLM service:

```diff
     def query(self, query: str) -> str:
+        from llama_index.core import Settings
+
+        httpx_client, base_url = _make_httpx_client(self.vllm_url, VLLM)
+        llm = OpenAILike(
+            api_base=base_url + "/v1/",
+            api_key="no-need",
+            is_chat_model=True,
+            http_client=httpx_client,
+            temperature=0.2,
+            model=LLM_MODEL_ID,
+        )
+        Settings.llm = llm
+
         query_engine = self.index.as_query_engine()
         response = query_engine.query(query)
         return str(response)
```

Here we use a utility function `_make_httpx_client` from BentoVLLM example's utility codes. The reason is when we have multiple services in a BentoML service, the service can call each other's API like calling a Python method. But underlying the calling is an HTTP call over either Unix domain socket (when all services are served on a single machine) or TCP (when services are served distributed on different nodes in a network). BentoML has its own [client](https://docs.bentoml.com/en/latest/guides/clients.html) to handle this difference. However, LlamaIndex uses OpenAI's client to call the LLM service, hence we need to replace OpenAI's default HTTPX client with our own client.

After these changes, we can run `bentoml serve .` on a machine with a GPU card.

Now our service uses completely self-hosted models. However, the documentation index is written and loaded from the file system. In [next section](../04a-vector-store-milvus/), we will use a proper vector database to store the index.
