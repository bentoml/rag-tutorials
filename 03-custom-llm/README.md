# Integrating a Custom LLM

This is the fourth tutorial of this BentoML RAG example project.

In the [last tutorial](../02-custom-embedding/), we replaced the default OpenAI embedding model with our custom model. In this tutorial, we will use our own LLM for the question-answering service.

LlamaIndex supports any LLM that has OpenAI-compatible APIs via [OpenAILike](https://docs.llamaindex.ai/en/stable/api_reference/llms/openai_like/). BentoML provides [an example project BentoVLLM](https://github.com/bentoml/BentoVLLM) of building an LLM service using [vLLM](https://github.com/vllm-project/vllm). It includes utility code to add OpenAI-compatible endpoints. Therefore, we can integrate it into our existing service setup.

## Set up the custom LLM

First, we need to copy `service.py` from the BentoVLLM example to a new file named `llm.py`. Also, bring over the `bentovllm_openai` directory containing the utility code. This is already done, and you can see the entire code in `llm.py` in this directory.

Then, update `service.py` to include the new LLM service. The modifications needed for importing and `__init__` method include:

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

Next, switch the LLM configuration in the `query` method to point to our newly integrated LLM service with `OpenAILike`:

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

Note that here we use a utility function `_make_httpx_client` from BentoVLLM example's utility code. The reason is that when we have multiple services in a BentoML service, the service can call each other's API like calling a Python method. But underlying the calling is an HTTP call over either Unix domain socket (when all services are served on a single machine) or TCP (when services are served distributed on different nodes in a network). BentoML has its own [client](https://docs.bentoml.com/en/latest/guides/clients.html) to handle this difference. However, LlamaIndex uses OpenAI's client to call the LLM service, so we need to replace OpenAI's default HTTPX client with our own client.

With these modifications, the RAG web service now uses completely self-hosted models. You can serve the Service using `bentoml serve .` on a machine with a GPU.

## Next step

While our RAG web service now runs entirely with self-hosted models, the documentation index remains file-based. In the [next tutorial](../04a-vector-store-milvus/), we'll use a dedicated vector database to manage our index more efficiently.
