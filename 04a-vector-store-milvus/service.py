from __future__ import annotations

import os
import bentoml

from llama_index.core import Document, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai_like import OpenAILike

import os
from pathlib import Path
from typing import Annotated
import openai

from embedding import SentenceTransformers, BentoMLEmbeddings
from llm import VLLM, LLM_MODEL_ID, LLM_MAX_TOKENS
from bentovllm_openai.utils import _make_httpx_client

PERSIST_DIR = "./storage"


@bentoml.service(
    traffic={"timeout": 600},
)
class RAGService:

    embedding_service = bentoml.depends(SentenceTransformers)
    llm_service = bentoml.depends(VLLM)

    def __init__(self):
        self.embed_model = BentoMLEmbeddings(self.embedding_service)

        from llama_index.core import Settings
        self.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
        Settings.node_parser = self.text_splitter
        Settings.embed_model = self.embed_model

        from transformers import AutoTokenizer
        Settings.num_output = 256
        Settings.context_window = LLM_MAX_TOKENS
        Settings.tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_ID
        )

        # more options at https://milvus.io/docs/integrate_with_llamaindex.md
        vector_store = MilvusVectorStore(dim=384, overwrite=False)
        self.storage_context = StorageContext.from_defaults(vector_store=vector_store)
        try:
            self.index = VectorStoreIndex.from_vector_store(vector_store)
        except ValueError:
            self.index = None

        from bentoml._internal.container import BentoMLContainer
        self.vllm_url = BentoMLContainer.remote_runner_mapping.get()["VLLM_OpenAI"]


    @bentoml.api
    def ingest_pdf(self, pdf: Annotated[Path, bentoml.validators.ContentType("application/pdf")]) -> str:

        import pypdf
        reader = pypdf.PdfReader(pdf)
        texts = []
        for page in reader.pages:
            text = page.extract_text()
            texts.append(text)
        all_text = "".join(texts)
        doc = Document(text=all_text)
        if self.index is None:
            self.index = VectorStoreIndex.from_documents(
                [doc], storage_context=self.storage_context
            )
        else:
            self.index.insert(doc)

        self.index.storage_context.persist()
        return "Successfully Loaded Document"

    
    @bentoml.api
    def ingest_text(self, txt: Annotated[Path, bentoml.validators.ContentType("text/plain")]) -> str:

        with open(txt) as f:
            text = f.read()

        doc = Document(text=text)
        if self.index is None:
            self.index = VectorStoreIndex.from_documents(
                [doc], storage_context=self.storage_context
            )
        else:
            self.index.insert(doc)

        self.index.storage_context.persist()
        return "Successfully Loaded Document"


    @bentoml.api
    def query(self, query: str) -> str:
        from llama_index.core import Settings

        httpx_client, base_url = _make_httpx_client(self.vllm_url, VLLM)
        llm = OpenAILike(
            api_base= base_url + "/v1/",
            api_key="no-need",
            is_chat_model=True,
            http_client=httpx_client,
            temperature=0.2,
            model=LLM_MODEL_ID,
        )
        Settings.llm = llm

        query_engine = self.index.as_query_engine()
        response = query_engine.query(query)
        return str(response)
