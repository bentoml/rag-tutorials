from __future__ import annotations

import typing as t

import numpy as np
import bentoml

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


@bentoml.service(
    traffic={"timeout": 600},
    resources={"memory": "2Gi"},
)
class SentenceTransformers:

    def __init__(self) -> None:

        import torch
        from sentence_transformers import SentenceTransformer, models
        
        # Load model and tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # define layers
        first_layer = SentenceTransformer(MODEL_ID)
        pooling_model = models.Pooling(first_layer.get_sentence_embedding_dimension())
        self.model = SentenceTransformer(modules=[first_layer, pooling_model])
        print("Model loaded, ", "device:", self.device)

    @bentoml.api(batchable=True)
    def encode(
        self,
        sentences: t.List[str],
    ) -> np.ndarray:
        print("encoding sentences:", len(sentences))
        # Tokenize sentences
        sentence_embeddings= self.model.encode(sentences)
        return sentence_embeddings


class BentoMLEmbeddings(BaseEmbedding):
    _model: bentoml.Service = PrivateAttr()

    def __init__(self, embed_model: bentoml.Service, **kwargs) -> None:
        super().__init__(**kwargs)
        self._model = embed_model
        
    def sync_client(self, query: str):

        response = {}
        if isinstance(query, list):
            response = self._model.encode(sentences=query)
        else:
            response = self._model.encode(sentences=[query])  
        return response
    
    async def async_client(self, query: str):
        response = {}
        if isinstance(query, list):
            response = await self._model.encode(sentences=query)
        else:
            response = await self._model.encode(sentences=[query])  
        return response
    
    async def _aget_query_embedding(self, query: str):
        res = await self.async_client(query)
        return res[0].tolist()
    
    def _get_query_embedding(self, query: str):
        return self.sync_client(query)[0].tolist()
    
    def _get_text_embedding(self, text):
        if isinstance(text, str):
            return self.sync_client(text).tolist()
        else:
            return self.sync_client(text[0].get_text()).tolist()

    def _get_text_embeddings(self, text):
        if isinstance(text, str) or isinstance(text[0], str):
            return self.sync_client(text).tolist()
        else:
            return self.sync_client(text[0].get_text())
