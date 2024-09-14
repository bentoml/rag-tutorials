import bentoml
from bentovllm_openai.utils import openai_endpoints

LLM_MAX_TOKENS = 4096
LLM_MODEL_ID = "meta-llama/Meta-Llama-3.1-8b-instruct"

@openai_endpoints(model_id=LLM_MODEL_ID)
@bentoml.service(
    traffic={
        "timeout": 600,
    },
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-l4",
    },
)
class VLLM:
    def __init__(self) -> None:
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        ENGINE_ARGS = AsyncEngineArgs(
            model=LLM_MODEL_ID,
            max_model_len=LLM_MAX_TOKENS
        )

        self.engine = AsyncLLMEngine.from_engine_args(ENGINE_ARGS)
