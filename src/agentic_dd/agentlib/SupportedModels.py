from typing import Dict, List, TypedDict

class TokenCounts(TypedDict):
    model_name: int

class FileTokenCounts(TypedDict):
    filename: Dict[str, int]

CONTEXT_WINDOWS: Dict[str, int] ={
    "deepseek-v4-flash": 1000000,
    "apertus-70b-instruct-2509": 65000,
    "devstral-2-123b-instruct-2512": 256000,
    "gemma-4-31b-it": 256000,
    "glm-4.7": 200000,
    "meta-llama-3.1-8b-instruct": 128000,
    "mistral-medium-3.5-128b": 256000,
    "openai-gpt-oss-120b": 128000,
    "qwen3-coder-next":256000,
    "qwen3-omni-30b-a3b-instruct":256000,
    "qwen3-30b-a3b-instruct-2507": 256000,
    "qwen3.5-122b-a10b":256000,
    "qwen3.5-397b-a17b":256000,
    "qwen3.6-35b-a3b":262000,
    "qwen3.8-27b":262000,
}

LOCAL_CONTEXT_WINDOWS: Dict[str, int] = {
    "qwen2.5:1.5b": 32768,           
    "qwen2.5:7b": 32768,    
}

GWDGID_to_HFID_map: Dict[str, str] = {
    'apertus-70b-instruct-2509': 'swiss-ai/Apertus-70B-Instruct-2509',
    'deepseek-v4-flash': 'deepseek-ai/DeepSeek-V4-Flash',
    'devstral-2-123b-instruct-2512': 'mistralai/Devstral-2-123B-Instruct-2512',
    'gemma-4-31b-it': 'google/gemma-4-31B-it',
    'glm-4.7': 'zai-org/GLM-4.7',
    #'meta-llama-3.1-8b-instruct': 'meta-llama/Llama-3.1-8B-Instruct', # the repo for the tokenizer is guarded
    'mistral-medium-3.5-128b': 'mistralai/Mistral-Medium-3.5-128B',
    'openai-gpt-oss-120b': 'openai/gpt-oss-120b',

    # no significant intra-family differences were found for tokenization, therefor all Qwen models use the same one
    'qwen3.8-27b': 'Qwen/Qwen3.6-27B',
    #'qwen3.6-35b-a3b': 'Qwen/Qwen3.6-35B-A3B',
    #'qwen3.5-397b-a17b': 'Qwen/Qwen3.5-397B-A17B',
    #'qwen3.5-122b-a10b': 'Qwen/Qwen3.5-122B-A10B',
    #'qwen3-30b-a3b-instruct-2507': 'Qwen/Qwen3-30B-A3B-Instruct-2507',
    #'qwen3-omni-30b-a3b-instruct': 'Qwen/Qwen3-Omni-30B-A3B-Instruct',
    #'qwen3-coder-next': 'Qwen/Qwen3-Coder-Next',
}

# Per-model overrides for AutoTokenizer.from_pretrained kwargs if necessary
SPECIAL_TOKENIZER_KWARGS: Dict[str, Dict] = {
    #'mistral-large-3-675b-instruct-2512': {'fix_mistral_regex': True},
}

class SupportedModels:
    @staticmethod
    def is_supported(name: str) -> bool:
        return name in GWDGID_to_HFID_map.keys()

    @staticmethod
    def get_all_supported_models() -> List[str]:
        return GWDGID_to_HFID_map.keys()

    @staticmethod
    def get_huggingface_tokenizer(model_name: str) -> str:
        if SupportedModels.is_supported(model_name):
            return GWDGID_to_HFID_map[model_name]
        else: raise NotImplementedError('Model is not supported')

    @staticmethod
    def get_special_tokenizer_kwargs():
        return SPECIAL_TOKENIZER_KWARGS

    

CAPABILITIES = [
    'Reasoning',
    'Tool Use'
]

class ModelTypes:
    @staticmethod
    def has_capabilities():
        pass