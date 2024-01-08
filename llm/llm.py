import os
import json
import together
import logging
from typing import Any, Dict, List, Mapping, Optional
from pydantic import Extra, Field
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.utils import get_from_dict_or_env
from langchain.chat_models import ChatOpenAI
os.environ['OPENAI_API_KEY'] ="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" 
os.environ['TOGETHER_API_KEY'] = ''

class TogetherLLM(LLM):
    """Together large language models.""" 

    model: str = "togethercomputer/llama-2-70b-chat"
    """model endpoint to use"""

    together_api_key: str = os.environ["TOGETHER_API_KEY"]
    """Together API key"""

    temperature: float = 0.7
    """What sampling temperature to use."""

    max_tokens: int = 512
    """The maximum number of tokens to generate in the completion."""

    class Config:
        extra = Extra.forbid

    
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the API key is set."""
        api_key = get_from_dict_or_env(
            values, "together_api_key", "TOGETHER_API_KEY"
        )
        values["together_api_key"] = api_key
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "together"

    def _call(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Call to Together endpoint."""
        together.api_key = self.together_api_key
        output = together.Complete.create(prompt,
                                          model=self.model,
                                          max_tokens=self.max_tokens,
                                          temperature=self.temperature,
                                          )
        text = output['output']['choices'][0]['text']
        return text
    
def together_llm():
    llm = TogetherLLM(
        model= "togethercomputer/llama-2-70b-chat",
        temperature=0.1,
        max_tokens=512
        ) 
    return llm


def get_llm():
    if len(os.environ['OPENAI_API_KEY']) > 1:
        return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    elif len(os.environ['TOGETHER_API_KEY']) > 1:
        return together_llm()


def get_llm_type():
    if len(os.environ['OPENAI_API_KEY']) > 1:
        return 'ChatOpenAI'
    elif len(os.environ['TOGETHER_API_KEY']) > 1:
        return 'TogetherLLM'
