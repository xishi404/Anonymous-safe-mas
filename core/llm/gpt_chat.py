import aiohttp
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Dict, Any
from dotenv import load_dotenv
import os
import requests
from groq import Groq, AsyncGroq
from openai import OpenAI, AsyncOpenAI, BadRequestError

from core.llm.price import cost_count
from core.llm.llm import LLM
from core.llm.llm_registry import LLMRegistry

load_dotenv()
MINE_BASE_URL = os.getenv('BASE_URL')
MINE_API_KEYS = os.getenv('API_KEY')

# Per-model URL dispatch for local inference backend pool. Falls back to env URL/KEY otherwise.
from core.llm.llm_profile import MODEL_BASE_URLS


def _resolve_endpoint(model_name: str):
    if model_name in MODEL_BASE_URLS:
        return MODEL_BASE_URLS[model_name], os.environ.get("KEY", "EMPTY")
    return os.environ.get("URL"), os.environ.get("KEY")


def _extra_body_for(model_name: str):
    """Qwen3 series defaults to hybrid-thinking; disable so this work's
    output parsers (looking for ```python blocks, boxed answers, etc.) see
    clean content instead of <think>...</think> reasoning traces."""
    if model_name.startswith("Qwen3-"):
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return None


@LLMRegistry.register('ALLChat')
class ALLChat(LLM):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @retry(wait=wait_random_exponential(max=30), stop=stop_after_attempt(3))
    def gen(
        self,
        messages: Union[List[Dict], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [{'role':"user", 'content':messages}]
        base_url, api_key = _resolve_endpoint(self.model_name)
        client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
        # Dynamically cap max_tokens based on rough prompt length (char/4 ≈ tokens).
        # max_model_len=4096 on the pool; leave at least 64 tokens of headroom.
        approx_in = sum(len(m.get('content', '')) for m in messages) // 4
        out_cap = max(64, min(max_tokens, 1024, 4000 - approx_in))
        kwargs = dict(
            messages=messages,
            model=self.model_name,
            max_tokens=out_cap,
            temperature=temperature,
        )
        eb = _extra_body_for(self.model_name)
        if eb:
            kwargs["extra_body"] = eb
        try:
            chat_completion = client.chat.completions.create(**kwargs)
            response = chat_completion.choices[0].message.content
        except BadRequestError as e:
            # Prompt too long or other 400 — return empty so training signals failure
            # rather than retrying 10 times on a non-retryable error.
            msg = str(e)[:200]
            response = ""
        prompt = "".join([item['content'] for item in messages])
        cost_count(prompt, response, self.model_name)
        return response

    async def agen(
        self,
        messages: Union[List[Dict], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [{'role':"user", 'content':messages}]
        base_url, api_key = _resolve_endpoint(self.model_name)
        client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0)
        approx_in = sum(len(m.get('content', '')) for m in messages) // 4
        out_cap = max(64, min(max_tokens, 1024, 4000 - approx_in))
        kwargs = dict(
            messages=messages,
            model=self.model_name,
            max_tokens=out_cap,
            temperature=temperature,
        )
        eb = _extra_body_for(self.model_name)
        if eb:
            kwargs["extra_body"] = eb
        try:
            chat_completion = await client.chat.completions.create(**kwargs)
            response = chat_completion.choices[0].message.content
        except BadRequestError:
            response = ""
        return response
    

@LLMRegistry.register('Deepseek')
class DSChat(LLM):
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    @retry(wait=wait_random_exponential(max=30), stop=stop_after_attempt(3))
    def gen(
        self,
        messages: Union[List[Dict], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [{'role':"user", 'content':messages}]
        client = OpenAI(base_url = os.environ.get("DS_URL"),
                        api_key = os.environ.get("DS_KEY"))
        chat_completion = client.chat.completions.create(
        messages = messages,
        model = self.model_name,
        )
        response = chat_completion.choices[0].message.content
        prompt = "".join([item['content'] for item in messages])
        cost_count(prompt, response, self.model_name)
        return response

    async def agen(
        self,
        messages: Union[List[Dict], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [{'role':"user", 'content':messages}]
        
        client = AsyncOpenAI(base_url = os.environ.get("DS_URL"),
                             api_key = os.environ.get("DS_KEY"),)
        chat_completion = await client.chat.completions.create(
        messages = messages,
        model = self.model_name,
        max_tokens = max_tokens,
        temperature = temperature,
        )
        response = chat_completion.choices[0].message.content

        return response

@retry(wait=wait_random_exponential(max=30), stop=stop_after_attempt(3))
async def achat(
    model: str,
    msg: List[Dict],):
    request_url = MINE_BASE_URL
    authorization_key = MINE_API_KEYS
    headers = {
        'Content-Type': 'application/json',
        'authorization': authorization_key
    }
    data = {
        "name": model + '-y',
        "inputs": {
            "stream": False,
            "msg": repr(msg),
        }
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(request_url, headers=headers ,json=data) as response:
            response_data = await response.json()
            if isinstance(response_data['data'],str):
                prompt = "".join([item['content'] for item in msg])
                cost_count(prompt,response_data['data'], model)
                return response_data['data']
            else:
                raise Exception("api error")

@retry(wait=wait_random_exponential(max=30), stop=stop_after_attempt(3))
def chat(
    model: str,
    msg: List[Dict],):
    request_url = MINE_BASE_URL
    authorization_key = MINE_API_KEYS
    headers = {
        'Content-Type': 'application/json',
        'authorization': authorization_key
    }
    data = {
        "name": model+'-y',
        "inputs": {
            "stream": False,
            "msg": repr(msg),
        }
    }
    response = requests.post(request_url, headers=headers ,json=data)
    response_data = response.json()
    if isinstance(response_data['data'],str):
        prompt = "".join([item['content'] for item in msg])
        cost_count(prompt,response_data['data'], model)
        return response_data['data']
    else:
        raise Exception("api error")

@LLMRegistry.register('GPTChat')
class GPTChat(LLM):
    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: Union[List[Dict], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        if isinstance(messages, str):
            messages = [{'role':"user", 'content':messages}]
        return await achat(self.model_name,messages)
    
    def gen(
        self,
        messages: Union[List[Dict], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:
        
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [{'role':"user", 'content':messages}]
        return chat(self.model_name,messages)
    

@LLMRegistry.register('Groq')
class GroqChat(LLM):
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    @retry(wait=wait_random_exponential(max=30), stop=stop_after_attempt(3))
    def gen(
        self,
        messages: Union[List[Dict], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:
        # TODO: Add num_comps to the request
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [{'role':"user", 'content':messages}]
        
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"),)
        chat_completion = client.chat.completions.create(
        messages = messages,
        model = self.model_name,
        )
        response = chat_completion.choices[0].message.content
        prompt = "".join([item['content'] for item in messages])
        cost_count(prompt, response, self.model_name)        
        return response

    async def agen(
        self,
        messages: Union[List[Dict], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:
        # TODO: Add num_comps to the request
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [{'role':"user", 'content':messages}]
        
        client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"),)
        chat_completion = await client.chat.completions.create(
        messages = messages,
        model = self.model_name,
        max_tokens = max_tokens,
        temperature = temperature,
        )
        response = chat_completion.choices[0].message.content

        return response
    

@LLMRegistry.register('OpenRouter')
class OpenRouterChat(LLM):
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    @retry(wait=wait_random_exponential(max=30), stop=stop_after_attempt(3))
    def gen(
        self,
        messages: Union[List[Dict], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [{'role':"user", 'content':messages}]
        client = OpenAI(base_url = os.environ.get("OPENROUTER_BASE_URL"),
                        api_key = os.environ.get("OPENROUTER_API_KEY"),)
        chat_completion = client.chat.completions.create(
        messages = messages,
        model = self.model_name,
        )
        response = chat_completion.choices[0].message.content
        return response

    async def agen(
        self,
        messages: Union[List[Dict], str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:
        # TODO
        return 0