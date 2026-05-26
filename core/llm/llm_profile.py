"""LLM profile for the heterogeneous routing pool.

Each entry provides a textual model description that the SentenceEncoder uses
to obtain LLM-side embeddings for the router. Benchmark numbers in the
descriptions are approximate reference values.
"""

import os as _os
_PORT_OFFSET = int(_os.environ.get('MAS_POOL_PORT_OFFSET', '0'))

MODEL_PORTS = {
    "Qwen3-14B":               8001 + _PORT_OFFSET,
    "Qwen3-8B":                8002 + _PORT_OFFSET,
    "Llama-3.1-8B-Instruct":   8003 + _PORT_OFFSET,
    "gemma-3-12b-it":          8004 + _PORT_OFFSET,
    "Mistral-7B-Instruct-v0.1": 8005 + _PORT_OFFSET,
}

MODEL_BASE_URLS = {name: f"http://127.0.0.1:{port}/v1" for name, port in MODEL_PORTS.items()}


llm_profile = [
    {'Name': 'Qwen3-14B',
     'Description': 'Qwen3-14B is a mid-large general-purpose open-weight model from Alibaba with strong reasoning, coding, and multilingual abilities.\n\
        The model costs $0.06 per million input tokens and $0.24 per million output tokens.\n\
        In General Q&A Benchmark MMLU, Qwen3-14B achieves an accuracy of 81.2.\n\
        In Reasoning Benchmark GPQA, Qwen3-14B achieves an accuracy of 52.1.\n\
        In Coding Benchmark HumanEval, Qwen3-14B achieves an accuracy of 85.3.\n\
        In Math Benchmark MATH, Qwen3-14B achieves an accuracy of 78.6.'},

    {'Name': 'Qwen3-8B',
     'Description': 'Qwen3-8B is a mid-size general-purpose open-weight model from Alibaba, balancing quality and latency.\n\
        The model costs $0.05 per million input tokens and $0.40 per million output tokens.\n\
        In General Q&A Benchmark MMLU, Qwen3-8B achieves an accuracy of 75.4.\n\
        In Reasoning Benchmark GPQA, Qwen3-8B achieves an accuracy of 42.3.\n\
        In Coding Benchmark HumanEval, Qwen3-8B achieves an accuracy of 80.1.\n\
        In Math Benchmark MATH, Qwen3-8B achieves an accuracy of 70.2.'},

    {'Name': 'Llama-3.1-8B-Instruct',
     'Description': 'Llama-3.1-8B-Instruct is an 8B instruction-tuned open-weight model from Meta, with strong general-purpose chat and reasoning behavior at low cost.\n\
        The model costs $0.02 per million input tokens and $0.05 per million output tokens.\n\
        In General Q&A Benchmark MMLU, Llama-3.1-8B-Instruct achieves an accuracy of 73.0.\n\
        In Reasoning Benchmark GPQA, Llama-3.1-8B-Instruct achieves an accuracy of 30.4.\n\
        In Coding Benchmark HumanEval, Llama-3.1-8B-Instruct achieves an accuracy of 72.6.\n\
        In Math Benchmark MATH, Llama-3.1-8B-Instruct achieves an accuracy of 51.9.'},

    {'Name': 'gemma-3-12b-it',
     'Description': 'Gemma-3-12B-IT is an instruction-tuned open-weight model from Google.\n\
        The model costs $0.12 per million input tokens and $0.12 per million output tokens.\n\
        In General Q&A Benchmark MMLU, gemma-3-12b-it achieves an accuracy of 74.0.\n\
        In Reasoning Benchmark GPQA, gemma-3-12b-it achieves an accuracy of 40.6.\n\
        In Coding Benchmark HumanEval, gemma-3-12b-it achieves an accuracy of 78.8.\n\
        In Math Benchmark MATH, gemma-3-12b-it achieves an accuracy of 67.9.'},

    {'Name': 'Mistral-7B-Instruct-v0.1',
     'Description': 'Mistral-7B-Instruct-v0.1 is an early-generation 7B instruction-tuned open-weight model from Mistral AI.\n\
        The model costs $0.05 per million input tokens and $0.05 per million output tokens.\n\
        In General Q&A Benchmark MMLU, Mistral-7B-Instruct-v0.1 achieves an accuracy of 55.0.\n\
        In Reasoning Benchmark GPQA, Mistral-7B-Instruct-v0.1 achieves an accuracy of 28.0.\n\
        In Coding Benchmark HumanEval, Mistral-7B-Instruct-v0.1 achieves an accuracy of 30.0.\n\
        In Math Benchmark MATH, Mistral-7B-Instruct-v0.1 achieves an accuracy of 14.0.'},
]
