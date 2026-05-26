import sys
import random
from contextvars import ContextVar
from typing import Union, Literal, List

# ContextVar-backed accumulators: each rollout (asyncio task / thread)
# can install its own counter via push_local_counters() / pop_local_counters().
# When no local is active, falls back to process-wide singleton (legacy callers).
# This pattern (per-rollout cost manager) enables intra-query and cross-query
# parallelism without singleton races. See feedback_cost_tracking memory.
_local_cost  = ContextVar('_local_cost',  default=None)  # mutable {'value': float}
_local_pt    = ContextVar('_local_pt',    default=None)
_local_ct    = ContextVar('_local_ct',    default=None)


class _Counter:
    __slots__ = ('value',)
    def __init__(self):
        self.value = 0.0


class Singleton:
    _instance = None

    @classmethod
    def instance(cls):
        # Prefer ContextVar local if installed, else fall back to process-wide.
        local_var = cls._local_var()
        if local_var is not None:
            local = local_var.get()
            if local is not None:
                return local
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def _local_var(cls):
        return None  # subclasses override

    def reset(self):
        self.value = 0.0


class Cost(Singleton):
    def __init__(self):
        self.value = 0.0
    @classmethod
    def _local_var(cls):
        return _local_cost


class PromptTokens(Singleton):
    def __init__(self):
        self.value = 0.0
    @classmethod
    def _local_var(cls):
        return _local_pt


class CompletionTokens(Singleton):
    def __init__(self):
        self.value = 0.0
    @classmethod
    def _local_var(cls):
        return _local_ct


def push_local_counters():
    """Install fresh per-rollout cost/token counters into the current context.
    Returns a token tuple to pass to pop_local_counters() for restore.
    Use with try/finally:
        token = push_local_counters()
        try:
            # all cost_count() calls in this scope go to the new local
            ...
            cost = Cost.instance().value
        finally:
            pop_local_counters(token)
    """
    cost_tok = _local_cost.set(_Counter())
    pt_tok   = _local_pt.set(_Counter())
    ct_tok   = _local_ct.set(_Counter())
    return (cost_tok, pt_tok, ct_tok)


def pop_local_counters(tokens):
    cost_tok, pt_tok, ct_tok = tokens
    _local_cost.reset(cost_tok)
    _local_pt.reset(pt_tok)
    _local_ct.reset(ct_tok)

class Time(Singleton):
    def __init__(self):
        self.value = ""

class Mode(Singleton):
    def __init__(self):
        self.value = ""
