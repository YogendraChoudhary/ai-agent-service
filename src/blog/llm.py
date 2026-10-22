"""
The model call every blog stage goes through. It lives in src/longform/llm.py now, shared with the refinement pipeline;
this module keeps the import path the blog code and its tests use.
"""

from src.longform.llm import AgnoLLM, CallSpec, LLM, model_id_for        # noqa: F401  (re-exported)

__all__ = ["AgnoLLM", "CallSpec", "LLM", "model_id_for"]
