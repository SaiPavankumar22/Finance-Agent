"""Canonical training namespace; legacy ``training`` imports remain supported."""

from training.data import first_json_object_complete, load_jsonl

__all__ = ["first_json_object_complete", "load_jsonl"]
