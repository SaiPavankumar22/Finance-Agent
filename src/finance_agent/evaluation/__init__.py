"""Canonical evaluation namespace; legacy ``eval`` imports remain supported."""

from eval.metrics import Prediction, score_prediction, summarize

__all__ = ["Prediction", "score_prediction", "summarize"]
