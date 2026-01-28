"""Flows do Prefect para orquestracao."""

from pipelines.flows.training_flow import training_pipeline

__all__ = ["training_pipeline"]
