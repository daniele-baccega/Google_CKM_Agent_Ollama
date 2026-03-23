"""Root agent entrypoint for ADK CLI/Web.

This exposes root_agent at the repository root so `adk run .` works.
"""

from src import root_agent

__all__ = ["root_agent"]
