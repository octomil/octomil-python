"""Shared execution kernel for Octomil.

All surfaces — CLI commands, SDK responses, serve, launch — converge on
this kernel for model resolution, policy evaluation, and inference dispatch.
"""

from octomil.execution.kernel import ExecutionKernel, ExecutionResult

__all__ = ["ExecutionKernel", "ExecutionResult"]
