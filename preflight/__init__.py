"""Preflight: cheap client-side checks before submitting an XManager TPU job.

Pipeline:
  L1 (in-process, us):    topology whitelist + PROD min-slice rules
  L2 (~1 RPC, ~1s):       chip-level capacity via GetCellAvailability
  L2.5 (heuristic):       PROD quota headroom warning

See `preflight.py` for the orchestrator entry point `run_preflight()`.
"""
