"""
Global configuration parameters for the beam vibration simulation.

This file centralizes user-defined settings that control numerical
integration accuracy and diagnostic output, allowing them to be
modified without changing the core implementation.
"""

DEBUG = True
"""If True, enables diagnostic output and consistency checks."""

NGP = 3
"""Number of Gauss integration points used for element integration."""
