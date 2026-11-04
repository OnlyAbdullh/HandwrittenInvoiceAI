"""Importing this package registers every geometric step. Order of these
imports doesn't matter -- pipeline order comes from config, not import
order.
"""

from preprocessing.steps.geometric import deskew, perspective_correction, working_resolution  # noqa: F401
