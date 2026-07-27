"""TODDC — Task-Oriented Dialogue Discourse Coherence.

Controlled coherence-violation injection over SGD via a 5-pass chain
(analyse -> document -> apply -> confirm -> edit), producing labeled
(coherent, perturbed) samples. Sibling of TODUQ.
"""

__version__ = "0.1.0"

DIMENSIONS = ("local", "cohesion", "global", "relevance", "state_consistency")
FAMILIES = ("control", "perturbation", "injection")
COHERENCE_LABELS = ("coherent", "incoherent")
