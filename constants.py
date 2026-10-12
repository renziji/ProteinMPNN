"""
Project-wide constants shared across all ProteinMPNN modules.

Centralising the alphabet here ensures a single source of truth: every module
that needs the amino-acid encoding imports from this file instead of
duplicating the string literal.
"""

# Model encoding alphabet: 20 standard amino acids + 'X' for unknown/masked residues.
# This order is baked into the model weights and must not be changed.
PROTEIN_ALPHABET: str = 'ACDEFGHIKLMNPQRSTVWYX'

# Alphabet with gap marker '-' appended, used for sequence validation in datasets.
PROTEIN_ALPHABET_WITH_GAP: str = PROTEIN_ALPHABET + '-'
