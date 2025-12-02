"""
Validity constraints for protein sequences.

This module implements hard constraints for R_correct(x) to ensure
generated sequences meet basic biological and structural requirements.
"""
import torch
from typing import List
from collections import Counter

CANONICAL_AAS = set('ACDEFGHIKLMNPQRSTVWY')
MIN_LENGTH = 8
MAX_LENGTH = 50
MIN_CHARGE = 0  # Minimum net charge for AMPs


def calculate_net_charge(sequence: str) -> float:
    """
    Calculate net charge of a sequence.
    
    Positive: K, R, H (+1 each)
    Negative: D, E (-1 each)
    
    Args:
        sequence: Protein sequence
    
    Returns:
        Net charge
    """
    positive = sequence.count('K') + sequence.count('R') + sequence.count('H')
    negative = sequence.count('D') + sequence.count('E')
    return positive - negative


def is_sequence_valid(sequence: str, min_charge: float = MIN_CHARGE) -> bool:
    """
    Check if sequence meets basic validity constraints.
    
    Constraints:
    1. Only canonical amino acids (20 standard AAs)
    2. Length within [MIN_LENGTH, MAX_LENGTH]
    3. Minimum net charge (for AMPs)
    4. No excessive repeats (no more than 50% of any single AA)
    
    Args:
        sequence: Protein sequence
        min_charge: Minimum net charge requirement
    
    Returns:
        True if valid, False otherwise
    """
    if not sequence:
        return False
    
    # 1. Only canonical amino acids
    if not all(aa in CANONICAL_AAS for aa in sequence):
        return False
    
    # 2. Length constraints
    if not (MIN_LENGTH <= len(sequence) <= MAX_LENGTH):
        return False
    
    # 3. Minimum charge (for AMPs)
    if calculate_net_charge(sequence) < min_charge:
        return False
    
    # 4. No excessive repeats (no more than 50% of any single AA)
    max_repeat = max(Counter(sequence).values())
    if max_repeat / len(sequence) > 0.5:
        return False
    
    return True


def validate_sequences(sequences: List[str], min_charge: float = MIN_CHARGE) -> torch.Tensor:
    """
    Return boolean mask of valid sequences.
    
    Args:
        sequences: List of protein sequences
        min_charge: Minimum net charge requirement
    
    Returns:
        Boolean tensor of shape (len(sequences),)
    """
    return torch.tensor(
        [is_sequence_valid(seq, min_charge) for seq in sequences], 
        dtype=torch.bool
    )


def get_validity_stats(sequences: List[str], min_charge: float = MIN_CHARGE) -> dict:
    """
    Get detailed statistics about sequence validity.
    
    Args:
        sequences: List of protein sequences
        min_charge: Minimum net charge requirement
    
    Returns:
        Dictionary with validity statistics
    """
    stats = {
        'total': len(sequences),
        'valid': 0,
        'invalid_aa': 0,
        'invalid_length': 0,
        'invalid_charge': 0,
        'invalid_repeats': 0,
    }
    
    for seq in sequences:
        if not seq:
            continue
        
        # Check each constraint
        has_valid_aa = all(aa in CANONICAL_AAS for aa in seq)
        has_valid_length = MIN_LENGTH <= len(seq) <= MAX_LENGTH
        has_valid_charge = calculate_net_charge(seq) >= min_charge
        has_valid_repeats = max(Counter(seq).values()) / len(seq) <= 0.5
        
        if has_valid_aa and has_valid_length and has_valid_charge and has_valid_repeats:
            stats['valid'] += 1
        else:
            if not has_valid_aa:
                stats['invalid_aa'] += 1
            if not has_valid_length:
                stats['invalid_length'] += 1
            if not has_valid_charge:
                stats['invalid_charge'] += 1
            if not has_valid_repeats:
                stats['invalid_repeats'] += 1
    
    stats['validity_rate'] = stats['valid'] / stats['total'] if stats['total'] > 0 else 0
    
    return stats

