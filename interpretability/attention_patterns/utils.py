import numpy as np
from typing import List, Dict, Tuple
import torch

def normalize_attention(attention: np.ndarray) -> np.ndarray:
    """
    Normalize attention weights to sum to 1.
    
    Args:
        attention: Attention weight matrix
        
    Returns:
        Normalized attention weights
    """
    return attention / (attention.sum(axis=-1, keepdims=True) + 1e-10)

def compute_attention_flow(attention: np.ndarray, 
                         source_idx: int, 
                         max_steps: int = 5) -> List[Tuple[int, float]]:
    """
    Compute attention flow from a source token to other tokens.
    
    Args:
        attention: Attention weight matrix
        source_idx: Index of the source token
        max_steps: Maximum number of steps to follow
        
    Returns:
        List of (token_idx, attention_weight) tuples
    """
    flow = []
    current_idx = source_idx
    visited = set()
    
    for _ in range(max_steps):
        if current_idx in visited:
            break
            
        visited.add(current_idx)
        weights = attention[current_idx]
        next_idx = np.argmax(weights)
        
        if next_idx in visited:
            break
            
        flow.append((next_idx, weights[next_idx]))
        current_idx = next_idx
        
    return flow

def aggregate_attention_patterns(attention_data: Dict, 
                               method: str = 'mean') -> np.ndarray:
    """
    Aggregate attention patterns across layers and heads.
    
    Args:
        attention_data: Output from AttentionAnalyzer.analyze()
        method: Aggregation method ('mean', 'max', or 'min')
        
    Returns:
        Aggregated attention matrix
    """
    patterns = attention_data['patterns']
    num_layers = len(patterns)
    num_heads = patterns[0]['attention'].shape[0]
    seq_length = patterns[0]['attention'].shape[1]
    
    aggregated = np.zeros((seq_length, seq_length))
    
    for layer_data in patterns:
        attention = layer_data['attention']
        
        if method == 'mean':
            aggregated += np.mean(attention, axis=0)
        elif method == 'max':
            aggregated = np.maximum(aggregated, np.max(attention, axis=0))
        elif method == 'min':
            if np.all(aggregated == 0):
                aggregated = np.min(attention, axis=0)
            else:
                aggregated = np.minimum(aggregated, np.min(attention, axis=0))
                
    if method == 'mean':
        aggregated /= num_layers
        
    return aggregated

def compute_attention_entropy(attention: np.ndarray) -> float:
    """
    Compute the entropy of attention weights.
    
    Args:
        attention: Attention weight matrix
        
    Returns:
        Attention entropy
    """
    # Add small epsilon to avoid log(0)
    attention = attention + 1e-10
    attention = attention / attention.sum(axis=-1, keepdims=True)
    return -np.sum(attention * np.log2(attention)) 