"""
Hierarchical Temporal Attention - Multi-Conformation Protein Encoder

Inspired by:
- AlphaPPImd (bioRxiv 2024): Self-attention for protein-protein conformational ensembles
- P2DFlow (JCTC 2025): Ensemble generative model with flow matching
- idpGAN (Nature Comms 2023): Conformational ensemble generation with transformers
- Temporal Attention Unit (CVPR 2023): Intra-frame + inter-frame attention

Architecture:
Input: Multiple conformations of a protein [num_confs, embed_dim]
Temporal Attention: Self-attention across conformations
Output: Aggregated conformational representation

Key design:
- Self-attention to weight important conformations (binding-relevant states)
- Multi-head attention for capturing diverse conformational features
- Positional encoding for conformation ordering (if applicable)
- Residual connections for gradient flow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HierarchicalTemporalAttention(nn.Module):
    """
    Temporal attention across protein conformations
    
    Learns which conformations are most relevant for binding prediction
    (e.g., open/closed states, active/inactive forms)
    
    Architecture:
    1. Conformation embeddings: [num_confs, embed_dim]
    2. Multi-head self-attention: Learn importance weights
    3. Weighted aggregation: Combine conformations
    4. Output: [embed_dim] aggregated representation
    """
    
    def __init__(self,
                 embed_dim=256,
                 num_heads=8,
                 dropout=0.2,
                 max_conformations=50,
                 use_positional_encoding=True):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_positional_encoding = use_positional_encoding
        
        # Multi-head self-attention
        # Q, K, V projections for all heads (but different per head)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Input: [batch, seq, features]
        )
        
        # Positional encoding (optional)
        # Useful if conformation order matters (e.g., MD trajectory)
        if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(
                embed_dim=embed_dim,
                max_len=max_conformations,
                dropout=dropout
            )
        
        # Feed-forward network (like Transformer encoder)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization (pre-LN like modern transformers)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        
        # Aggregation method
        self.aggregation_method = 'attention_weighted'  # or 'mean', 'max'
        
        # Attention pooling (learnable query for aggregation)
        self.attention_pool_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, conformation_embeddings, mask=None):
        """
        Forward pass: Attend over conformations
        
        Args:
            conformation_embeddings: [batch_size, num_confs, embed_dim]
                                     OR [num_confs, embed_dim] (will expand)
            mask: Optional [batch_size, num_confs] boolean mask (True = valid)
        
        Returns:
            aggregated: [batch_size, embed_dim] or [embed_dim]
            attention_weights: [batch_size, num_heads, num_confs, num_confs]
        """
        # Handle single protein (no batch dimension)
        squeeze_output = False
        if conformation_embeddings.dim() == 2:
            conformation_embeddings = conformation_embeddings.unsqueeze(0)  # [1, num_confs, embed_dim]
            squeeze_output = True
            if mask is not None:
                mask = mask.unsqueeze(0)
        
        batch_size, num_confs, embed_dim = conformation_embeddings.shape
        
        # ============================================================
        # Step 1: Add positional encoding (if enabled)
        # ============================================================
        x = conformation_embeddings
        if self.use_positional_encoding:
            x = self.positional_encoding(x)
        
        # ============================================================
        # Step 2: Self-attention across conformations
        # ============================================================
        # Pre-LN: Normalize before attention
        x_norm = self.ln1(x)
        
        # Self-attention: each conformation attends to all others
        # Q=K=V=x (self-attention)
        attn_output, attn_weights = self.attention(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            key_padding_mask=~mask if mask is not None else None,  # Invert mask (True = ignore)
            need_weights=True,
            average_attn_weights=False  # Return per-head weights
        )
        
        # Residual connection
        x = x + self.dropout(attn_output)
        
        # ============================================================
        # Step 3: Feed-forward network
        # ============================================================
        # Pre-LN: Normalize before FFN
        x_norm = self.ln2(x)
        
        # FFN
        ffn_output = self.ffn(x_norm)
        
        # Residual connection
        x = x + ffn_output  # [batch_size, num_confs, embed_dim]
        
        # ============================================================
        # Step 4: Aggregate conformations
        # ============================================================
        if self.aggregation_method == 'attention_weighted':
            # Learnable attention pooling: use a learned query to pool
            # Similar to AlphaPPImd's approach
            query = self.attention_pool_query.expand(batch_size, -1, -1)  # [batch_size, 1, embed_dim]
            
            # Attention pooling: query attends to all conformations
            pooled, pool_weights = self.attention(
                query=query,
                key=x,
                value=x,
                key_padding_mask=~mask if mask is not None else None,
                need_weights=True,
                average_attn_weights=True  # Average across heads
            )
            
            aggregated = pooled.squeeze(1)  # [batch_size, embed_dim]
            
        elif self.aggregation_method == 'mean':
            # Simple mean pooling
            if mask is not None:
                # Masked mean
                mask_expanded = mask.unsqueeze(-1).float()  # [batch_size, num_confs, 1]
                aggregated = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                aggregated = x.mean(dim=1)  # [batch_size, embed_dim]
        
        elif self.aggregation_method == 'max':
            # Max pooling
            aggregated = x.max(dim=1)[0]  # [batch_size, embed_dim]
        
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
        
        # Squeeze if input was single protein
        if squeeze_output:
            aggregated = aggregated.squeeze(0)  # [embed_dim]
        
        return aggregated, attn_weights


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for conformations
    
    Useful if conformation order matters (e.g., MD trajectory frames)
    """
    
    def __init__(self, embed_dim, max_len=50, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, embed_dim]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class MultiScaleTemporalAttention(nn.Module):
    """
    Multi-scale temporal attention for hierarchical conformations
    
    Operates at multiple levels:
    1. Intra-conformation: Features within each conformation
    2. Inter-conformation: Relationships between conformations
    
    Inspired by TAU (Temporal Attention Unit, CVPR 2023)
    """
    
    def __init__(self,
                 embed_dim=256,
                 num_heads=8,
                 dropout=0.2):
        super().__init__()
        
        # Intra-conformation attention (static features)
        self.intra_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Inter-conformation attention (dynamical features)
        # Squeeze-and-excitation style channel attention
        self.inter_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Global pooling: [batch, embed_dim, num_confs] → [batch, embed_dim, 1]
            nn.Conv1d(embed_dim, embed_dim // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(embed_dim // 4, embed_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.ln = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, num_confs, embed_dim]
        
        Returns:
            output: [batch_size, num_confs, embed_dim]
        """
        # Intra-conformation: Self-attention within conformations
        x_norm = self.ln(x)
        intra_out, _ = self.intra_attention(x_norm, x_norm, x_norm)
        x = x + intra_out
        
        # Inter-conformation: Channel attention across conformations
        # Reshape for 1D conv: [batch, num_confs, embed_dim] → [batch, embed_dim, num_confs]
        x_permuted = x.permute(0, 2, 1)
        inter_weights = self.inter_attention(x_permuted)  # [batch, embed_dim, 1]
        
        # Apply attention weights
        x_permuted = x_permuted * inter_weights  # Broadcasting
        x = x_permuted.permute(0, 2, 1)  # Back to [batch, num_confs, embed_dim]
        
        return x


def test_hierarchical_temporal_attention():
    """Test temporal attention module"""
    print("Testing HierarchicalTemporalAttention...")
    
    # Simulate protein conformations
    batch_size = 2
    num_conformations = 10
    embed_dim = 256
    
    # Create dummy conformation embeddings
    conformations = torch.randn(batch_size, num_conformations, embed_dim)
    
    # Optional mask (some conformations invalid)
    mask = torch.ones(batch_size, num_conformations, dtype=torch.bool)
    mask[0, 8:] = False  # First protein has only 8 valid conformations
    mask[1, 9:] = False  # Second protein has only 9 valid conformations
    
    # Initialize model
    model = HierarchicalTemporalAttention(
        embed_dim=embed_dim,
        num_heads=8,
        dropout=0.1,
        use_positional_encoding=True
    )
    
    # Forward pass
    aggregated, attn_weights = model(conformations, mask=mask)
    
    print(f"✓ Input shape: {conformations.shape}")
    print(f"✓ Output shape: {aggregated.shape}")
    print(f"✓ Attention weights shape: {attn_weights.shape}")
    print(f"✓ Expected output: [{batch_size}, {embed_dim}]")
    
    assert aggregated.shape == (batch_size, embed_dim), f"Output shape mismatch: {aggregated.shape}"
    print("✓ HierarchicalTemporalAttention test passed!")
    
    # Test single protein (no batch)
    print("\nTesting single protein (no batch dimension)...")
    single_conformations = torch.randn(num_conformations, embed_dim)
    single_aggregated, _ = model(single_conformations)
    
    print(f"✓ Single protein input: {single_conformations.shape}")
    print(f"✓ Single protein output: {single_aggregated.shape}")
    assert single_aggregated.shape == (embed_dim,), f"Single output shape mismatch: {single_aggregated.shape}"
    print("✓ Single protein test passed!")


def test_multiscale_temporal_attention():
    """Test multi-scale temporal attention"""
    print("\n" + "="*60)
    print("Testing MultiScaleTemporalAttention...")
    
    batch_size = 2
    num_conformations = 10
    embed_dim = 256
    
    conformations = torch.randn(batch_size, num_conformations, embed_dim)
    
    model = MultiScaleTemporalAttention(
        embed_dim=embed_dim,
        num_heads=8,
        dropout=0.1
    )
    
    output = model(conformations)
    
    print(f"✓ Input shape: {conformations.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Multi-scale attention preserves shape: {output.shape == conformations.shape}")
    
    assert output.shape == conformations.shape, f"Output shape mismatch: {output.shape}"
    print("✓ MultiScaleTemporalAttention test passed!")


if __name__ == '__main__':
    test_hierarchical_temporal_attention()
    test_multiscale_temporal_attention()
