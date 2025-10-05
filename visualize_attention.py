import numpy as np


def visualize_attention_ascii(attention_weights, layer_idx=0, head_idx=0, batch_idx=0,
                               max_tokens=10, token_labels=None):
    """
    Visualisasi attention pattern dalam format ASCII/text.

    Args:
        attention_weights: list of [batch, num_heads, seq_len, seq_len] untuk setiap layer
        layer_idx: index layer yang akan divisualisasikan
        head_idx: index head yang akan divisualisasikan
        batch_idx: index batch
        max_tokens: maksimum token yang ditampilkan (untuk readability)
        token_labels: optional list of token labels
    """
    # Extract attention untuk layer, head, dan batch tertentu
    attn = attention_weights[layer_idx][batch_idx, head_idx, :, :]  # [seq_len, seq_len]
    seq_len = attn.shape[0]

    # Limit untuk readability
    display_len = min(seq_len, max_tokens)
    attn_display = attn[:display_len, :display_len]

    print(f"\n{'='*70}")
    print(f" Attention Pattern - Layer {layer_idx}, Head {head_idx}, Batch {batch_idx}")
    print(f"{'='*70}")
    print(f"Sequence Length: {seq_len} (showing first {display_len} tokens)")
    print(f"\nAttention Matrix (rows=query positions, cols=key positions):")
    print(f"Brighter values = stronger attention\n")

    # Header
    if token_labels is not None:
        print("      ", end="")
        for j in range(display_len):
            label = token_labels[j] if j < len(token_labels) else f"t{j}"
            print(f"{label:^7}", end="")
        print()
    else:
        print("      ", end="")
        for j in range(display_len):
            print(f"  t{j:<4}", end="")
        print()

    # Attention matrix dengan visual encoding
    for i in range(display_len):
        # Row label
        if token_labels is not None:
            label = token_labels[i] if i < len(token_labels) else f"t{i}"
            print(f"{label:>5} ", end="")
        else:
            print(f" t{i:<3} ", end="")

        # Attention values
        for j in range(display_len):
            val = attn_display[i, j]
            # Visual encoding: berbeda simbol untuk berbeda intensitas
            if val < 0.01:
                symbol = "  ·   "
            elif val < 0.05:
                symbol = "  ░   "
            elif val < 0.15:
                symbol = "  ▒   "
            elif val < 0.30:
                symbol = "  ▓   "
            else:
                symbol = "  █   "
            print(symbol, end="")
        print()

    print("\nLegend:")
    print("  · = 0.00-0.01  (very weak)")
    print("  ░ = 0.01-0.05  (weak)")
    print("  ▒ = 0.05-0.15  (moderate)")
    print("  ▓ = 0.15-0.30  (strong)")
    print("  █ = 0.30+      (very strong)")


def visualize_attention_heatmap_ascii(attention_weights, layer_idx=0, head_idx=0,
                                      batch_idx=0, max_tokens=15):
    """
    Visualisasi attention dengan numerik values.

    Args:
        attention_weights: list of [batch, num_heads, seq_len, seq_len]
        layer_idx: layer index
        head_idx: head index
        batch_idx: batch index
        max_tokens: max tokens to display
    """
    attn = attention_weights[layer_idx][batch_idx, head_idx, :, :]
    seq_len = attn.shape[0]
    display_len = min(seq_len, max_tokens)
    attn_display = attn[:display_len, :display_len]

    print(f"\n{'='*70}")
    print(f" Attention Heatmap - Layer {layer_idx}, Head {head_idx}")
    print(f"{'='*70}\n")

    # Header
    print("      ", end="")
    for j in range(display_len):
        print(f" t{j:<4}", end="")
    print()

    # Values
    for i in range(display_len):
        print(f" t{i:<3} ", end="")
        for j in range(display_len):
            val = attn_display[i, j]
            print(f"{val:5.3f} ", end="")
        print()


def attention_statistics(attention_weights, layer_idx=0):
    """
    Hitung statistik dari attention patterns.

    Args:
        attention_weights: list of [batch, num_heads, seq_len, seq_len]
        layer_idx: layer index

    Returns:
        dict dengan statistik
    """
    attn = attention_weights[layer_idx]  # [batch, num_heads, seq_len, seq_len]
    batch_size, num_heads, seq_len, _ = attn.shape

    stats = {
        'layer': layer_idx,
        'batch_size': batch_size,
        'num_heads': num_heads,
        'seq_len': seq_len,
        'mean': attn.mean(),
        'std': attn.std(),
        'min': attn.min(),
        'max': attn.max(),
    }

    # Statistik per head
    stats['per_head'] = {}
    for h in range(num_heads):
        head_attn = attn[:, h, :, :]
        stats['per_head'][h] = {
            'mean': head_attn.mean(),
            'std': head_attn.std(),
            'max': head_attn.max(),
        }

    # Sparsity: berapa persen attention < threshold
    threshold = 0.01
    sparsity = (attn < threshold).mean() * 100
    stats['sparsity'] = sparsity

    # Concentration: apakah attention terpusat atau terdistribusi?
    # Hitung entropy
    epsilon = 1e-10
    entropy = -(attn * np.log(attn + epsilon)).sum(axis=-1).mean()
    stats['entropy'] = entropy

    return stats


def compare_attention_patterns(attention_weights, layer_idx=0, batch_idx=0):
    """
    Bandingkan attention patterns across heads.

    Args:
        attention_weights: list of [batch, num_heads, seq_len, seq_len]
        layer_idx: layer index
        batch_idx: batch index
    """
    attn = attention_weights[layer_idx][batch_idx]  # [num_heads, seq_len, seq_len]
    num_heads = attn.shape[0]

    print(f"\n{'='*70}")
    print(f" Attention Pattern Comparison Across Heads - Layer {layer_idx}")
    print(f"{'='*70}\n")

    print(f"Number of heads: {num_heads}\n")

    print("Head Statistics:")
    print(f"{'Head':<6} {'Mean':<8} {'Std':<8} {'Max':<8} {'Sparsity':<10}")
    print("-" * 50)

    for h in range(num_heads):
        head_attn = attn[h]
        mean_val = head_attn.mean()
        std_val = head_attn.std()
        max_val = head_attn.max()
        sparsity = (head_attn < 0.01).mean() * 100

        print(f"{h:<6} {mean_val:<8.4f} {std_val:<8.4f} {max_val:<8.4f} {sparsity:<9.1f}%")


def analyze_attention_by_position(attention_weights, layer_idx=0, batch_idx=0):
    """
    Analisis bagaimana setiap posisi attend ke posisi lain.

    Args:
        attention_weights: list of [batch, num_heads, seq_len, seq_len]
        layer_idx: layer index
        batch_idx: batch index
    """
    attn = attention_weights[layer_idx][batch_idx]  # [num_heads, seq_len, seq_len]

    # Average across heads
    avg_attn = attn.mean(axis=0)  # [seq_len, seq_len]
    seq_len = avg_attn.shape[0]

    print(f"\n{'='*70}")
    print(f" Position-wise Attention Analysis - Layer {layer_idx}")
    print(f"{'='*70}\n")

    print("Where does each position attend to? (averaged across heads)\n")

    for i in range(min(seq_len, 10)):  # Show first 10 positions
        print(f"Position t{i}:")

        # Top-5 attended positions
        top_k = min(5, seq_len)
        top_indices = np.argsort(avg_attn[i])[-top_k:][::-1]

        print("  Top attended positions:")
        for rank, idx in enumerate(top_indices, 1):
            weight = avg_attn[i, idx]
            print(f"    {rank}. t{idx}: {weight:.4f} ({weight*100:.1f}%)")

        # Statistics
        self_attention = avg_attn[i, i]
        past_attention = avg_attn[i, :i].sum() if i > 0 else 0

        print(f"  Self-attention: {self_attention:.4f}")
        if i > 0:
            print(f"  Total attention to past: {past_attention:.4f}")
        print()


def visualize_layer_comparison(attention_weights, batch_idx=0, head_idx=0):
    """
    Bandingkan attention patterns across layers.

    Args:
        attention_weights: list of [batch, num_heads, seq_len, seq_len]
        batch_idx: batch index
        head_idx: head index
    """
    num_layers = len(attention_weights)

    print(f"\n{'='*70}")
    print(f" Attention Comparison Across Layers (Head {head_idx})")
    print(f"{'='*70}\n")

    print(f"{'Layer':<8} {'Mean':<8} {'Std':<8} {'Max':<8} {'Entropy':<10}")
    print("-" * 50)

    for layer_idx in range(num_layers):
        attn = attention_weights[layer_idx][batch_idx, head_idx]

        mean_val = attn.mean()
        std_val = attn.std()
        max_val = attn.max()

        # Entropy
        epsilon = 1e-10
        entropy = -(attn * np.log(attn + epsilon)).sum(axis=-1).mean()

        print(f"{layer_idx:<8} {mean_val:<8.4f} {std_val:<8.4f} {max_val:<8.4f} {entropy:<10.4f}")


def print_attention_summary(attention_weights):
    """
    Print summary statistics untuk semua attention weights.

    Args:
        attention_weights: list of [batch, num_heads, seq_len, seq_len]
    """
    num_layers = len(attention_weights)

    print(f"\n{'='*70}")
    print(f" Attention Weights Summary")
    print(f"{'='*70}\n")

    print(f"Number of layers: {num_layers}")

    for layer_idx in range(num_layers):
        attn = attention_weights[layer_idx]
        batch_size, num_heads, seq_len, _ = attn.shape

        print(f"\nLayer {layer_idx}:")
        print(f"  Shape: {attn.shape} (batch, heads, seq_len, seq_len)")
        print(f"  Mean: {attn.mean():.6f}")
        print(f"  Std: {attn.std():.6f}")
        print(f"  Min: {attn.min():.6f}")
        print(f"  Max: {attn.max():.6f}")

        # Check causal mask property
        # Future positions should have near-zero attention
        total_future_attn = 0
        count_future = 0
        for i in range(seq_len):
            for j in range(i+1, seq_len):
                total_future_attn += attn[:, :, i, j].sum()
                count_future += batch_size * num_heads

        avg_future_attn = total_future_attn / count_future if count_future > 0 else 0
        print(f"  Avg attention to future positions: {avg_future_attn:.8f} (should be ~0)")
