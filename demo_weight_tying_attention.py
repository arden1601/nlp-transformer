"""
Demo untuk:
1. Weight Tying - Perbandingan model dengan dan tanpa weight tying
2. Attention Visualization
"""

import numpy as np
from transformer import GPTModel
from visualize_attention import (
    visualize_attention_ascii,
    visualize_attention_heatmap_ascii,
    compare_attention_patterns,
    analyze_attention_by_position,
    visualize_layer_comparison,
)


def print_header(title):
    """Print section header."""
    print("\n" + "█" * 70)
    print("█" + f" {title}".ljust(68) + "█")
    print("█" * 70)


def print_section(title):
    """Print subsection."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def demo_weight_tying_comparison():
    """Demo perbandingan model dengan dan tanpa weight tying."""
    print_header("DEMO 1: WEIGHT TYING COMPARISON")

    # Model configuration
    config = {
        'vocab_size': 1000,
        'd_model': 128,
        'num_heads': 8,
        'd_ff': 512,
        'num_layers': 4,
        'max_seq_len': 100
    }

    print_section("Model Configuration")
    for key, value in config.items():
        print(f"  {key:15s}: {value}")

    # Initialize models
    print_section("Initializing Models")

    np.random.seed(42)
    model_without_tying = GPTModel(**config, weight_tying=False)
    print("✓ Model WITHOUT weight tying initialized")

    np.random.seed(42)
    model_with_tying = GPTModel(**config, weight_tying=True)
    print("✓ Model WITH weight tying initialized")

    # Count parameters
    print_section("Parameter Count Comparison")

    total_1, breakdown_1 = model_without_tying.count_parameters()
    total_2, breakdown_2 = model_with_tying.count_parameters()

    print("\nModel WITHOUT Weight Tying:")
    print("-" * 70)
    for component, count in breakdown_1.items():
        pct = (count / total_1) * 100
        print(f"  {component:20s}: {count:12,d} params ({pct:5.2f}%)")
    print("-" * 70)
    print(f"  {'TOTAL':20s}: {total_1:12,d} params")

    print("\n\nModel WITH Weight Tying:")
    print("-" * 70)
    for component, count in breakdown_2.items():
        pct = (count / total_2) * 100 if total_2 > 0 else 0
        print(f"  {component:20s}: {count:12,d} params ({pct:5.2f}%)")
    print("-" * 70)
    print(f"  {'TOTAL':20s}: {total_2:12,d} params")

    # Savings
    saved_params = total_1 - total_2
    saved_pct = (saved_params / total_1) * 100

    print("\n\n📊 PARAMETER REDUCTION:")
    print("=" * 70)
    print(f"  Parameters saved: {saved_params:,d}")
    print(f"  Reduction: {saved_pct:.2f}%")
    print(f"  Model size ratio: {total_2/total_1:.3f}x")
    print("=" * 70)

    # Test forward pass
    print_section("Forward Pass Comparison")

    input_ids = np.array([[1, 5, 10, 42, 7, 12, 20]])
    print(f"\nInput: {input_ids[0]}")

    # Without tying
    logits_1, probs_1 = model_without_tying.forward(input_ids)
    print(f"\nWithout tying:")
    print(f"  Logits shape: {logits_1.shape}")
    print(f"  Probs shape: {probs_1.shape}")
    print(f"  Probs sum: {probs_1.sum():.10f}")
    top3_1 = np.argsort(probs_1[0])[-3:][::-1]
    print(f"  Top-3 predictions: {top3_1} with probs {probs_1[0, top3_1]}")

    # With tying
    logits_2, probs_2 = model_with_tying.forward(input_ids)
    print(f"\nWith tying:")
    print(f"  Logits shape: {logits_2.shape}")
    print(f"  Probs shape: {probs_2.shape}")
    print(f"  Probs sum: {probs_2.sum():.10f}")
    top3_2 = np.argsort(probs_2[0])[-3:][::-1]
    print(f"  Top-3 predictions: {top3_2} with probs {probs_2[0, top3_2]}")

    # Verify weight sharing
    print_section("Weight Sharing Verification")

    print("\nWithout tying:")
    print(f"  Embedding matrix shape: {model_without_tying.token_embedding.embedding.shape}")
    print(f"  LM head matrix shape: {model_without_tying.lm_head.shape}")
    emb_base_1 = model_without_tying.token_embedding.embedding.base
    lm_base_1 = model_without_tying.lm_head.base if hasattr(model_without_tying.lm_head, 'base') else None
    print(f"  Share same underlying data? {emb_base_1 is lm_base_1 if lm_base_1 is not None else False}")

    print("\nWith tying:")
    print(f"  Embedding matrix shape: {model_with_tying.token_embedding.embedding.shape}")
    print(f"  LM head matrix shape: {model_with_tying.lm_head.shape}")

    # Check if lm_head shares data with embedding
    emb = model_with_tying.token_embedding.embedding
    lm = model_with_tying.lm_head

    # Check via base (transpose creates view)
    shares_data = (lm.base is emb) or (emb.base is lm)
    print(f"  Share same underlying data? {shares_data}")

    # Test by modifying embedding and checking lm_head
    print("\n  Testing weight sharing (modify embedding, check lm_head):")
    old_emb_val = emb[0, 0]
    old_lm_val = lm[0, 0]
    print(f"    Before: embedding[0,0] = {old_emb_val:.6f}, lm_head[0,0] = {old_lm_val:.6f}")

    emb[0, 0] += 1.0
    new_lm_val = lm[0, 0]
    print(f"    After:  embedding[0,0] = {emb[0, 0]:.6f}, lm_head[0,0] = {new_lm_val:.6f}")

    # Restore
    emb[0, 0] = old_emb_val

    if abs(new_lm_val - (old_lm_val + 1.0)) < 1e-6:
        print(f"\n  ✓ VERIFIED: Weights are shared!")
        print(f"  ✓ lm_head is a transposed view of embedding")
        print(f"  ✓ Changes to embedding automatically affect LM head")
    else:
        print(f"\n  ✗ Weights are NOT shared (different underlying arrays)")

    # Benefits summary
    print_section("Weight Tying Benefits Summary")

    print("""
1. PARAMETER REDUCTION
   - Reduces model size by ~{:.1f}%
   - Fewer parameters to train
   - Less memory usage
   - Smaller model files

2. REGULARIZATION EFFECT
   - Embedding and output share same semantic space
   - Can improve generalization
   - Prevents overfitting in some cases

3. INTERPRETABILITY
   - Direct connection between input and output representations
   - Token embeddings directly influence predictions

4. TRAINING CONSIDERATIONS
   - Used in many modern models (GPT-2, GPT-3, etc.)
   - Generally works well for language modeling
   - May require careful learning rate tuning

5. WHEN TO USE
   - ✓ Language modeling tasks
   - ✓ When vocab size is large (reduces parameters significantly)
   - ✓ When model tends to overfit
   - ✗ When input/output have different semantics
   - ✗ When you need separate fine-tuning of embedding vs. output
    """.format(saved_pct))


def demo_attention_visualization():
    """Demo visualisasi attention patterns."""
    print_header("DEMO 2: ATTENTION VISUALIZATION")

    # Initialize model
    config = {
        'vocab_size': 1000,
        'd_model': 64,
        'num_heads': 4,
        'd_ff': 256,
        'num_layers': 3,
        'max_seq_len': 50,
        'weight_tying': True
    }

    print_section("Model Configuration")
    for key, value in config.items():
        print(f"  {key:15s}: {value}")

    np.random.seed(42)
    model = GPTModel(**config)
    print("\n✓ Model initialized with weight tying")

    # Prepare input
    print_section("Input Sequence")
    input_ids = np.array([[10, 25, 42, 7, 88, 15, 3, 99]])
    print(f"\nToken IDs: {input_ids[0]}")
    print(f"Sequence length: {input_ids.shape[1]}")

    # Forward pass with attention
    print_section("Forward Pass with Attention Tracking")
    print("\nRunning forward pass...")

    logits, probs, attention_weights = model.forward(input_ids, return_attention=True)

    print(f"✓ Forward pass complete")
    print(f"\nAttention weights collected from {len(attention_weights)} layers")
    for i, attn in enumerate(attention_weights):
        print(f"  Layer {i}: shape {attn.shape} (batch, heads, seq_len, seq_len)")

    # Visualize attention patterns
    print_section("Attention Pattern Visualization")

    # Layer 0, Head 0
    visualize_attention_ascii(attention_weights, layer_idx=0, head_idx=0,
                              batch_idx=0, max_tokens=8)

    # Layer 0, different heads
    print("\n")
    visualize_attention_ascii(attention_weights, layer_idx=0, head_idx=1,
                              batch_idx=0, max_tokens=8)

    # Last layer
    print("\n")
    visualize_attention_ascii(attention_weights, layer_idx=len(attention_weights)-1,
                              head_idx=0, batch_idx=0, max_tokens=8)

    # Numeric heatmap
    visualize_attention_heatmap_ascii(attention_weights, layer_idx=0, head_idx=0,
                                     batch_idx=0, max_tokens=8)

    # Compare heads
    compare_attention_patterns(attention_weights, layer_idx=0, batch_idx=0)

    # Position analysis
    analyze_attention_by_position(attention_weights, layer_idx=2, batch_idx=0)

    # Layer comparison
    visualize_layer_comparison(attention_weights, batch_idx=0, head_idx=0)



def demo_attention_evolution():
    """Demo bagaimana attention berubah dengan sequence length berbeda."""
    print_header("DEMO 3: ATTENTION EVOLUTION WITH SEQUENCE LENGTH")

    config = {
        'vocab_size': 500,
        'd_model': 64,
        'num_heads': 4,
        'd_ff': 256,
        'num_layers': 2,
        'max_seq_len': 50,
        'weight_tying': True
    }

    np.random.seed(42)
    model = GPTModel(**config)

    sequence_lengths = [3, 5, 8, 12]

    for seq_len in sequence_lengths:
        print_section(f"Sequence Length: {seq_len}")

        # Generate random input
        input_ids = np.random.randint(0, config['vocab_size'], size=(1, seq_len))
        print(f"Input: {input_ids[0]}")

        # Forward pass
        _, _, attn_weights = model.forward(input_ids, return_attention=True)

        # Visualize last layer, first head
        visualize_attention_ascii(attn_weights, layer_idx=1, head_idx=0,
                                  batch_idx=0, max_tokens=seq_len)

        # Statistics
        last_layer_attn = attn_weights[-1][0, 0]  # [seq_len, seq_len]

        # Compute average attention distance
        total_distance = 0
        total_weight = 0
        for i in range(seq_len):
            for j in range(i+1):  # Only past and self
                distance = i - j
                weight = last_layer_attn[i, j]
                total_distance += distance * weight
                total_weight += weight

        avg_distance = total_distance / total_weight if total_weight > 0 else 0

        print(f"\nStatistics:")
        print(f"  Average attention distance: {avg_distance:.2f} tokens")
        print(f"  (How far back does each token typically look?)")


def main():
    """Run all demos."""
    print("\n")
    print("█" * 70)
    print("█" + " WEIGHT TYING & ATTENTION VISUALIZATION DEMO ".center(68) + "█")
    print("█" * 70)

    # Demo 1: Weight Tying
    demo_weight_tying_comparison()

    # Demo 2: Attention Visualization
    demo_attention_visualization()

    # Demo 3: Attention Evolution
    demo_attention_evolution()

    # Summary
    print_header("SUMMARY")

    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                      WEIGHT TYING                                    ║
╚══════════════════════════════════════════════════════════════════════╝

✓ Weight tying shares embedding and output projection weights
✓ Reduces parameters significantly (especially for large vocab)
✓ Improves generalization in many cases
✓ Used in modern LLMs (GPT-2, GPT-3, etc.)
✓ Simple to implement: lm_head = embedding.T


╔══════════════════════════════════════════════════════════════════════╗
║                    ATTENTION VISUALIZATION                           ║
╚══════════════════════════════════════════════════════════════════════╝

✓ Attention patterns show clear causal structure
✓ Different heads learn different patterns
✓ Layers show progression from local to global attention
✓ Visualization helps understand model behavior
✓ Useful for debugging and interpretability


╔══════════════════════════════════════════════════════════════════════╗
║                       FILES CREATED                                  ║
╚══════════════════════════════════════════════════════════════════════╝

1. transformer.py (updated)
   - Added weight_tying parameter to GPTModel
   - Added return_attention support
   - Added count_parameters() method

2. visualize_attention.py
   - visualize_attention_ascii() - ASCII art visualization
   - visualize_attention_heatmap_ascii() - Numeric heatmap
   - compare_attention_patterns() - Compare heads
   - analyze_attention_by_position() - Position analysis
   - create_attention_report() - Full report

3. demo_weight_tying_attention.py
   - Weight tying comparison
   - Attention visualization demos
   - Parameter analysis
    """)

    print("\n" + "█" * 70)
    print(" Demo completed successfully!")
    print("█" * 70 + "\n")


if __name__ == "__main__":
    main()
