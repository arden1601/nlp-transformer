"""
Script untuk menghasilkan output hasil uji yang bisa ditambahkan ke laporan.
Jalankan script ini dan copy-paste output ke bagian "Bukti Uji Implementasi".
"""

import numpy as np
from transformer import GPTModel, create_causal_mask


def print_separator(title):
    """Print separator dengan title."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_1_dimensi_tensor():
    """Test 1: Validasi dimensi tensor di setiap tahap."""
    print_separator("TEST 1: VALIDASI DIMENSI TENSOR")

    # Setup
    np.random.seed(42)
    model = GPTModel(
        vocab_size=1000,
        d_model=128,
        num_heads=8,
        d_ff=512,
        num_layers=4,
        max_seq_len=100
    )

    # Input
    batch_size, seq_len = 2, 10
    input_ids = np.random.randint(0, 1000, size=(batch_size, seq_len))

    print(f"\nInput Configuration:")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Sequence Length: {seq_len}")
    print(f"  - Input Token IDs shape: {input_ids.shape}")
    print(f"  - Input sample (batch 0): {input_ids[0]}")

    # Intermediate steps
    print(f"\nDimensi di Setiap Tahap:")

    # Token embedding
    embeddings = model.token_embedding.forward(input_ids)
    print(f"  1. Token Embedding Output: {embeddings.shape}")
    print(f"     Expected: ({batch_size}, {seq_len}, 128) ✓" if embeddings.shape == (batch_size, seq_len, 128) else "     FAILED!")

    # Positional encoding
    with_pos = model.positional_encoding.forward(embeddings)
    print(f"  2. After Positional Encoding: {with_pos.shape}")
    print(f"     Expected: ({batch_size}, {seq_len}, 128) ✓" if with_pos.shape == (batch_size, seq_len, 128) else "     FAILED!")

    # Through decoder blocks
    x = with_pos.copy()
    mask = create_causal_mask(seq_len)
    for i, block in enumerate(model.decoder_blocks):
        x = block.forward(x, mask)
        print(f"  3.{i+1}. After Decoder Block {i+1}: {x.shape}")

    # Final outputs
    logits, next_token_probs = model.forward(input_ids)
    print(f"  4. Final Logits: {logits.shape}")
    print(f"     Expected: ({batch_size}, {seq_len}, 1000) ✓" if logits.shape == (batch_size, seq_len, 1000) else "     FAILED!")
    print(f"  5. Next Token Probabilities: {next_token_probs.shape}")
    print(f"     Expected: ({batch_size}, 1000) ✓" if next_token_probs.shape == (batch_size, 1000) else "     FAILED!")


def test_2_softmax_properties():
    """Test 2: Validasi properties softmax."""
    print_separator("TEST 2: VALIDASI SOFTMAX PROPERTIES")

    np.random.seed(42)
    model = GPTModel(
        vocab_size=1000,
        d_model=128,
        num_heads=8,
        d_ff=512,
        num_layers=4,
        max_seq_len=100
    )

    # Forward pass
    input_ids = np.array([[1, 5, 10, 42, 7], [2, 8, 15, 30, 9]])
    logits, next_token_probs = model.forward(input_ids)

    print(f"\nTest Input:")
    print(f"  - Batch 0: {input_ids[0]}")
    print(f"  - Batch 1: {input_ids[1]}")

    # Test 1: Sum = 1.0
    print(f"\n[Property 1] Sum of Probabilities = 1.0")
    for i in range(len(next_token_probs)):
        prob_sum = next_token_probs[i].sum()
        print(f"  - Batch {i} sum: {prob_sum:.10f} {'✓' if np.isclose(prob_sum, 1.0) else '✗'}")

    # Test 2: Range [0, 1]
    print(f"\n[Property 2] All Probabilities in Range [0, 1]")
    print(f"  - Minimum probability: {next_token_probs.min():.10f}")
    print(f"  - Maximum probability: {next_token_probs.max():.10f}")
    print(f"  - All in [0,1]: {'✓' if (next_token_probs >= 0).all() and (next_token_probs <= 1).all() else '✗'}")

    # Test 3: Top predictions
    print(f"\n[Property 3] Top-5 Predictions per Batch")
    for batch_idx in range(len(next_token_probs)):
        print(f"\n  Batch {batch_idx}:")
        top5_indices = np.argsort(next_token_probs[batch_idx])[-5:][::-1]
        for rank, token_id in enumerate(top5_indices, 1):
            prob = next_token_probs[batch_idx, token_id]
            print(f"    {rank}. Token {token_id:4d}: {prob:.8f} ({prob*100:.4f}%)")

    # Test 4: Probabilitas non-zero
    print(f"\n[Property 4] Non-Zero Probabilities")
    for i in range(len(next_token_probs)):
        non_zero_count = np.count_nonzero(next_token_probs[i] > 1e-10)
        print(f"  - Batch {i}: {non_zero_count}/{len(next_token_probs[i])} tokens have P > 1e-10")



def test_3_causal_masking():
    """Test 3: Validasi causal masking."""
    print_separator("TEST 3: VALIDASI CAUSAL MASKING")

    # Create mask
    seq_len = 6
    mask = create_causal_mask(seq_len)

    print(f"\nCausal Mask Matrix (seq_len={seq_len}):")
    print("True = Masked (tidak bisa diakses), False = Not Masked (bisa diakses)\n")

    # Print header
    print("       ", end="")
    for i in range(seq_len):
        print(f"t{i}    ", end="")
    print()

    # Print mask
    for i in range(seq_len):
        print(f"  t{i}  ", end="")
        for j in range(seq_len):
            symbol = " X " if mask[i, j] else " O "
            print(f" {symbol}  ", end="")
        print()

    print("\nLegend:")
    print("  O = False (dapat diakses)")
    print("  X = True  (di-mask, tidak dapat diakses)")

    # Validasi properties
    print(f"\n[Validation 1] Upper Triangular (above diagonal) = True")
    upper_correct = True
    for i in range(seq_len):
        for j in range(i+1, seq_len):
            if not mask[i, j]:
                upper_correct = False
                break
    print(f"  Result: {'✓ PASS' if upper_correct else '✗ FAIL'}")

    print(f"\n[Validation 2] Lower Triangular + Diagonal = False")
    lower_correct = True
    for i in range(seq_len):
        for j in range(i+1):
            if mask[i, j]:
                lower_correct = False
                break
    print(f"  Result: {'✓ PASS' if lower_correct else '✗ FAIL'}")

    # Test dengan attention
    print(f"\n[Validation 3] Attention Scores dengan Mask")
    from transformer import scaled_dot_product_attention

    # Dummy Q, K, V
    batch_size = 1
    d_k = 8
    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_k)

    # Attention dengan mask
    output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

    print(f"  Attention Weights Shape: {attn_weights.shape}")
    print(f"  Sample Attention Weights (batch 0):")
    print(f"  (Each row shows where that position attends to)\n")

    for i in range(min(5, seq_len)):
        print(f"    Position t{i} attends to: ", end="")
        for j in range(seq_len):
            if attn_weights[0, i, j] > 1e-5:
                print(f"t{j}({attn_weights[0, i, j]:.3f}) ", end="")
        print()

    # Check future positions have ~0 attention
    print(f"\n[Validation 4] Future Positions Have Near-Zero Attention")
    future_attention_valid = True
    for i in range(seq_len):
        for j in range(i+1, seq_len):
            if attn_weights[0, i, j] > 1e-5:
                future_attention_valid = False
                print(f"  ✗ Position {i} attends to future position {j}: {attn_weights[0, i, j]:.6f}")
    if future_attention_valid:
        print(f"  ✓ PASS - All future positions have attention weight < 1e-5")



def test_4_autoregressive_generation():
    """Test 4: Autoregressive generation."""
    print_separator("TEST 4: AUTOREGRESSIVE GENERATION")

    np.random.seed(42)
    from transformer import generate_next_token

    model = GPTModel(
        vocab_size=1000,
        d_model=128,
        num_heads=8,
        d_ff=512,
        num_layers=4,
        max_seq_len=100
    )

    start_token = 42
    num_tokens = 12

    print(f"\nConfiguration:")
    print(f"  - Start Token: {start_token}")
    print(f"  - Number of Tokens to Generate: {num_tokens}")
    print(f"  - Temperature: 0.8")

    print(f"\nGeneration Process:")
    current_sequence = np.array([[start_token]])
    print(f"  Initial: {current_sequence[0]}")

    for step in range(num_tokens):
        next_token = generate_next_token(model, current_sequence, temperature=0.8)
        current_sequence = np.concatenate([current_sequence, next_token.reshape(1, 1)], axis=1)
        print(f"  Step {step+1:2d}: Generated token {next_token[0]:3d} -> Sequence: {current_sequence[0]}")

    print(f"\n✅ Final Generated Sequence:")
    print(f"   {list(current_sequence[0])}")
    print(f"   Length: {len(current_sequence[0])} tokens")

    # Temperature comparison
    print(f"\n" + "-" * 70)
    print(f"  Temperature Effect Comparison")
    print("-" * 70)

    temps = [0.5, 1.0, 2.0]
    start_seq = np.array([[42, 15, 8]])

    for temp in temps:
        print(f"\n  Temperature = {temp}:")
        seq = start_seq.copy()
        for _ in range(8):
            next_token = generate_next_token(model, seq, temperature=temp)
            seq = np.concatenate([seq, next_token.reshape(1, 1)], axis=1)
        print(f"    Generated: {list(seq[0])}")


def test_5_layer_normalization():
    """Test 5: Layer normalization properties."""
    print_separator("TEST 5: LAYER NORMALIZATION")

    from transformer import layer_norm

    # Create test data
    batch, seq_len, d_model = 2, 5, 64
    x = np.random.randn(batch, seq_len, d_model) * 3 + 5  # Mean ~5, std ~3
    gamma = np.ones(d_model)
    beta = np.zeros(d_model)

    print(f"\nInput Statistics (before normalization):")
    print(f"  Shape: {x.shape}")
    print(f"  Mean (sample): {x[0, 0].mean():.6f}")
    print(f"  Std (sample): {x[0, 0].std():.6f}")

    # Apply layer norm
    x_norm = layer_norm(x, gamma, beta)

    print(f"\nOutput Statistics (after normalization):")
    print(f"  Shape: {x_norm.shape}")
    for i in range(min(3, batch)):
        for j in range(min(3, seq_len)):
            mean = x_norm[i, j].mean()
            std = x_norm[i, j].std()
            print(f"  Batch {i}, Position {j}: Mean={mean:.8f}, Std={std:.6f}")

    # Validate
    print(f"\n[Validation] Mean ≈ 0, Std ≈ 1")
    all_means = x_norm.mean(axis=-1).flatten()
    all_stds = x_norm.std(axis=-1).flatten()

    mean_check = np.allclose(all_means, 0, atol=1e-6)
    std_check = np.allclose(all_stds, 1, atol=1e-5)

    print(f"  Mean close to 0: {'✓ PASS' if mean_check else '✗ FAIL'}")
    print(f"  Std close to 1: {'✓ PASS' if std_check else '✗ FAIL'}")



def main():
    """Run all tests."""
    print("\n")
    print("█" * 70)
    print("█" + " HASIL UJI IMPLEMENTASI TRANSFORMER ".center(68) + "█")
    print("█" * 70)

    test_1_dimensi_tensor()
    test_2_softmax_properties()
    test_3_causal_masking()
    test_4_autoregressive_generation()
    test_5_layer_normalization()

if __name__ == "__main__":
    main()
