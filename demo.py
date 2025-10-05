"""
Demo script untuk menguji implementasi Decoder-Only Transformer (GPT-style).
Mendemonstrasikan forward pass dan prediksi token berikutnya.
"""

import numpy as np
from transformer import GPTModel, generate_next_token


def print_section(title):
    """Helper untuk print section headers."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def demo_basic_forward_pass():
    """Demo basic forward pass dari model."""
    print_section("DEMO 1: Basic Forward Pass")

    # Hyperparameters
    vocab_size = 1000      # Ukuran vocabulary
    d_model = 128          # Dimensi model
    num_heads = 8          # Jumlah attention heads
    d_ff = 512             # Dimensi feed-forward network (4 * d_model)
    num_layers = 4         # Jumlah decoder blocks
    max_seq_len = 100      # Panjang sequence maksimum

    print(f"\nModel Configuration:")
    print(f"  - Vocab Size: {vocab_size}")
    print(f"  - d_model: {d_model}")
    print(f"  - Num Heads: {num_heads}")
    print(f"  - d_ff: {d_ff}")
    print(f"  - Num Layers: {num_layers}")
    print(f"  - Max Seq Len: {max_seq_len}")

    # Inisialisasi model
    print("\n[1] Initializing GPT Model...")
    model = GPTModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        max_seq_len=max_seq_len
    )
    print("✓ Model initialized successfully!")

    # Input dummy (batch of token sequences)
    batch_size = 2
    seq_len = 10
    input_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))

    print(f"\n[2] Input Shape: {input_ids.shape}")
    print(f"    Sample Input (first sequence): {input_ids[0]}")

    # Forward pass
    print("\n[3] Running Forward Pass...")
    logits, next_token_probs = model.forward(input_ids)

    print(f"✓ Forward pass completed!")
    print(f"\n[4] Output Shapes:")
    print(f"    - Logits: {logits.shape} (batch, seq_len, vocab_size)")
    print(f"    - Next Token Probs: {next_token_probs.shape} (batch, vocab_size)")

    # Analisis output
    print(f"\n[5] Output Analysis:")
    print(f"    - Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
    print(f"    - Next token probs sum (should be ~1.0): {next_token_probs.sum(axis=-1)}")
    print(f"    - Top-5 most likely next tokens (batch 0):")

    top5_indices = np.argsort(next_token_probs[0])[-5:][::-1]
    for rank, idx in enumerate(top5_indices, 1):
        print(f"        {rank}. Token {idx}: {next_token_probs[0, idx]:.6f}")


def demo_batch_processing():
    """Demo batch processing dengan berbagai panjang sequence."""
    print_section("DEMO 2: Batch Processing")

    # Model kecil untuk demo
    vocab_size = 500
    model = GPTModel(
        vocab_size=vocab_size,
        d_model=64,
        num_heads=4,
        d_ff=256,
        num_layers=2,
        max_seq_len=50
    )

    # Test dengan batch size berbeda
    test_configs = [
        (1, 5),    # Single sequence, short
        (4, 15),   # Small batch, medium length
        (8, 20),   # Larger batch, longer sequence
    ]

    for batch_size, seq_len in test_configs:
        input_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
        logits, next_token_probs = model.forward(input_ids)

        print(f"\nBatch Size: {batch_size}, Seq Len: {seq_len}")
        print(f"  Input Shape: {input_ids.shape}")
        print(f"  Logits Shape: {logits.shape}")
        print(f"  Next Token Probs Shape: {next_token_probs.shape}")
        print(f"  ✓ All probability sums ~1.0: {np.allclose(next_token_probs.sum(axis=-1), 1.0)}")


def demo_autoregressive_generation():
    """Demo autoregressive text generation."""
    print_section("DEMO 3: Autoregressive Generation")

    vocab_size = 100
    model = GPTModel(
        vocab_size=vocab_size,
        d_model=64,
        num_heads=4,
        d_ff=256,
        num_layers=3,
        max_seq_len=50
    )

    # Start dengan satu token
    start_token = 42
    input_ids = np.array([[start_token]])  # [1, 1]
    max_new_tokens = 10

    print(f"\nGenerating {max_new_tokens} tokens autoregressively...")
    print(f"Start token: {start_token}")

    generated_sequence = [start_token]

    for i in range(max_new_tokens):
        # Generate next token
        next_token = generate_next_token(model, input_ids, temperature=0.8)

        # Append to sequence
        generated_sequence.append(next_token[0])
        input_ids = np.concatenate([input_ids, next_token.reshape(1, 1)], axis=1)

        print(f"  Step {i+1}: Generated token {next_token[0]} | Sequence length: {len(generated_sequence)}")

    print(f"\n✓ Final generated sequence: {generated_sequence}")
    print(f"  Total length: {len(generated_sequence)} tokens")


def demo_causal_masking():
    """Demo causal masking behavior."""
    print_section("DEMO 4: Causal Masking Verification")

    from transformer import create_causal_mask

    # Test causal mask
    seq_len = 5
    mask = create_causal_mask(seq_len)

    print(f"\nCausal Mask for seq_len={seq_len}:")
    print("(True = masked/tidak bisa diakses, False = tidak masked/bisa diakses)\n")

    print("     ", end="")
    for i in range(seq_len):
        print(f"t{i}    ", end="")
    print()

    for i in range(seq_len):
        print(f"t{i}  ", end="")
        for j in range(seq_len):
            symbol = "X" if mask[i, j] else "O"
            print(f" {symbol}    ", end="")
        print()

    print("\nPenjelasan:")
    print("  - 'O' = posisi bisa diakses (attend)")
    print("  - 'X' = posisi di-mask (tidak bisa attend, future token)")
    print("  - Token t0 hanya bisa attend ke dirinya sendiri")
    print("  - Token t4 bisa attend ke semua token sebelumnya (t0-t4)")


def demo_components():
    """Demo individual components."""
    print_section("DEMO 5: Individual Components")

    from transformer import (
        TokenEmbedding,
        PositionalEncoding,
        scaled_dot_product_attention,
        MultiHeadAttention,
        FeedForwardNetwork,
        layer_norm,
        softmax,
        gelu,
        create_causal_mask
    )

    # 1. Token Embedding
    print("\n[1] Token Embedding:")
    vocab_size, d_model = 100, 32
    token_emb = TokenEmbedding(vocab_size, d_model)
    tokens = np.array([[1, 5, 10]])
    embeddings = token_emb.forward(tokens)
    print(f"    Input tokens: {tokens.shape} -> Embeddings: {embeddings.shape}")

    # 2. Positional Encoding
    print("\n[2] Positional Encoding:")
    pos_enc = PositionalEncoding(max_seq_len=50, d_model=d_model)
    embeddings_with_pos = pos_enc.forward(embeddings)
    print(f"    Embeddings: {embeddings.shape} -> With positional: {embeddings_with_pos.shape}")

    # 3. Scaled Dot-Product Attention
    print("\n[3] Scaled Dot-Product Attention:")
    batch, seq_len, d_k = 2, 4, 8
    Q = np.random.randn(batch, seq_len, d_k)
    K = np.random.randn(batch, seq_len, d_k)
    V = np.random.randn(batch, seq_len, d_k)
    mask = create_causal_mask(seq_len)
    attn_out, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
    print(f"    Q,K,V: {Q.shape} -> Output: {attn_out.shape}, Weights: {attn_weights.shape}")

    # 4. Multi-Head Attention
    print("\n[4] Multi-Head Attention:")
    mha = MultiHeadAttention(d_model=64, num_heads=8)
    x = np.random.randn(2, 10, 64)
    mha_out = mha.forward(x, mask=create_causal_mask(10))
    print(f"    Input: {x.shape} -> Output: {mha_out.shape}")

    # 5. Feed-Forward Network
    print("\n[5] Feed-Forward Network:")
    ffn = FeedForwardNetwork(d_model=64, d_ff=256)
    ffn_out = ffn.forward(x)
    print(f"    Input: {x.shape} -> Output: {ffn_out.shape}")

    # 6. Layer Normalization
    print("\n[6] Layer Normalization:")
    gamma = np.ones(64)
    beta = np.zeros(64)
    ln_out = layer_norm(x, gamma, beta)
    print(f"    Input: {x.shape} -> Normalized: {ln_out.shape}")
    print(f"    Mean: {ln_out.mean(axis=-1)[0, 0]:.6f}, Var: {ln_out.var(axis=-1)[0, 0]:.6f}")

    # 7. Activation Functions
    print("\n[7] Activation Functions:")
    test_input = np.array([-2, -1, 0, 1, 2])
    print(f"    Input: {test_input}")
    print(f"    GELU output: {gelu(test_input)}")
    print(f"    Softmax output: {softmax(test_input)}")


def main():

    # Set random seed untuk reproducibility
    np.random.seed(42)

    # Run all demos
    demo_basic_forward_pass()
    demo_batch_processing()
    demo_autoregressive_generation()
    demo_causal_masking()
    demo_components()


if __name__ == "__main__":
    main()
