"""
Unit tests untuk implementasi Transformer.
Jalankan dengan: python -m unittest test_transformer.py
Atau dengan pytest: pytest test_transformer.py -v
"""

import unittest
import numpy as np
import sys
from io import StringIO

from transformer import (
    GPTModel,
    TokenEmbedding,
    PositionalEncoding,
    MultiHeadAttention,
    FeedForwardNetwork,
    TransformerDecoderBlock,
    create_causal_mask,
    scaled_dot_product_attention,
    layer_norm,
    softmax,
    gelu,
    generate_next_token
)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions (softmax, layer_norm, gelu)."""

    def test_softmax_sum_to_one(self):
        """Test bahwa softmax sum ke 1.0."""
        x = np.random.randn(10)
        result = softmax(x)
        self.assertAlmostEqual(result.sum(), 1.0, places=6)

    def test_softmax_all_positive(self):
        """Test bahwa semua nilai softmax positif."""
        x = np.random.randn(10)
        result = softmax(x)
        self.assertTrue((result >= 0).all())
        self.assertTrue((result <= 1).all())

    def test_softmax_2d(self):
        """Test softmax untuk 2D array."""
        x = np.random.randn(5, 10)
        result = softmax(x, axis=-1)
        self.assertEqual(result.shape, (5, 10))
        for i in range(5):
            self.assertAlmostEqual(result[i].sum(), 1.0, places=6)

    def test_layer_norm_mean_zero(self):
        """Test bahwa layer norm menghasilkan mean ~0."""
        x = np.random.randn(2, 5, 64) * 3 + 5
        gamma = np.ones(64)
        beta = np.zeros(64)
        result = layer_norm(x, gamma, beta)

        # Check mean across last dimension
        means = result.mean(axis=-1)
        self.assertTrue(np.allclose(means, 0, atol=1e-6))

    def test_layer_norm_std_one(self):
        """Test bahwa layer norm menghasilkan std ~1."""
        x = np.random.randn(2, 5, 64) * 3 + 5
        gamma = np.ones(64)
        beta = np.zeros(64)
        result = layer_norm(x, gamma, beta)

        # Check std across last dimension
        stds = result.std(axis=-1)
        self.assertTrue(np.allclose(stds, 1, atol=1e-5))

    def test_gelu_properties(self):
        """Test GELU activation properties."""
        x = np.array([-2, -1, 0, 1, 2])
        result = gelu(x)

        # GELU(0) should be close to 0
        self.assertAlmostEqual(result[2], 0, places=5)

        # GELU should be approximately monotonic for positive values
        # (GELU is smooth and generally increasing, but not strictly everywhere)
        self.assertLess(result[3], result[4])  # 1 < 2

        # Negative values should be negative, positive values positive (mostly)
        self.assertLess(result[0], 0)  # GELU(-2) < 0
        self.assertGreater(result[4], 0)  # GELU(2) > 0


class TestCausalMask(unittest.TestCase):
    """Test causal masking functionality."""

    def test_mask_shape(self):
        """Test bahwa mask memiliki shape yang benar."""
        seq_len = 10
        mask = create_causal_mask(seq_len)
        self.assertEqual(mask.shape, (seq_len, seq_len))

    def test_mask_upper_triangular(self):
        """Test bahwa mask adalah upper triangular."""
        seq_len = 5
        mask = create_causal_mask(seq_len)

        # Upper triangle (excluding diagonal) should be True
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                self.assertTrue(mask[i, j], f"Position ({i},{j}) should be masked")

    def test_mask_lower_triangular_false(self):
        """Test bahwa lower triangle + diagonal adalah False."""
        seq_len = 5
        mask = create_causal_mask(seq_len)

        # Lower triangle and diagonal should be False
        for i in range(seq_len):
            for j in range(i + 1):
                self.assertFalse(mask[i, j], f"Position ({i},{j}) should NOT be masked")

    def test_mask_dtype(self):
        """Test bahwa mask adalah boolean."""
        mask = create_causal_mask(5)
        self.assertEqual(mask.dtype, bool)


class TestTokenEmbedding(unittest.TestCase):
    """Test token embedding layer."""

    def setUp(self):
        """Setup untuk setiap test."""
        np.random.seed(42)
        self.vocab_size = 100
        self.d_model = 32
        self.embedding = TokenEmbedding(self.vocab_size, self.d_model)

    def test_embedding_shape(self):
        """Test shape dari embedding matrix."""
        self.assertEqual(self.embedding.embedding.shape, (self.vocab_size, self.d_model))

    def test_forward_shape(self):
        """Test output shape dari forward pass."""
        input_ids = np.array([[1, 5, 10], [2, 8, 15]])
        output = self.embedding.forward(input_ids)
        self.assertEqual(output.shape, (2, 3, self.d_model))

    def test_forward_lookup(self):
        """Test bahwa forward melakukan lookup yang benar."""
        input_ids = np.array([[5]])
        output = self.embedding.forward(input_ids)
        expected = self.embedding.embedding[5]
        np.testing.assert_array_almost_equal(output[0, 0], expected)


class TestPositionalEncoding(unittest.TestCase):
    """Test positional encoding."""

    def setUp(self):
        """Setup untuk setiap test."""
        self.max_seq_len = 50
        self.d_model = 32
        self.pos_enc = PositionalEncoding(self.max_seq_len, self.d_model)

    def test_encoding_shape(self):
        """Test shape dari positional encoding."""
        self.assertEqual(self.pos_enc.encoding.shape, (self.max_seq_len, self.d_model))

    def test_forward_adds_encoding(self):
        """Test bahwa forward menambahkan positional encoding."""
        x = np.random.randn(2, 5, self.d_model)
        output = self.pos_enc.forward(x)

        # Output should be x + encoding[:5]
        expected = x + self.pos_enc.encoding[:5]
        np.testing.assert_array_almost_equal(output, expected)

    def test_sinusoidal_pattern(self):
        """Test bahwa encoding menggunakan pola sinusoidal."""
        enc = self.pos_enc.encoding

        # Even indices should use sine
        # Odd indices should use cosine
        # Check first position
        pos = 0
        for i in range(0, self.d_model, 2):
            div_term = np.exp(i * -(np.log(10000.0) / self.d_model))
            expected_sin = np.sin(pos * div_term)
            self.assertAlmostEqual(enc[pos, i], expected_sin, places=5)


class TestScaledDotProductAttention(unittest.TestCase):
    """Test scaled dot-product attention."""

    def setUp(self):
        """Setup untuk setiap test."""
        np.random.seed(42)
        self.batch_size = 2
        self.seq_len = 4
        self.d_k = 8

    def test_output_shape(self):
        """Test output shape."""
        Q = np.random.randn(self.batch_size, self.seq_len, self.d_k)
        K = np.random.randn(self.batch_size, self.seq_len, self.d_k)
        V = np.random.randn(self.batch_size, self.seq_len, self.d_k)

        output, weights = scaled_dot_product_attention(Q, K, V)

        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_k))
        self.assertEqual(weights.shape, (self.batch_size, self.seq_len, self.seq_len))

    def test_attention_weights_sum_to_one(self):
        """Test bahwa attention weights sum ke 1."""
        Q = np.random.randn(self.batch_size, self.seq_len, self.d_k)
        K = np.random.randn(self.batch_size, self.seq_len, self.d_k)
        V = np.random.randn(self.batch_size, self.seq_len, self.d_k)

        _, weights = scaled_dot_product_attention(Q, K, V)

        # Sum over keys (last dimension)
        sums = weights.sum(axis=-1)
        self.assertTrue(np.allclose(sums, 1.0))

    def test_causal_mask_blocks_future(self):
        """Test bahwa causal mask mencegah attention ke future."""
        Q = np.random.randn(self.batch_size, self.seq_len, self.d_k)
        K = np.random.randn(self.batch_size, self.seq_len, self.d_k)
        V = np.random.randn(self.batch_size, self.seq_len, self.d_k)
        mask = create_causal_mask(self.seq_len)

        _, weights = scaled_dot_product_attention(Q, K, V, mask)

        # Future positions should have near-zero attention
        for i in range(self.seq_len):
            for j in range(i + 1, self.seq_len):
                self.assertLess(weights[0, i, j], 1e-5,
                               f"Position {i} should not attend to future {j}")


class TestMultiHeadAttention(unittest.TestCase):
    """Test multi-head attention."""

    def setUp(self):
        """Setup untuk setiap test."""
        np.random.seed(42)
        self.d_model = 64
        self.num_heads = 4
        self.mha = MultiHeadAttention(self.d_model, self.num_heads)

    def test_output_shape(self):
        """Test output shape."""
        x = np.random.randn(2, 10, self.d_model)
        output = self.mha.forward(x)
        self.assertEqual(output.shape, x.shape)

    def test_with_mask(self):
        """Test dengan causal mask."""
        x = np.random.randn(2, 10, self.d_model)
        mask = create_causal_mask(10)
        output = self.mha.forward(x, mask)
        self.assertEqual(output.shape, x.shape)

    def test_return_attention(self):
        """Test return attention weights."""
        x = np.random.randn(2, 10, self.d_model)
        output, attn = self.mha.forward(x, return_attention=True)

        self.assertEqual(output.shape, x.shape)
        self.assertEqual(attn.shape, (2, self.num_heads, 10, 10))


class TestFeedForwardNetwork(unittest.TestCase):
    """Test feed-forward network."""

    def setUp(self):
        """Setup untuk setiap test."""
        np.random.seed(42)
        self.d_model = 64
        self.d_ff = 256
        self.ffn = FeedForwardNetwork(self.d_model, self.d_ff)

    def test_output_shape(self):
        """Test output shape."""
        x = np.random.randn(2, 10, self.d_model)
        output = self.ffn.forward(x)
        self.assertEqual(output.shape, x.shape)

    def test_hidden_dimension(self):
        """Test bahwa hidden layer memiliki d_ff dimensions."""
        self.assertEqual(self.ffn.W1.shape, (self.d_model, self.d_ff))
        self.assertEqual(self.ffn.W2.shape, (self.d_ff, self.d_model))


class TestTransformerDecoderBlock(unittest.TestCase):
    """Test transformer decoder block."""

    def setUp(self):
        """Setup untuk setiap test."""
        np.random.seed(42)
        self.d_model = 64
        self.num_heads = 4
        self.d_ff = 256
        self.block = TransformerDecoderBlock(self.d_model, self.num_heads, self.d_ff)

    def test_output_shape(self):
        """Test output shape."""
        x = np.random.randn(2, 10, self.d_model)
        mask = create_causal_mask(10)
        output = self.block.forward(x, mask)
        self.assertEqual(output.shape, x.shape)

    def test_return_attention(self):
        """Test return attention weights."""
        x = np.random.randn(2, 10, self.d_model)
        mask = create_causal_mask(10)
        output, attn = self.block.forward(x, mask, return_attention=True)

        self.assertEqual(output.shape, x.shape)
        self.assertEqual(attn.shape, (2, self.num_heads, 10, 10))


class TestGPTModel(unittest.TestCase):
    """Test GPT model."""

    def setUp(self):
        """Setup untuk setiap test."""
        np.random.seed(42)
        self.config = {
            'vocab_size': 1000,
            'd_model': 128,
            'num_heads': 8,
            'd_ff': 512,
            'num_layers': 4,
            'max_seq_len': 100
        }

    def test_model_initialization(self):
        """Test bahwa model dapat diinisialisasi."""
        model = GPTModel(**self.config)
        self.assertIsNotNone(model)

    def test_forward_output_shapes(self):
        """Test output shapes dari forward pass."""
        model = GPTModel(**self.config)
        input_ids = np.array([[1, 5, 10, 42, 7]])

        logits, probs = model.forward(input_ids)

        self.assertEqual(logits.shape, (1, 5, self.config['vocab_size']))
        self.assertEqual(probs.shape, (1, self.config['vocab_size']))

    def test_next_token_probs_valid(self):
        """Test bahwa next token probabilities valid."""
        model = GPTModel(**self.config)
        input_ids = np.array([[1, 5, 10, 42, 7]])

        _, probs = model.forward(input_ids)

        # Should sum to 1
        self.assertAlmostEqual(probs.sum(), 1.0, places=6)

        # All values should be in [0, 1]
        self.assertTrue((probs >= 0).all())
        self.assertTrue((probs <= 1).all())

    def test_batch_processing(self):
        """Test batch processing."""
        model = GPTModel(**self.config)
        input_ids = np.array([[1, 5, 10], [2, 8, 15]])

        logits, probs = model.forward(input_ids)

        self.assertEqual(logits.shape, (2, 3, self.config['vocab_size']))
        self.assertEqual(probs.shape, (2, self.config['vocab_size']))

    def test_return_attention(self):
        """Test return attention weights."""
        model = GPTModel(**self.config)
        input_ids = np.array([[1, 5, 10, 42, 7]])

        logits, probs, attn_weights = model.forward(input_ids, return_attention=True)

        self.assertEqual(len(attn_weights), self.config['num_layers'])
        for attn in attn_weights:
            self.assertEqual(attn.shape, (1, self.config['num_heads'], 5, 5))


class TestWeightTying(unittest.TestCase):
    """Test weight tying functionality."""

    def setUp(self):
        """Setup untuk setiap test."""
        np.random.seed(42)
        self.config = {
            'vocab_size': 1000,
            'd_model': 128,
            'num_heads': 8,
            'd_ff': 512,
            'num_layers': 4,
            'max_seq_len': 100
        }

    def test_weight_tying_disabled(self):
        """Test model tanpa weight tying."""
        model = GPTModel(**self.config, weight_tying=False)
        self.assertFalse(model.weight_tying)

        # Embedding and lm_head should be different objects
        emb = model.token_embedding.embedding
        lm = model.lm_head
        self.assertFalse(lm.base is emb)

    def test_weight_tying_enabled(self):
        """Test model dengan weight tying."""
        model = GPTModel(**self.config, weight_tying=True)
        self.assertTrue(model.weight_tying)

        # lm_head should be transpose view of embedding
        emb = model.token_embedding.embedding
        lm = model.lm_head

        # Check they share data
        self.assertTrue(lm.base is emb)

    def test_weight_tying_shares_memory(self):
        """Test bahwa weight tying benar-benar share memory."""
        model = GPTModel(**self.config, weight_tying=True)

        emb = model.token_embedding.embedding
        lm = model.lm_head

        # Modify embedding
        old_val = emb[0, 0]
        emb[0, 0] += 1.0

        # Check lm_head changed
        self.assertAlmostEqual(lm[0, 0], old_val + 1.0, places=6)

        # Restore
        emb[0, 0] = old_val

    def test_parameter_count_reduction(self):
        """Test bahwa weight tying mengurangi parameter count."""
        model_without = GPTModel(**self.config, weight_tying=False)
        model_with = GPTModel(**self.config, weight_tying=True)

        total_without, _ = model_without.count_parameters()
        total_with, _ = model_with.count_parameters()

        # Should have fewer parameters
        self.assertLess(total_with, total_without)

        # Reduction should be vocab_size * d_model
        expected_reduction = self.config['vocab_size'] * self.config['d_model']
        actual_reduction = total_without - total_with
        self.assertEqual(actual_reduction, expected_reduction)


class TestParameterCounting(unittest.TestCase):
    """Test parameter counting functionality."""

    def test_count_parameters(self):
        """Test count_parameters method."""
        config = {
            'vocab_size': 1000,
            'd_model': 128,
            'num_heads': 8,
            'd_ff': 512,
            'num_layers': 4,
            'max_seq_len': 100
        }
        model = GPTModel(**config)

        total, breakdown = model.count_parameters()

        # Check total is sum of components
        self.assertEqual(total, sum(breakdown.values()))

        # Check all components present
        self.assertIn('token_embedding', breakdown)
        self.assertIn('lm_head', breakdown)
        self.assertIn('decoder_blocks', breakdown)
        self.assertIn('final_ln', breakdown)


class TestAutoRegressiveGeneration(unittest.TestCase):
    """Test autoregressive generation."""

    def setUp(self):
        """Setup untuk setiap test."""
        np.random.seed(42)
        self.model = GPTModel(
            vocab_size=1000,
            d_model=128,
            num_heads=8,
            d_ff=512,
            num_layers=4,
            max_seq_len=100
        )

    def test_generate_next_token(self):
        """Test generate_next_token function."""
        input_ids = np.array([[42]])
        next_token = generate_next_token(self.model, input_ids)

        self.assertEqual(next_token.shape, (1,))
        self.assertGreaterEqual(next_token[0], 0)
        self.assertLess(next_token[0], 1000)

    def test_autoregressive_sequence_grows(self):
        """Test bahwa sequence bertambah panjang saat generation."""
        input_ids = np.array([[42]])
        initial_len = input_ids.shape[1]

        for _ in range(5):
            next_token = generate_next_token(self.model, input_ids)
            input_ids = np.concatenate([input_ids, next_token.reshape(1, 1)], axis=1)

        self.assertEqual(input_ids.shape[1], initial_len + 5)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases dan error conditions."""

    def test_single_token_sequence(self):
        """Test dengan sequence length 1."""
        model = GPTModel(
            vocab_size=100,
            d_model=32,
            num_heads=4,
            d_ff=128,
            num_layers=2,
            max_seq_len=50
        )

        input_ids = np.array([[42]])
        logits, probs = model.forward(input_ids)

        self.assertEqual(logits.shape, (1, 1, 100))
        self.assertEqual(probs.shape, (1, 100))

    def test_max_sequence_length(self):
        """Test dengan max sequence length."""
        max_len = 20
        model = GPTModel(
            vocab_size=100,
            d_model=32,
            num_heads=4,
            d_ff=128,
            num_layers=2,
            max_seq_len=max_len
        )

        input_ids = np.random.randint(0, 100, size=(1, max_len))
        logits, probs = model.forward(input_ids)

        self.assertEqual(logits.shape, (1, max_len, 100))
        self.assertEqual(probs.shape, (1, 100))


def run_tests_with_summary():
    """Run all tests dan print summary."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])

    # Run tests dengan verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print(" TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✓ ALL TESTS PASSED!")
    else:
        print("\n✗ SOME TESTS FAILED")

    print("=" * 70 + "\n")

    return result


if __name__ == '__main__':
    # Run dengan summary
    result = run_tests_with_summary()

    # Exit dengan appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
