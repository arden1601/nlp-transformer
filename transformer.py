import numpy as np


# ==================== UTILITY FUNCTIONS ====================

def softmax(x, axis=-1):
    """
    Implementasi numerically stable softmax.

    Args:
        x: array dengan shape arbitrary
        axis: axis untuk menghitung softmax

    Returns:
        Softmax dari x pada axis yang ditentukan
    """
    # Subtract max for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def layer_norm(x, gamma, beta, eps=1e-5):
    """
    Layer Normalization.

    Args:
        x: input dengan shape [batch, seq_len, d_model]
        gamma: scale parameter [d_model]
        beta: shift parameter [d_model]
        eps: epsilon untuk stabilitas numerik

    Returns:
        Normalized output dengan shape yang sama dengan x
    """
    # Normalize across the last dimension (d_model)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta


def gelu(x):
    """
    GELU activation function (Gaussian Error Linear Unit).
    Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


# ==================== EMBEDDING & POSITIONAL ENCODING ====================

class TokenEmbedding:
    """Token Embedding Layer."""

    def __init__(self, vocab_size, d_model):
        """
        Args:
            vocab_size: ukuran vocabulary
            d_model: dimensi embedding
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        # Initialize embedding matrix dengan distribusi normal
        self.embedding = np.random.randn(vocab_size, d_model) * 0.02

    def forward(self, x):
        """
        Args:
            x: token indices dengan shape [batch, seq_len]

        Returns:
            Embeddings dengan shape [batch, seq_len, d_model]
        """
        return self.embedding[x]


class PositionalEncoding:
    """Sinusoidal Positional Encoding."""

    def __init__(self, max_seq_len, d_model):
        """
        Args:
            max_seq_len: panjang sequence maksimum
            d_model: dimensi model
        """
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.encoding = self._create_encoding()

    def _create_encoding(self):
        """
        Membuat sinusoidal positional encoding.
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """
        position = np.arange(self.max_seq_len)[:, np.newaxis]  # [max_seq_len, 1]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))

        encoding = np.zeros((self.max_seq_len, self.d_model))
        encoding[:, 0::2] = np.sin(position * div_term)  # Even indices
        encoding[:, 1::2] = np.cos(position * div_term)  # Odd indices

        return encoding

    def forward(self, x):
        """
        Args:
            x: embeddings dengan shape [batch, seq_len, d_model]

        Returns:
            Embeddings + positional encoding dengan shape yang sama
        """
        seq_len = x.shape[1]
        return x + self.encoding[:seq_len, :]


# ==================== ATTENTION MECHANISMS ====================

def create_causal_mask(seq_len):
    """
    Membuat causal mask untuk mencegah attention ke token masa depan.

    Args:
        seq_len: panjang sequence

    Returns:
        Mask dengan shape [seq_len, seq_len], True menandakan posisi yang di-mask
    """
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    return mask


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled Dot-Product Attention dengan optional causal masking.

    Args:
        Q: Query matrix [batch, seq_len, d_k]
        K: Key matrix [batch, seq_len, d_k]
        V: Value matrix [batch, seq_len, d_v]
        mask: Optional causal mask [seq_len, seq_len]

    Returns:
        Output [batch, seq_len, d_v] dan attention weights [batch, seq_len, seq_len]
    """
    d_k = Q.shape[-1]

    # Compute attention scores: QK^T / sqrt(d_k)
    # Q: [batch, seq_len, d_k], K^T: [batch, d_k, seq_len]
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)  # [batch, seq_len, seq_len]

    # Apply causal mask (set future positions to -inf)
    if mask is not None:
        scores = np.where(mask, -1e9, scores)

    # Apply softmax to get attention weights
    attention_weights = softmax(scores, axis=-1)  # [batch, seq_len, seq_len]

    # Apply attention weights to values
    output = np.matmul(attention_weights, V)  # [batch, seq_len, d_v]

    return output, attention_weights


class MultiHeadAttention:
    """Multi-Head Attention Layer."""

    def __init__(self, d_model, num_heads):
        """
        Args:
            d_model: dimensi model
            num_heads: jumlah attention heads
        """
        assert d_model % num_heads == 0, "d_model harus habis dibagi num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimensi per head

        # Weight matrices untuk Q, K, V projections
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02

        # Output projection
        self.W_o = np.random.randn(d_model, d_model) * 0.02

    def _split_heads(self, x):
        """
        Split the last dimension into (num_heads, d_k).

        Args:
            x: [batch, seq_len, d_model]

        Returns:
            [batch, num_heads, seq_len, d_k]
        """
        batch_size, seq_len, d_model = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)  # [batch, num_heads, seq_len, d_k]

    def _combine_heads(self, x):
        """
        Combine heads back.

        Args:
            x: [batch, num_heads, seq_len, d_k]

        Returns:
            [batch, seq_len, d_model]
        """
        batch_size, num_heads, seq_len, d_k = x.shape
        x = x.transpose(0, 2, 1, 3)  # [batch, seq_len, num_heads, d_k]
        return x.reshape(batch_size, seq_len, self.d_model)

    def forward(self, x, mask=None, return_attention=False):
        """
        Args:
            x: input [batch, seq_len, d_model]
            mask: optional causal mask [seq_len, seq_len]
            return_attention: whether to return attention weights

        Returns:
            output [batch, seq_len, d_model]
            attention_weights (optional): [batch, num_heads, seq_len, seq_len]
        """
        batch_size = x.shape[0]

        # Linear projections untuk Q, K, V
        Q = np.matmul(x, self.W_q)  # [batch, seq_len, d_model]
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)

        # Split into multiple heads
        Q = self._split_heads(Q)  # [batch, num_heads, seq_len, d_k]
        K = self._split_heads(K)
        V = self._split_heads(V)

        # Apply scaled dot-product attention untuk setiap head
        attention_output = []
        all_attention_weights = []
        for i in range(self.num_heads):
            q_head = Q[:, i, :, :]  # [batch, seq_len, d_k]
            k_head = K[:, i, :, :]
            v_head = V[:, i, :, :]

            attn_out, attn_weights = scaled_dot_product_attention(q_head, k_head, v_head, mask)
            attention_output.append(attn_out)
            if return_attention:
                all_attention_weights.append(attn_weights)

        # Stack heads: [num_heads, batch, seq_len, d_k] -> [batch, num_heads, seq_len, d_k]
        attention_output = np.stack(attention_output, axis=1)

        # Combine heads
        concat_attention = self._combine_heads(attention_output)  # [batch, seq_len, d_model]

        # Final linear projection
        output = np.matmul(concat_attention, self.W_o)  # [batch, seq_len, d_model]

        if return_attention:
            # Stack attention weights: [num_heads, batch, seq_len, seq_len] -> [batch, num_heads, seq_len, seq_len]
            all_attention_weights = np.stack(all_attention_weights, axis=1)
            return output, all_attention_weights

        return output


# ==================== FEED-FORWARD NETWORK ====================

class FeedForwardNetwork:
    """Position-wise Feed-Forward Network."""

    def __init__(self, d_model, d_ff):
        """
        Args:
            d_model: dimensi model
            d_ff: dimensi hidden layer (biasanya 4 * d_model)
        """
        self.d_model = d_model
        self.d_ff = d_ff

        # Two-layer FFN dengan GELU activation
        self.W1 = np.random.randn(d_model, d_ff) * 0.02
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.02
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        """
        Args:
            x: input [batch, seq_len, d_model]

        Returns:
            output [batch, seq_len, d_model]
        """
        # First layer dengan GELU activation
        hidden = gelu(np.matmul(x, self.W1) + self.b1)  # [batch, seq_len, d_ff]

        # Second layer
        output = np.matmul(hidden, self.W2) + self.b2  # [batch, seq_len, d_model]

        return output


# ==================== TRANSFORMER DECODER BLOCK ====================

class TransformerDecoderBlock:
    """Single Transformer Decoder Block dengan Pre-Layer Normalization."""

    def __init__(self, d_model, num_heads, d_ff):
        """
        Args:
            d_model: dimensi model
            num_heads: jumlah attention heads
            d_ff: dimensi feed-forward network
        """
        self.d_model = d_model

        # Multi-Head Attention
        self.attention = MultiHeadAttention(d_model, num_heads)

        # Feed-Forward Network
        self.ffn = FeedForwardNetwork(d_model, d_ff)

        # Layer Normalization parameters (Pre-Norm)
        self.ln1_gamma = np.ones(d_model)
        self.ln1_beta = np.zeros(d_model)
        self.ln2_gamma = np.ones(d_model)
        self.ln2_beta = np.zeros(d_model)

    def forward(self, x, mask=None, return_attention=False):
        """
        Args:
            x: input [batch, seq_len, d_model]
            mask: causal mask [seq_len, seq_len]
            return_attention: whether to return attention weights

        Returns:
            output [batch, seq_len, d_model]
            attention_weights (optional): [batch, num_heads, seq_len, seq_len]
        """
        # Pre-Norm Architecture

        # 1. Multi-Head Attention dengan residual connection
        x_norm1 = layer_norm(x, self.ln1_gamma, self.ln1_beta)
        if return_attention:
            attn_output, attn_weights = self.attention.forward(x_norm1, mask, return_attention=True)
        else:
            attn_output = self.attention.forward(x_norm1, mask)
        x = x + attn_output  # Residual connection

        # 2. Feed-Forward Network dengan residual connection
        x_norm2 = layer_norm(x, self.ln2_gamma, self.ln2_beta)
        ffn_output = self.ffn.forward(x_norm2)
        x = x + ffn_output  # Residual connection

        if return_attention:
            return x, attn_weights
        return x


# ==================== GPT MODEL ====================

class GPTModel:
    """Decoder-Only Transformer Model (GPT-style)."""

    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, weight_tying=False):
        """
        Args:
            vocab_size: ukuran vocabulary
            d_model: dimensi model
            num_heads: jumlah attention heads
            d_ff: dimensi feed-forward network
            num_layers: jumlah decoder blocks
            max_seq_len: panjang sequence maksimum
            weight_tying: whether to tie input embedding and output projection weights
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.weight_tying = weight_tying

        # Token Embedding
        self.token_embedding = TokenEmbedding(vocab_size, d_model)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(max_seq_len, d_model)

        # Stack of Decoder Blocks
        self.decoder_blocks = [
            TransformerDecoderBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]

        # Final Layer Norm
        self.ln_f_gamma = np.ones(d_model)
        self.ln_f_beta = np.zeros(d_model)

        # Output projection to vocabulary (language model head)
        if weight_tying:
            # Share weights dengan token embedding (transpose untuk output projection)
            self.lm_head = self.token_embedding.embedding.T  # [d_model, vocab_size]
        else:
            # Separate weight matrix
            self.lm_head = np.random.randn(d_model, vocab_size) * 0.02

    def forward(self, input_ids, return_attention=False):
        """
        Forward pass untuk GPT model.

        Args:
            input_ids: token indices [batch, seq_len]
            return_attention: whether to return attention weights from all layers

        Returns:
            logits: [batch, seq_len, vocab_size]
            next_token_probs: distribusi probabilitas untuk token berikutnya [batch, vocab_size]
            attention_weights (optional): list of [batch, num_heads, seq_len, seq_len] for each layer
        """
        batch_size, seq_len = input_ids.shape

        # 1. Token Embedding
        x = self.token_embedding.forward(input_ids)  # [batch, seq_len, d_model]

        # 2. Add Positional Encoding
        x = self.positional_encoding.forward(x)  # [batch, seq_len, d_model]

        # 3. Create causal mask
        mask = create_causal_mask(seq_len)  # [seq_len, seq_len]

        # 4. Pass through decoder blocks
        all_attention_weights = []
        for block in self.decoder_blocks:
            if return_attention:
                x, attn_weights = block.forward(x, mask, return_attention=True)
                all_attention_weights.append(attn_weights)
            else:
                x = block.forward(x, mask)  # [batch, seq_len, d_model]

        # 5. Final Layer Normalization
        x = layer_norm(x, self.ln_f_gamma, self.ln_f_beta)  # [batch, seq_len, d_model]

        # 6. Project to vocabulary size (logits)
        logits = np.matmul(x, self.lm_head)  # [batch, seq_len, vocab_size]

        # 7. Get next token prediction (softmax pada posisi terakhir)
        last_token_logits = logits[:, -1, :]  # [batch, vocab_size]
        next_token_probs = softmax(last_token_logits, axis=-1)  # [batch, vocab_size]

        if return_attention:
            return logits, next_token_probs, all_attention_weights
        return logits, next_token_probs

    def count_parameters(self):
        """
        Hitung total jumlah parameters dalam model.

        Returns:
            total_params: total jumlah parameter
            param_breakdown: dictionary dengan breakdown per komponen
        """
        params = {}

        # Token Embedding
        params['token_embedding'] = self.vocab_size * self.d_model

        # Output LM head (jika tidak weight tying)
        if not self.weight_tying:
            params['lm_head'] = self.d_model * self.vocab_size
        else:
            params['lm_head'] = 0  # Shared dengan embedding

        # Decoder blocks
        # Setiap block: MHA (4 * d_model^2) + FFN (2 * d_model * d_ff) + LayerNorm (2 * 2 * d_model)
        d_ff = self.decoder_blocks[0].ffn.d_ff
        params_per_block = (
            4 * self.d_model * self.d_model +  # Q, K, V, O projections
            2 * self.d_model * d_ff +           # FFN W1, W2
            2 * self.d_model +                  # FFN biases (b1, b2)
            4 * self.d_model                    # 2 LayerNorms (gamma, beta)
        )
        params['decoder_blocks'] = self.num_layers * params_per_block

        # Final LayerNorm
        params['final_ln'] = 2 * self.d_model

        # Total
        total_params = sum(params.values())

        return total_params, params


# ==================== HELPER FUNCTION ====================

def generate_next_token(model, input_ids, temperature=1.0):
    """
    Generate token berikutnya menggunakan sampling dari distribusi probabilitas.

    Args:
        model: GPTModel instance
        input_ids: token indices [batch, seq_len]
        temperature: kontrol randomness (default=1.0, lebih tinggi = lebih random)

    Returns:
        next_token: token yang di-sample [batch]
    """
    _, next_token_probs = model.forward(input_ids)

    # Apply temperature
    if temperature != 1.0:
        logits = np.log(next_token_probs + 1e-10) / temperature
        next_token_probs = softmax(logits, axis=-1)

    # Sample dari distribusi
    batch_size = next_token_probs.shape[0]
    next_tokens = []
    for i in range(batch_size):
        next_token = np.random.choice(model.vocab_size, p=next_token_probs[i])
        next_tokens.append(next_token)

    return np.array(next_tokens)
