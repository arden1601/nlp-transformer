# nlp-tranformer
# Pemrosesan Bahasa Alami 2025
Cornelius Arden Satwika Hermawan
22/505482/TK/55313


## 🎯 Fitur Utama

Implementasi ini mencakup semua komponen penting dari transformer decoder:

### Core Components
- ✅ **Token Embedding** - Embedding matrix untuk mengkonversi token ID ke vektor
- ✅ **Positional Encoding** - Sinusoidal positional encoding untuk informasi posisi
- ✅ **Scaled Dot-Product Attention** - Attention mechanism dengan softmax dan scaling
- ✅ **Multi-Head Attention** - Parallel attention heads dengan Q, K, V projections
- ✅ **Feed-Forward Network** - 2-layer FFN dengan aktivasi GELU
- ✅ **Layer Normalization** - Pre-norm architecture dengan residual connections
- ✅ **Causal Masking** - Mencegah attention ke token masa depan
- ✅ **Output Layer** - Proyeksi ke vocab size dan softmax untuk prediksi

### Advanced Features
- ✅ **Weight Tying** - Share embedding dan output weights untuk efisiensi parameter
- ✅ **Attention Visualization** - Visualisasi attention patterns untuk interpretability
- ✅ **Parameter Counting** - Hitung total parameters

## 🚀 Cara Menggunakan

### 1. Instalasi Dependensi

Hanya membutuhkan NumPy:

```bash
pip install numpy
```

### 2. Menjalankan Demo

**Demo Dasar:**
```bash
python demo.py
```

Demo akan menampilkan:
- Forward pass dengan berbagai ukuran batch
- Autoregressive text generation
- Visualisasi causal masking
- Testing individual components

**Demo Weight Tying & Attention Visualization:**
```bash
python demo_weight_tying_attention.py
```

Demo advanced features:
- Perbandingan model dengan/tanpa weight tying
- Parameter count comparison
- Attention pattern visualization
- Multi-head attention analysis
- Layer-wise attention evolution

**Contoh Penggunaan Sederhana:**
```bash
python example_usage.py
```

### 3. Menggunakan Model

```python
import numpy as np
from transformer import GPTModel

# Inisialisasi model
model = GPTModel(
    vocab_size=1000,      # Ukuran vocabulary
    d_model=128,          # Dimensi model
    num_heads=8,          # Jumlah attention heads
    d_ff=512,             # Dimensi feed-forward (biasanya 4 * d_model)
    num_layers=4,         # Jumlah decoder blocks
    max_seq_len=100       # Panjang sequence maksimum
)

# Input: batch of token sequences
input_ids = np.array([[1, 5, 10, 42, 7]])  # [batch_size, seq_len]

# Forward pass
logits, next_token_probs = model.forward(input_ids)

# Output:
# - logits: [batch_size, seq_len, vocab_size]
# - next_token_probs: [batch_size, vocab_size] (softmax pada posisi terakhir)

print(f"Logits shape: {logits.shape}")
print(f"Next token probs shape: {next_token_probs.shape}")
print(f"Top 5 next tokens: {np.argsort(next_token_probs[0])[-5:][::-1]}")
```

### 4. Autoregressive Generation

```python
from transformer import GPTModel, generate_next_token

model = GPTModel(vocab_size=1000, d_model=128, num_heads=8,
                 d_ff=512, num_layers=4, max_seq_len=100)

# Start dengan satu token
input_ids = np.array([[42]])  # Start token

# Generate 10 token berikutnya
for i in range(10):
    next_token = generate_next_token(model, input_ids, temperature=0.8)
    input_ids = np.concatenate([input_ids, next_token.reshape(1, 1)], axis=1)

print(f"Generated sequence: {input_ids}")
```

### 5. Weight Tying

Weight tying mengurangi parameter dengan share embedding dan output weights:

```python
# Model TANPA weight tying
model_without = GPTModel(
    vocab_size=1000, d_model=128, num_heads=8,
    d_ff=512, num_layers=4, max_seq_len=100,
    weight_tying=False  # Default
)

# Model DENGAN weight tying
model_with = GPTModel(
    vocab_size=1000, d_model=128, num_heads=8,
    d_ff=512, num_layers=4, max_seq_len=100,
    weight_tying=True  # Enable weight tying
)

# Compare parameter counts
total_1, _ = model_without.count_parameters()
total_2, _ = model_with.count_parameters()

print(f"Without tying: {total_1:,} params")
print(f"With tying: {total_2:,} params")
print(f"Reduction: {(total_1-total_2)/total_1*100:.1f}%")

### 6. Attention Visualization

Visualisasi attention patterns untuk interpretability:

```python
from visualize_attention import visualize_attention_ascii

# Forward with attention tracking
logits, probs, attention_weights = model.forward(
    input_ids,
    return_attention=True  # Enable attention tracking
)

# Visualize attention pattern
visualize_attention_ascii(
    attention_weights,
    layer_idx=0,  # Which layer
    head_idx=0,   # Which head
    batch_idx=0,
    max_tokens=8
)
```

**Output:**
```
Attention Pattern - Layer 0, Head 0
        t0   t1   t2   t3   t4
 t0     █    ·    ·    ·    ·
 t1     █    █    ·    ·    ·
 t2     ▓    ▓    ▓    ·    ·
 t3     ▒    ▒    ▒    ▒    ·
 t4     ▒    ▒    ▒    ▒    ▒

Legend: · (weak) ░ ▒ ▓ █ (strong)
```

## 🏗️ Arsitektur Model

### Forward Pass Flow

```
Input Token IDs [batch, seq_len]
    ↓
Token Embedding [batch, seq_len, d_model]
    ↓
+ Positional Encoding (Sinusoidal)
    ↓
Decoder Block 1
    ├─ Layer Norm (Pre-Norm)
    ├─ Multi-Head Attention (dengan Causal Mask)
    ├─ + Residual Connection
    ├─ Layer Norm
    ├─ Feed-Forward Network
    └─ + Residual Connection
    ↓
Decoder Block 2, 3, ... N (sama seperti di atas)
    ↓
Final Layer Norm
    ↓
Linear Projection ke Vocab Size [batch, seq_len, vocab_size]
    ↓
Softmax (pada posisi terakhir) → [batch, vocab_size]
```

### Causal Masking

Model menggunakan causal masking untuk memastikan setiap token hanya bisa attend ke token sebelumnya:

```
     t0    t1    t2    t3    t4
t0   ✓     ✗     ✗     ✗     ✗
t1   ✓     ✓     ✗     ✗     ✗
t2   ✓     ✓     ✓     ✗     ✗
t3   ✓     ✓     ✓     ✓     ✗
t4   ✓     ✓     ✓     ✓     ✓

✓ = dapat diakses (attend)
✗ = di-mask (tidak dapat diakses)
```

## 📊 Komponen Detail

### 1. Token Embedding

Mengkonversi token ID ke vektor berdimensi `d_model`:

```python
embedding = TokenEmbedding(vocab_size=1000, d_model=128)
embeddings = embedding.forward(token_ids)  # [batch, seq_len, d_model]
```

### 2. Positional Encoding (Sinusoidal)

Menambahkan informasi posisi menggunakan fungsi sinus dan kosinus:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### 3. Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Dengan causal masking untuk mencegah akses ke token masa depan.

### 4. Multi-Head Attention

- Split input menjadi `num_heads` heads
- Setiap head melakukan scaled dot-product attention
- Concatenate hasil dari semua heads
- Linear projection untuk output final

### 5. Feed-Forward Network

```
FFN(x) = W2 * GELU(W1 * x + b1) + b2
```

Dua layer linear dengan aktivasi GELU di antaranya.

### 6. Layer Normalization (Pre-Norm)

```
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

Pre-norm architecture (normalisasi sebelum sub-layer).

## 🧪 Output Demo

Ketika menjalankan `demo.py`:

```
============================================================
 DEMO 1: Basic Forward Pass
============================================================

Model Configuration:
  - Vocab Size: 1000
  - d_model: 128
  - Num Heads: 8
  - d_ff: 512
  - Num Layers: 4
  - Max Seq Len: 100

✓ Forward pass completed!

Output Shapes:
    - Logits: (2, 10, 1000) (batch, seq_len, vocab_size)
    - Next Token Probs: (2, 1000) (batch, vocab_size)

Output Analysis:
    - Next token probs sum: [1. 1.] ✓
    - Top-5 most likely next tokens: [225, 679, 508, 250, 983]
```

