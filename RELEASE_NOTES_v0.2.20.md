# Release Notes - v0.2.20

**Release Date:** December 30, 2024

## 🎯 NEW: RNN Operations Module

Added complete Recurrent Neural Network operations - the foundation of sequence modeling!

### 📦 What's New

#### Basic RNN (2 functions)
1. **rnn_cell_forward()** - Single timestep RNN
2. **rnn_forward()** - Full sequence RNN

#### LSTM (2 functions)
3. **lstm_cell_forward()** - Single timestep LSTM
4. **lstm_forward()** - Full sequence LSTM

#### GRU (2 functions)
5. **gru_cell_forward()** - Single timestep GRU
6. **gru_forward()** - Full sequence GRU

#### Bidirectional (1 function)
7. **bidirectional_rnn_forward()** - Bidirectional RNN

#### Utilities (3 functions)
8. **sigmoid()** - Sigmoid activation
9. **initialize_rnn_weights()** - Weight initialization
10. **clip_gradients()** - Gradient clipping

#### Aliases (4 shortcuts)
11. **vanilla_rnn**, **lstm**, **gru**, **bidirectional_rnn**

## 💻 Installation

```bash
pip install --upgrade ilovetools
```

## ✅ Quick Start

```python
from ilovetools.ml.rnn import (
    rnn_forward,
    lstm_forward,
    gru_forward,
    bidirectional_rnn_forward,
    initialize_rnn_weights
)
import numpy as np

# Basic RNN
x = np.random.randn(32, 10, 128)  # (batch, seq_len, input_size)
h_0 = np.zeros((32, 256))

weights = initialize_rnn_weights(128, 256, cell_type='rnn')
outputs, hidden_states = rnn_forward(
    x, h_0, weights['W_xh'], weights['W_hh'], weights['b_h']
)
print(f"RNN: {x.shape} -> {outputs.shape}")  # (32, 10, 256)

# LSTM
c_0 = np.zeros((32, 256))
weights_lstm = initialize_rnn_weights(128, 256, cell_type='lstm')
outputs, h_states, c_states = lstm_forward(
    x, h_0, c_0,
    weights_lstm['W_f'], weights_lstm['W_i'], 
    weights_lstm['W_c'], weights_lstm['W_o'],
    weights_lstm['b_f'], weights_lstm['b_i'], 
    weights_lstm['b_c'], weights_lstm['b_o']
)
print(f"LSTM: {x.shape} -> {outputs.shape}")  # (32, 10, 256)

# GRU
weights_gru = initialize_rnn_weights(128, 256, cell_type='gru')
outputs, hidden_states = gru_forward(
    x, h_0,
    weights_gru['W_z'], weights_gru['W_r'], weights_gru['W_h'],
    weights_gru['b_z'], weights_gru['b_r'], weights_gru['b_h']
)
print(f"GRU: {x.shape} -> {outputs.shape}")  # (32, 10, 256)

# Bidirectional RNN
h_0_f = np.zeros((32, 256))
h_0_b = np.zeros((32, 256))
weights_f = initialize_rnn_weights(128, 256, cell_type='rnn')
weights_b = initialize_rnn_weights(128, 256, cell_type='rnn')

outputs, f_states, b_states = bidirectional_rnn_forward(
    x, h_0_f, h_0_b,
    weights_f['W_xh'], weights_f['W_hh'], weights_f['b_h'],
    weights_b['W_xh'], weights_b['W_hh'], weights_b['b_h']
)
print(f"Bidirectional: {x.shape} -> {outputs.shape}")  # (32, 10, 512)
```

## 🔧 Advanced Usage

### Language Modeling

```python
# Character-level language model
vocab_size = 100
hidden_size = 512

# Input: sequence of character indices
x = np.random.randint(0, vocab_size, (32, 50))  # (batch, seq_len)

# Embed characters
embedding = np.random.randn(vocab_size, 128)
x_embedded = embedding[x]  # (32, 50, 128)

# LSTM forward
h_0 = np.zeros((32, hidden_size))
c_0 = np.zeros((32, hidden_size))
weights = initialize_rnn_weights(128, hidden_size, cell_type='lstm')

outputs, _, _ = lstm_forward(x_embedded, h_0, c_0, ...)

# Predict next character
logits = np.dot(outputs, W_out) + b_out  # (32, 50, vocab_size)
```

### Sentiment Analysis

```python
# Bidirectional LSTM for sentiment
x = np.random.randn(32, 20, 300)  # (batch, seq_len, word_dim)

# Bidirectional LSTM
outputs, _, _ = bidirectional_rnn_forward(...)  # (32, 20, 512)

# Take last hidden state
final_hidden = outputs[:, -1, :]  # (32, 512)

# Classify sentiment
sentiment = sigmoid(np.dot(final_hidden, W_class) + b_class)
```

### Time Series Prediction

```python
# Stock price prediction with GRU
x = np.random.randn(32, 60, 5)  # (batch, 60 days, 5 features)

weights = initialize_rnn_weights(5, 128, cell_type='gru')
outputs, _ = gru_forward(x, h_0, ...)

# Predict next day
prediction = np.dot(outputs[:, -1, :], W_pred) + b_pred
```

## 💡 Pro Tips

✅ **Use LSTM for long sequences** - Avoids vanishing gradient  
✅ **Use GRU for faster training** - Fewer parameters  
✅ **Clip gradients** - Prevents exploding gradients  
✅ **Bidirectional for context** - Sees past and future  
✅ **Stack multiple layers** - Deeper = better  
✅ **Add dropout** - Regularization  

❌ **Don't use vanilla RNN for long sequences** - Vanishing gradient  
❌ **Don't forget gradient clipping** - Exploding gradients  
❌ **Don't skip normalization** - Training instability  

## 📊 Comparison

### RNN vs LSTM vs GRU

| Feature | RNN | LSTM | GRU |
|---------|-----|------|-----|
| Gates | 0 | 3 | 2 |
| Parameters | Fewest | Most | Medium |
| Speed | Fastest | Slowest | Fast |
| Long-term memory | ❌ | ✅ | ✅ |
| Use case | Short sequences | Long sequences | Medium sequences |

### Parameter Count

For input_size=128, hidden_size=256:

- **RNN:** 98,560 parameters
- **LSTM:** 394,240 parameters (4x RNN)
- **GRU:** 295,680 parameters (3x RNN)

## 🔗 Links

- **PyPI:** https://pypi.org/project/ilovetools/
- **GitHub:** https://github.com/AliMehdi512/ilovetools
- **RNN Code:** https://github.com/AliMehdi512/ilovetools/blob/main/ilovetools/ml/rnn.py
- **Tests:** https://github.com/AliMehdi512/ilovetools/blob/main/tests/test_rnn.py
- **Verification:** https://github.com/AliMehdi512/ilovetools/blob/main/scripts/verify_rnn.py

## 📈 Total ML Functions

- **Previous (v0.2.19):** 301+ functions
- **New (v0.2.20):** **316+ functions** (15+ new RNN functions!)

## 🎓 Educational Content

Check out our LinkedIn posts:
- **RNNs Guide:** https://www.linkedin.com/feed/update/urn:li:share:7411784681609895937
- **CNNs:** https://www.linkedin.com/feed/update/urn:li:share:7411263336107036672
- **Attention:** https://www.linkedin.com/feed/update/urn:li:share:7410931572444385280

## 📚 Research Papers

- **LSTM:** Hochreiter & Schmidhuber (1997)
- **GRU:** Cho et al. (2014)
- **Bidirectional RNN:** Schuster & Paliwal (1997)

## 🚀 What's Next

Coming in future releases:
- Peephole LSTM
- Layer normalization for RNNs
- Attention-based RNNs
- Encoder-decoder architectures

## 📝 Version History

- **v0.2.20** (Dec 30, 2024): ✅ RNN operations module
- **v0.2.19** (Dec 29, 2024): CNN operations module
- **v0.2.18** (Dec 28, 2024): Attention mechanisms module
- **v0.2.17** (Dec 27, 2024): Normalization techniques module

---

**Process Sequences, Remember Context! 🎯**
