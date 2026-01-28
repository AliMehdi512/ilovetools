"""
Comprehensive Examples: Recurrent Layers

This file demonstrates all recurrent layer types with practical examples and use cases.

Author: Ali Mehdi
Date: January 22, 2026
"""

import numpy as np
from ilovetools.ml.recurrent import (
    RNN,
    LSTM,
    GRU,
    BiLSTM,
    BiGRU,
)

print("=" * 80)
print("RECURRENT LAYERS - COMPREHENSIVE EXAMPLES")
print("=" * 80)
print()

# ============================================================================
# EXAMPLE 1: Vanilla RNN - Simple Sequence Modeling
# ============================================================================
print("EXAMPLE 1: Vanilla RNN - Simple Sequence Modeling")
print("-" * 80)

rnn = RNN(input_size=128, hidden_size=256)

print("Vanilla RNN for short sequences:")
print(f"Input size: {rnn.input_size}")
print(f"Hidden size: {rnn.hidden_size}")
print()

# Simulate text embeddings
x = np.random.randn(32, 10, 128)  # (batch, seq_len, input_size)
print(f"Input shape: {x.shape}")
print(f"Batch: 32 sentences")
print(f"Sequence length: 10 words")
print(f"Embedding dim: 128")
print()

output, hidden = rnn.forward(x)
print(f"Output shape: {output.shape}")
print(f"Hidden state shape: {hidden.shape}")
print(f"Output contains hidden states for all time steps")

print("\n✓ Example 1 completed\n")

# ============================================================================
# EXAMPLE 2: LSTM - Long-Term Dependencies
# ============================================================================
print("EXAMPLE 2: LSTM - Long-Term Dependencies")
print("-" * 80)

lstm = LSTM(input_size=128, hidden_size=256)

print("LSTM for long sequences:")
print(f"Input size: {lstm.input_size}")
print(f"Hidden size: {lstm.hidden_size}")
print(f"Gates: Forget, Input, Output")
print(f"Cell state: Long-term memory")
print()

# Simulate long text sequence
x = np.random.randn(32, 100, 128)  # Long sequence
print(f"Input shape: {x.shape}")
print(f"Sequence length: 100 words (long)")
print()

output, (hidden, cell) = lstm.forward(x)
print(f"Output shape: {output.shape}")
print(f"Hidden state shape: {hidden.shape}")
print(f"Cell state shape: {cell.shape}")
print(f"LSTM maintains long-term dependencies via cell state")

print("\n✓ Example 2 completed\n")

# ============================================================================
# EXAMPLE 3: GRU - Efficient Alternative
# ============================================================================
print("EXAMPLE 3: GRU - Efficient Alternative to LSTM")
print("-" * 80)

gru = GRU(input_size=128, hidden_size=256)

print("GRU for efficient sequence modeling:")
print(f"Input size: {gru.input_size}")
print(f"Hidden size: {gru.hidden_size}")
print(f"Gates: Update, Reset")
print(f"Fewer parameters than LSTM")
print()

x = np.random.randn(32, 100, 128)
print(f"Input shape: {x.shape}")

output, hidden = gru.forward(x)
print(f"Output shape: {output.shape}")
print(f"Hidden state shape: {hidden.shape}")

# Parameter comparison
lstm_params = (lstm.W_f.size + lstm.W_i.size + lstm.W_o.size + lstm.W_c.size +
               lstm.b_f.size + lstm.b_i.size + lstm.b_o.size + lstm.b_c.size)
gru_params = (gru.W_z.size + gru.W_r.size + gru.W_h.size +
              gru.b_z.size + gru.b_r.size + gru.b_h.size)

print(f"\nParameter comparison:")
print(f"LSTM parameters: {lstm_params:,}")
print(f"GRU parameters: {gru_params:,}")
print(f"Reduction: {lstm_params / gru_params:.2f}x")

print("\n✓ Example 3 completed\n")

# ============================================================================
# EXAMPLE 4: BiLSTM - Bidirectional Context
# ============================================================================
print("EXAMPLE 4: BiLSTM - Bidirectional Context")
print("-" * 80)

bilstm = BiLSTM(input_size=128, hidden_size=256)

print("Bidirectional LSTM:")
print(f"Input size: {bilstm.input_size}")
print(f"Hidden size per direction: {bilstm.hidden_size}")
print(f"Output size: {2 * bilstm.hidden_size} (concatenated)")
print()

x = np.random.randn(32, 100, 128)
print(f"Input shape: {x.shape}")

output, ((h_fwd, c_fwd), (h_bwd, c_bwd)) = bilstm.forward(x)
print(f"Output shape: {output.shape}")
print(f"Forward hidden: {h_fwd.shape}")
print(f"Backward hidden: {h_bwd.shape}")
print(f"Processes sequence in both directions")
print(f"Richer context for each position")

print("\n✓ Example 4 completed\n")

# ============================================================================
# EXAMPLE 5: BiGRU - Efficient Bidirectional
# ============================================================================
print("EXAMPLE 5: BiGRU - Efficient Bidirectional Processing")
print("-" * 80)

bigru = BiGRU(input_size=128, hidden_size=256)

print("Bidirectional GRU:")
print(f"Input size: {bigru.input_size}")
print(f"Hidden size per direction: {bigru.hidden_size}")
print(f"Output size: {2 * bigru.hidden_size}")
print()

x = np.random.randn(32, 100, 128)
output, (h_fwd, h_bwd) = bigru.forward(x)

print(f"Output shape: {output.shape}")
print(f"Forward hidden: {h_fwd.shape}")
print(f"Backward hidden: {h_bwd.shape}")
print(f"Faster than BiLSTM, similar performance")

print("\n✓ Example 5 completed\n")

# ============================================================================
# EXAMPLE 6: Text Classification with LSTM
# ============================================================================
print("EXAMPLE 6: Text Classification with LSTM")
print("-" * 80)

print("Sentiment analysis pipeline:")
print()

# Simulate text data
vocab_size = 10000
embedding_dim = 128
hidden_size = 256
num_classes = 3  # Positive, Negative, Neutral

# Text to embeddings (simulated)
batch_size = 32
seq_len = 50
embeddings = np.random.randn(batch_size, seq_len, embedding_dim)

print(f"Input: {batch_size} reviews")
print(f"Sequence length: {seq_len} words")
print(f"Embedding dim: {embedding_dim}")
print()

# LSTM layer
lstm = LSTM(input_size=embedding_dim, hidden_size=hidden_size)
output, (hidden, cell) = lstm.forward(embeddings)

print(f"LSTM output: {output.shape}")
print(f"Final hidden state: {hidden.shape}")
print()

# Use final hidden state for classification
# (In practice, would add a linear layer here)
print(f"Classification:")
print(f"Use final hidden state ({hidden.shape}) → Linear({hidden_size}, {num_classes})")
print(f"Output: Sentiment probabilities")

print("\n✓ Example 6 completed\n")

# ============================================================================
# EXAMPLE 7: Named Entity Recognition with BiLSTM
# ============================================================================
print("EXAMPLE 7: Named Entity Recognition with BiLSTM")
print("-" * 80)

print("NER pipeline:")
print()

# Simulate token embeddings
batch_size = 16
seq_len = 100
embedding_dim = 300
hidden_size = 256
num_tags = 9  # B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG, B-MISC, I-MISC, O

embeddings = np.random.randn(batch_size, seq_len, embedding_dim)

print(f"Input: {batch_size} sentences")
print(f"Sequence length: {seq_len} tokens")
print(f"Embedding dim: {embedding_dim} (GloVe/Word2Vec)")
print()

# BiLSTM layer
bilstm = BiLSTM(input_size=embedding_dim, hidden_size=hidden_size)
output, _ = bilstm.forward(embeddings)

print(f"BiLSTM output: {output.shape}")
print(f"Output size: {2 * hidden_size} (bidirectional)")
print()

print(f"Tag prediction:")
print(f"For each token: BiLSTM output → Linear({2*hidden_size}, {num_tags})")
print(f"Tags: B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG, B-MISC, I-MISC, O")

print("\n✓ Example 7 completed\n")

# ============================================================================
# EXAMPLE 8: Time Series Forecasting with GRU
# ============================================================================
print("EXAMPLE 8: Time Series Forecasting with GRU")
print("-" * 80)

print("Stock price prediction:")
print()

# Simulate time series data
batch_size = 64
lookback = 60  # 60 days
num_features = 5  # Open, High, Low, Close, Volume
hidden_size = 128

time_series = np.random.randn(batch_size, lookback, num_features)

print(f"Input: {batch_size} stocks")
print(f"Lookback: {lookback} days")
print(f"Features: {num_features} (OHLCV)")
print()

# GRU layer
gru = GRU(input_size=num_features, hidden_size=hidden_size)
output, hidden = gru.forward(time_series)

print(f"GRU output: {output.shape}")
print(f"Final hidden state: {hidden.shape}")
print()

print(f"Forecasting:")
print(f"Use final hidden state → Linear({hidden_size}, 1)")
print(f"Output: Next day's closing price")

print("\n✓ Example 8 completed\n")

# ============================================================================
# EXAMPLE 9: Sequence-to-Sequence with LSTM (Encoder)
# ============================================================================
print("EXAMPLE 9: Sequence-to-Sequence Encoder (Machine Translation)")
print("-" * 80)

print("Encoder for machine translation:")
print()

# Simulate source sentence
batch_size = 32
src_len = 20
src_vocab_size = 10000
embedding_dim = 256
hidden_size = 512

src_embeddings = np.random.randn(batch_size, src_len, embedding_dim)

print(f"Source language: English")
print(f"Batch: {batch_size} sentences")
print(f"Length: {src_len} words")
print(f"Embedding: {embedding_dim}")
print()

# Encoder LSTM
encoder = LSTM(input_size=embedding_dim, hidden_size=hidden_size)
encoder_output, (encoder_hidden, encoder_cell) = encoder.forward(src_embeddings)

print(f"Encoder output: {encoder_output.shape}")
print(f"Encoder hidden: {encoder_hidden.shape}")
print(f"Encoder cell: {encoder_cell.shape}")
print()

print(f"Decoder initialization:")
print(f"Use encoder's final (hidden, cell) as decoder's initial state")
print(f"Decoder generates target language (e.g., French)")

print("\n✓ Example 9 completed\n")

# ============================================================================
# EXAMPLE 10: Speech Recognition with BiLSTM
# ============================================================================
print("EXAMPLE 10: Speech Recognition with BiLSTM")
print("-" * 80)

print("Audio to text pipeline:")
print()

# Simulate audio features (MFCCs)
batch_size = 16
time_steps = 200  # Audio frames
num_mfcc = 40  # MFCC features
hidden_size = 256

audio_features = np.random.randn(batch_size, time_steps, num_mfcc)

print(f"Input: {batch_size} audio clips")
print(f"Time steps: {time_steps} frames")
print(f"Features: {num_mfcc} MFCCs")
print()

# BiLSTM layer
bilstm = BiLSTM(input_size=num_mfcc, hidden_size=hidden_size)
output, _ = bilstm.forward(audio_features)

print(f"BiLSTM output: {output.shape}")
print(f"Output size: {2 * hidden_size}")
print()

print(f"Character/phoneme prediction:")
print(f"For each frame: BiLSTM output → Linear({2*hidden_size}, num_chars)")
print(f"Use CTC loss for alignment")

print("\n✓ Example 10 completed\n")

# ============================================================================
# EXAMPLE 11: Comparing RNN, LSTM, GRU
# ============================================================================
print("EXAMPLE 11: Comparing RNN, LSTM, GRU")
print("-" * 80)

print("Performance comparison on long sequences:")
print()

# Test on long sequence
batch_size = 8
seq_len = 200  # Long sequence
input_size = 64
hidden_size = 128

x = np.random.randn(batch_size, seq_len, input_size)

# RNN
rnn = RNN(input_size, hidden_size)
out_rnn, _ = rnn.forward(x)

# LSTM
lstm = LSTM(input_size, hidden_size)
out_lstm, _ = lstm.forward(x)

# GRU
gru = GRU(input_size, hidden_size)
out_gru, _ = gru.forward(x)

print(f"Sequence length: {seq_len} (long)")
print()

print(f"RNN output: {out_rnn.shape}")
print(f"LSTM output: {out_lstm.shape}")
print(f"GRU output: {out_gru.shape}")
print()

print("Observations:")
print("✓ RNN: Fastest, but vanishing gradients on long sequences")
print("✓ LSTM: Best for long-term dependencies, more parameters")
print("✓ GRU: Good balance, fewer parameters than LSTM")

print("\n✓ Example 11 completed\n")

# ============================================================================
# EXAMPLE 12: Stacking Recurrent Layers
# ============================================================================
print("EXAMPLE 12: Stacking Recurrent Layers (Deep RNN)")
print("-" * 80)

print("Multi-layer LSTM:")
print()

# Layer 1
lstm1 = LSTM(input_size=128, hidden_size=256)
# Layer 2
lstm2 = LSTM(input_size=256, hidden_size=256)
# Layer 3
lstm3 = LSTM(input_size=256, hidden_size=128)

x = np.random.randn(32, 50, 128)
print(f"Input: {x.shape}")

# Forward through layers
out1, _ = lstm1.forward(x)
print(f"Layer 1 output: {out1.shape}")

out2, _ = lstm2.forward(out1)
print(f"Layer 2 output: {out2.shape}")

out3, _ = lstm3.forward(out2)
print(f"Layer 3 output: {out3.shape}")

print()
print("Deep RNN architecture:")
print("Input (128) → LSTM(256) → LSTM(256) → LSTM(128) → Output")
print("Deeper networks learn more complex patterns")

print("\n✓ Example 12 completed\n")

# ============================================================================
# EXAMPLE 13: Recurrent Layer Selection Guide
# ============================================================================
print("EXAMPLE 13: Recurrent Layer Selection Guide")
print("-" * 80)

print("When to use each recurrent layer:")
print()

print("RNN (Vanilla):")
print("  ✓ Short sequences (< 20 time steps)")
print("  ✓ Simple patterns")
print("  ✓ Baseline model")
print("  ✓ Fast training")
print("  ✗ Vanishing gradients on long sequences")
print()

print("LSTM:")
print("  ✓ Long sequences (100+ time steps)")
print("  ✓ Long-term dependencies")
print("  ✓ NLP tasks (translation, QA)")
print("  ✓ Time series with trends")
print("  ✓ Best accuracy on complex tasks")
print("  ✗ More parameters, slower training")
print()

print("GRU:")
print("  ✓ Long sequences (efficient)")
print("  ✓ Good balance accuracy/speed")
print("  ✓ Fewer parameters than LSTM")
print("  ✓ Faster training than LSTM")
print("  ✓ Similar performance to LSTM")
print("  ✗ Slightly less capacity than LSTM")
print()

print("BiLSTM:")
print("  ✓ NER, POS tagging")
print("  ✓ Sentiment analysis")
print("  ✓ When context from both directions helps")
print("  ✓ Classification tasks")
print("  ✗ 2x parameters, can't use for generation")
print()

print("BiGRU:")
print("  ✓ Efficient bidirectional processing")
print("  ✓ Similar to BiLSTM but faster")
print("  ✓ Good for tagging tasks")
print("  ✗ Can't use for generation")

print("\n✓ Example 14 completed\n")

# ============================================================================
# EXAMPLE 14: Parameter Comparison
# ============================================================================
print("EXAMPLE 14: Parameter Comparison")
print("-" * 80)

input_size, hidden_size = 128, 256

print(f"Input size: {input_size}")
print(f"Hidden size: {hidden_size}")
print()

# RNN
rnn = RNN(input_size, hidden_size)
rnn_params = rnn.W_xh.size + rnn.W_hh.size + rnn.b_h.size

# LSTM
lstm = LSTM(input_size, hidden_size)
lstm_params = (lstm.W_f.size + lstm.W_i.size + lstm.W_o.size + lstm.W_c.size +
               lstm.b_f.size + lstm.b_i.size + lstm.b_o.size + lstm.b_c.size)

# GRU
gru = GRU(input_size, hidden_size)
gru_params = (gru.W_z.size + gru.W_r.size + gru.W_h.size +
              gru.b_z.size + gru.b_r.size + gru.b_h.size)

print(f"RNN parameters: {rnn_params:,}")
print(f"LSTM parameters: {lstm_params:,}")
print(f"GRU parameters: {gru_params:,}")
print()

print("Ratios:")
print(f"LSTM vs RNN: {lstm_params / rnn_params:.2f}x")
print(f"GRU vs RNN: {gru_params / rnn_params:.2f}x")
print(f"LSTM vs GRU: {lstm_params / gru_params:.2f}x")

print("\n✓ Example 14 completed\n")

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY! ✓")
print("=" * 80)
print()
print("Summary of what we covered:")
print("1. ✓ Vanilla RNN - Simple Sequence Modeling")
print("2. ✓ LSTM - Long-Term Dependencies")
print("3. ✓ GRU - Efficient Alternative")
print("4. ✓ BiLSTM - Bidirectional Context")
print("5. ✓ BiGRU - Efficient Bidirectional")
print("6. ✓ Text Classification with LSTM")
print("7. ✓ Named Entity Recognition with BiLSTM")
print("8. ✓ Time Series Forecasting with GRU")
print("9. ✓ Sequence-to-Sequence Encoder")
print("10. ✓ Speech Recognition with BiLSTM")
print("11. ✓ Comparing RNN, LSTM, GRU")
print("12. ✓ Stacking Recurrent Layers")
print("13. ✓ Recurrent Layer Selection Guide")
print("14. ✓ Parameter Comparison")
print()
print("You now have a complete understanding of recurrent layers!")
print()
print("Next steps:")
print("- Use LSTM for long sequences")
print("- Use GRU for efficiency")
print("- Use BiLSTM/BiGRU for tagging tasks")
print("- Stack layers for complex patterns")
print("- Choose based on your task requirements")
