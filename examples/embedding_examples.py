"""
Comprehensive Examples: Embedding Layers

This file demonstrates all embedding layer types with practical examples and use cases.

Author: Ali Mehdi
Date: January 22, 2026
"""

import numpy as np
from ilovetools.ml.embedding import (
    Embedding,
    PositionalEncoding,
    LearnedPositionalEmbedding,
    TokenTypeEmbedding,
    CharacterEmbedding,
    cosine_similarity,
    euclidean_distance,
    most_similar,
)

print("=" * 80)
print("EMBEDDING LAYERS - COMPREHENSIVE EXAMPLES")
print("=" * 80)
print()

# ============================================================================
# EXAMPLE 1: Standard Embedding - Word Embeddings
# ============================================================================
print("EXAMPLE 1: Standard Embedding - Word Embeddings")
print("-" * 80)

vocab_size = 10000
embedding_dim = 300

emb = Embedding(vocab_size=vocab_size, embedding_dim=embedding_dim)

print("Word embedding layer:")
print(f"Vocabulary size: {vocab_size:,}")
print(f"Embedding dimension: {embedding_dim}")
print(f"Embedding matrix shape: {emb.weight.shape}")
print()

# Simulate tokenized text
tokens = np.array([[1, 45, 234, 567, 890, 2]])  # (batch, seq_len)
print(f"Input tokens: {tokens.shape}")
print(f"Token indices: {tokens[0]}")
print()

output = emb.forward(tokens)
print(f"Output embeddings: {output.shape}")
print(f"Each token → {embedding_dim}D vector")

print("\n✓ Example 1 completed\n")

# ============================================================================
# EXAMPLE 2: Positional Encoding - Transformers
# ============================================================================
print("EXAMPLE 2: Positional Encoding - Transformers")
print("-" * 80)

embedding_dim = 512
pos_enc = PositionalEncoding(embedding_dim=embedding_dim, max_len=5000)

print("Sinusoidal positional encoding:")
print(f"Embedding dimension: {embedding_dim}")
print(f"Max sequence length: 5000")
print(f"Formula: PE(pos, 2i) = sin(pos / 10000^(2i/d))")
print(f"         PE(pos, 2i+1) = cos(pos / 10000^(2i/d))")
print()

# Token embeddings
x = np.random.randn(32, 100, 512)
print(f"Token embeddings: {x.shape}")

# Add positional encoding
output = pos_enc.forward(x, training=False)
print(f"With positional encoding: {output.shape}")
print(f"Position information added to each token")

print("\n✓ Example 2 completed\n")

# ============================================================================
# EXAMPLE 3: Semantic Similarity
# ============================================================================
print("EXAMPLE 3: Semantic Similarity with Embeddings")
print("-" * 80)

emb = Embedding(vocab_size=10000, embedding_dim=300)

# Get embeddings for some words (simulated)
king = emb.weight[100]
queen = emb.weight[101]
man = emb.weight[102]
woman = emb.weight[103]

print("Computing semantic similarity:")
print()

# Cosine similarity
sim_king_queen = cosine_similarity(king, queen)
sim_king_man = cosine_similarity(king, man)
sim_queen_woman = cosine_similarity(queen, woman)

print(f"Similarity(king, queen): {sim_king_queen:.4f}")
print(f"Similarity(king, man): {sim_king_man:.4f}")
print(f"Similarity(queen, woman): {sim_queen_woman:.4f}")
print()

# Famous analogy: king - man + woman ≈ queen
analogy_vector = king - man + woman
sim_analogy_queen = cosine_similarity(analogy_vector, queen)
print(f"king - man + woman ≈ queen")
print(f"Similarity: {sim_analogy_queen:.4f}")

print("\n✓ Example 3 completed\n")

# ============================================================================
# EXAMPLE 4: Finding Most Similar Words
# ============================================================================
print("EXAMPLE 4: Finding Most Similar Words")
print("-" * 80)

emb = Embedding(vocab_size=10000, embedding_dim=300)

# Query word
query_idx = 500
query_vector = emb.weight[query_idx]

print(f"Query word index: {query_idx}")
print()

# Find most similar
indices, similarities = most_similar(emb.weight, query_vector, top_k=5, exclude_idx=query_idx)

print("Top 5 most similar words:")
for i, (idx, sim) in enumerate(zip(indices, similarities), 1):
    print(f"{i}. Word {idx}: similarity = {sim:.4f}")

print("\n✓ Example 4 completed\n")

# ============================================================================
# EXAMPLE 5: Loading Pretrained Embeddings
# ============================================================================
print("EXAMPLE 5: Loading Pretrained Embeddings (Word2Vec/GloVe)")
print("-" * 80)

vocab_size = 10000
embedding_dim = 300

emb = Embedding(vocab_size=vocab_size, embedding_dim=embedding_dim)

print("Loading pretrained embeddings:")
print(f"Vocabulary: {vocab_size:,} words")
print(f"Dimension: {embedding_dim}")
print()

# Simulate pretrained embeddings (e.g., from Word2Vec or GloVe)
pretrained = np.random.randn(vocab_size, embedding_dim)

emb.load_pretrained(pretrained)
print("✓ Pretrained embeddings loaded")
print()

# Freeze embeddings (don't update during training)
emb.freeze()
print("✓ Embeddings frozen (won't be updated during training)")

print("\n✓ Example 5 completed\n")

# ============================================================================
# EXAMPLE 6: BERT-Style Embeddings
# ============================================================================
print("EXAMPLE 6: BERT-Style Embeddings (Token + Position + Segment)")
print("-" * 80)

vocab_size = 30000
embedding_dim = 768
max_len = 512

# Token embeddings
token_emb = Embedding(vocab_size=vocab_size, embedding_dim=embedding_dim)

# Position embeddings (learned)
pos_emb = LearnedPositionalEmbedding(max_len=max_len, embedding_dim=embedding_dim)

# Token type embeddings (segment A vs B)
token_type_emb = TokenTypeEmbedding(num_types=2, embedding_dim=embedding_dim)

print("BERT embedding components:")
print(f"1. Token embeddings: {vocab_size:,} vocab, {embedding_dim}D")
print(f"2. Position embeddings: {max_len} max length, {embedding_dim}D")
print(f"3. Token type embeddings: 2 types, {embedding_dim}D")
print()

# Input: [CLS] sentence A [SEP] sentence B [SEP]
tokens = np.array([[101, 2023, 2003, 102, 2054, 2003, 102]])
token_type_ids = np.array([[0, 0, 0, 0, 1, 1, 1]])

print(f"Input tokens: {tokens.shape}")
print(f"Token type IDs: {token_type_ids[0]}")
print()

# Compute embeddings
token_embeddings = token_emb.forward(tokens)
print(f"Token embeddings: {token_embeddings.shape}")

position_embeddings = pos_emb.forward(token_embeddings)
print(f"+ Position embeddings")

token_type_embeddings = token_type_emb.forward(token_type_ids)
print(f"+ Token type embeddings")

# Combine
final_embeddings = token_embeddings + position_embeddings + token_type_embeddings
print(f"\nFinal BERT embeddings: {final_embeddings.shape}")

print("\n✓ Example 6 completed\n")

# ============================================================================
# EXAMPLE 7: Character-Level Embeddings
# ============================================================================
print("EXAMPLE 7: Character-Level Embeddings")
print("-" * 80)

num_chars = 128  # ASCII characters
char_embedding_dim = 50

char_emb = CharacterEmbedding(num_chars=num_chars, embedding_dim=char_embedding_dim)

print("Character-level embeddings:")
print(f"Number of characters: {num_chars}")
print(f"Character embedding dim: {char_embedding_dim}")
print()

# Simulate character IDs for words
# Shape: (batch, num_words, max_word_len)
char_ids = np.random.randint(0, num_chars, (32, 20, 15))

print(f"Input: {char_ids.shape}")
print(f"Batch: 32 sentences")
print(f"Words per sentence: 20")
print(f"Characters per word: 15")
print()

output = char_emb.forward(char_ids)
print(f"Word embeddings from characters: {output.shape}")
print(f"Each word → {char_embedding_dim}D vector")

print("\n✓ Example 7 completed\n")

# ============================================================================
# EXAMPLE 8: Transformer Encoder with Embeddings
# ============================================================================
print("EXAMPLE 8: Transformer Encoder with Embeddings")
print("-" * 80)

vocab_size = 10000
embedding_dim = 512
max_len = 1000

# Token embeddings
token_emb = Embedding(vocab_size=vocab_size, embedding_dim=embedding_dim)

# Positional encoding
pos_enc = PositionalEncoding(embedding_dim=embedding_dim, max_len=max_len, dropout=0.1)

print("Transformer encoder input:")
print(f"Vocabulary: {vocab_size:,}")
print(f"Embedding dim: {embedding_dim}")
print()

# Input tokens
tokens = np.array([[1, 45, 234, 567, 890, 2, 3, 4, 5, 6]])
print(f"Input tokens: {tokens.shape}")

# Embed tokens
embedded = token_emb.forward(tokens)
print(f"Token embeddings: {embedded.shape}")

# Add positional encoding
encoder_input = pos_enc.forward(embedded, training=True)
print(f"+ Positional encoding: {encoder_input.shape}")
print()

print("Ready for Transformer encoder layers!")

print("\n✓ Example 8 completed\n")

# ============================================================================
# EXAMPLE 9: Embedding Dimension Comparison
# ============================================================================
print("EXAMPLE 9: Embedding Dimension Comparison")
print("-" * 80)

vocab_size = 10000
dimensions = [50, 100, 200, 300, 512, 768]

print(f"Vocabulary size: {vocab_size:,}")
print()

print("Embedding dimension comparison:")
for dim in dimensions:
    emb = Embedding(vocab_size=vocab_size, embedding_dim=dim)
    params = emb.weight.size
    memory_mb = params * 4 / (1024 * 1024)  # 4 bytes per float32
    
    print(f"Dim {dim:3d}: {params:,} params, {memory_mb:.2f} MB")

print()
print("Trade-off:")
print("✓ Higher dim: More expressive, better semantic capture")
print("✗ Higher dim: More parameters, more memory, slower")

print("\n✓ Example 9 completed\n")

# ============================================================================
# EXAMPLE 10: Padding and Unknown Tokens
# ============================================================================
print("EXAMPLE 10: Padding and Unknown Tokens")
print("-" * 80)

vocab_size = 10000
embedding_dim = 300
padding_idx = 0
unk_idx = 1

emb = Embedding(vocab_size=vocab_size, embedding_dim=embedding_dim, padding_idx=padding_idx)

print("Special tokens:")
print(f"Padding token: index {padding_idx}")
print(f"Unknown token: index {unk_idx}")
print()

# Check padding embedding is zeros
print(f"Padding embedding norm: {np.linalg.norm(emb.weight[padding_idx]):.6f}")
print("(Should be 0.0)")
print()

# Simulate padded sequence
tokens = np.array([
    [5, 10, 15, 20, 0, 0, 0],  # Sentence 1 (padded)
    [3, 7, 11, 13, 17, 19, 23]  # Sentence 2 (no padding)
])

print(f"Input tokens: {tokens.shape}")
print(f"Sentence 1: {tokens[0]} (3 padding tokens)")
print(f"Sentence 2: {tokens[1]} (no padding)")
print()

output = emb.forward(tokens)
print(f"Output embeddings: {output.shape}")

print("\n✓ Example 10 completed\n")

# ============================================================================
# EXAMPLE 11: Embedding Space Visualization (Conceptual)
# ============================================================================
print("EXAMPLE 11: Embedding Space Visualization (Conceptual)")
print("-" * 80)

emb = Embedding(vocab_size=10000, embedding_dim=300)

print("Embedding space properties:")
print()

# Simulate word clusters
animals = [100, 101, 102, 103, 104]  # dog, cat, bird, fish, etc.
colors = [200, 201, 202, 203, 204]   # red, blue, green, yellow, etc.

print("Word clusters in embedding space:")
print()

# Animals cluster
print("Animals cluster:")
for i in range(len(animals) - 1):
    sim = cosine_similarity(emb.weight[animals[i]], emb.weight[animals[i+1]])
    print(f"  Word {animals[i]} ↔ Word {animals[i+1]}: {sim:.4f}")

print()

# Colors cluster
print("Colors cluster:")
for i in range(len(colors) - 1):
    sim = cosine_similarity(emb.weight[colors[i]], emb.weight[colors[i+1]])
    print(f"  Word {colors[i]} ↔ Word {colors[i+1]}: {sim:.4f}")

print()

# Cross-cluster similarity (should be lower)
cross_sim = cosine_similarity(emb.weight[animals[0]], emb.weight[colors[0]])
print(f"Cross-cluster similarity: {cross_sim:.4f}")
print("(Should be lower than within-cluster)")

print("\n✓ Example 11 completed\n")

# ============================================================================
# EXAMPLE 12: Embedding Layer Selection Guide
# ============================================================================
print("EXAMPLE 12: Embedding Layer Selection Guide")
print("-" * 80)

print("When to use each embedding type:")
print()

print("Standard Embedding:")
print("  ✓ Word-level NLP tasks")
print("  ✓ Text classification, sentiment analysis")
print("  ✓ Can load pretrained (Word2Vec, GloVe)")
print("  ✓ Most common choice")
print()

print("Positional Encoding (Sinusoidal):")
print("  ✓ Transformers (original paper)")
print("  ✓ No learnable parameters")
print("  ✓ Can extrapolate to longer sequences")
print("  ✓ Deterministic")
print()

print("Learned Positional Embedding:")
print("  ✓ BERT-style models")
print("  ✓ Learnable from data")
print("  ✓ Task-specific position info")
print("  ✗ Fixed max length")
print()

print("Token Type Embedding:")
print("  ✓ BERT sentence pairs")
print("  ✓ Multi-segment inputs")
print("  ✓ Question-answering tasks")
print()

print("Character Embedding:")
print("  ✓ Handling rare/OOV words")
print("  ✓ Morphologically rich languages")
print("  ✓ Spelling correction")
print("  ✓ Named entity recognition")

print("\n✓ Example 12 completed\n")

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY! ✓")
print("=" * 80)
print()
print("Summary of what we covered:")
print("1. ✓ Standard Embedding - Word Embeddings")
print("2. ✓ Positional Encoding - Transformers")
print("3. ✓ Semantic Similarity")
print("4. ✓ Finding Most Similar Words")
print("5. ✓ Loading Pretrained Embeddings")
print("6. ✓ BERT-Style Embeddings")
print("7. ✓ Character-Level Embeddings")
print("8. ✓ Transformer Encoder with Embeddings")
print("9. ✓ Embedding Dimension Comparison")
print("10. ✓ Padding and Unknown Tokens")
print("11. ✓ Embedding Space Visualization")
print("12. ✓ Embedding Layer Selection Guide")
print()
print("You now have a complete understanding of embedding layers!")
print()
print("Next steps:")
print("- Use pretrained embeddings (Word2Vec, GloVe)")
print("- Combine token + position + segment embeddings")
print("- Choose dimension based on task complexity")
print("- Handle padding and unknown tokens properly")
print("- Explore semantic relationships in embedding space")
