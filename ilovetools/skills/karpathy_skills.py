"""
Andrej Karpathy-inspired LLM/agent coding skills.

This module packages the four canonical Karpathy coding-agent guidelines
(Think Before Coding, Simplicity First, Surgical Changes, Goal-Driven
Execution) as readable Python string constants that LLMs can load at
runtime and use as system-prompt instructions.

The guidelines are derived from the widely-adopted
``andrej-karpathy-skills`` community repositories on GitHub
(forrestchang/andrej-karpathy-skills, multica-ai/andrej-karpathy-skills,
duolahypercho/andrej-karpathy-skills, swarmclawai/andrej-karpathy-skills)
which collectively codify Karpathy's public observations on LLM coding
pitfalls into a reusable skill format.

Also includes summary descriptions of Karpathy's flagship open-source
projects (nanoGPT, llm.c, minGPT, micrograd, nn-zero-to-hero) so that
agents can reference them for context.

Usage
-----
>>> from ilovetools.skills import karpathy_skills
>>> print(karpathy_skills.KARPATHY_CODING_GUIDELINES[:80])
# Karpathy-Inspired Coding Agent Guidelines
# ==========================================
# 1. Th
>>> "Think Before Coding" in karpathy_skills.KARPATHY_CODING_GUIDELINES
True
>>> "nanoGPT" in karpathy_skills.NANOGPT_SKILL
True
>>> "llm.c" in karpathy_skills.LLM_C_SKILL
True
"""

# ---------------------------------------------------------------------------
# Karpathy Coding-Agent Guidelines (the "four principles")
# ---------------------------------------------------------------------------

KARPATHY_CODING_GUIDELINES = """\\
# Karpathy-Inspired Coding Agent Guidelines
# ==========================================
# Source: andrej-karpathy-skills community repos on GitHub
#         (forrestchang, multica-ai, duolahypercho, swarmclawai)
# License: MIT
#
# These four principles improve LLM coding-agent behaviour by reducing
# common pitfalls: premature coding, over-engineering, sprawling diffs,
# and losing sight of the goal.

## 1. Think Before Coding
- Restate the problem in your own words before writing any code.
- Identify edge cases, failure modes, and assumptions explicitly.
- Write a brief plan (3-5 bullet points) of the approach.
- Do NOT start typing code until the plan is articulated.
- If the task is ambiguous, ask for clarification rather than guessing.

## 2. Simplicity First
- Choose the simplest solution that satisfies the requirements.
- Avoid speculative generality ("we might need this later").
- Prefer standard-library functions over hand-rolled equivalents.
- Fewer lines of clear code > more lines of clever code.
- If a design feels complex, step back and simplify before proceeding.

## 3. Surgical Changes
- Make the smallest possible change that achieves the goal.
- Keep edits orthogonal — one logical change per commit/diff.
- Do NOT refactor unrelated code in the same change.
- Preserve existing interfaces and behaviour unless explicitly asked to change them.
- Run the test suite after each change to catch regressions early.

## 4. Goal-Driven Execution
- Define what "done" looks like before starting (acceptance criteria).
- Write or identify tests that verify the goal is met.
- After implementation, verify against the acceptance criteria.
- If the goal is not met, iterate — do not declare success prematurely.
- Summarise what was changed and why at the end.
"""

# ---------------------------------------------------------------------------
# nanoGPT skill
# ---------------------------------------------------------------------------

NANOGPT_SKILL = """\\
# nanoGPT — Minimal GPT Training Boilerplate
# ===========================================
# Source: https://github.com/karpathy/nanoGPT (~60k stars, MIT)
# Author: Andrej Karpathy
#
# nanoGPT is a rewrite of minGPT focused on simplicity and speed.
# It provides a ~300-line training loop (train.py) and a ~300-line
# GPT model definition (model.py) that can reproduce GPT-2 (124M)
# on OpenWebText using a single 8xA100-40GB node in ~4 days.
#
# Key design lessons for agents:
# 1. Readability over abstraction — the entire model fits in 300 lines.
# 2. Prefer PyTorch primitives over custom wrappers.
# 3. Use well-known optimiser configs (AdamW, cosine decay, grad clipping).
# 4. Keep data loading simple and deterministic.
# 5. Log loss and grad-norm frequently; checkpoint regularly.
#
# Architecture overview:
#   - Decoder-only transformer with causal self-attention
#   - LayerNorm before (not after) attention and MLP (pre-norm)
#   - GELU activation in MLP
#   - Learned positional embeddings
#   - Tied input/output embeddings
#
# When an agent is asked to build or explain a GPT-style model,
# nanoGPT is the canonical reference for minimal, correct implementation.
"""

# ---------------------------------------------------------------------------
# llm.c skill
# ---------------------------------------------------------------------------

LLM_C_SKILL = """\\
# llm.c — LLM Training in Pure C/CUDA
# ====================================
# Source: https://github.com/karpathy/llm.c (~30k stars, MIT)
# Author: Andrej Karpathy
#
# llm.c implements GPT-2 training in ~1,000 lines of clean C (CPU fp32)
# with an optimised CUDA version (train_gpt2.cu) that rivals PyTorch
# Nightly throughput.  No PyTorch or Python dependency required.
#
# Key design lessons for agents:
# 1. Understanding emerges from implementing from scratch in a low-level language.
# 2. A reference implementation in C clarifies what the framework hides.
# 3. Performance optimisations (cuBLAS, cuDNN, flash attention) can be
#    layered incrementally without changing the core logic.
# 4. Keep a PyTorch reference (train_gpt2.py) alongside the C/CUDA code
#    for correctness checking.
#
# Architecture mapping (C <-> PyTorch):
#   - matmul -> nn.Linear / F.linear
#   - gelu_forward / gelu_backward -> F.gelu
#   - layernorm_forward / layernorm_backward -> nn.LayerNorm
#   - attention_forward / attention_backward -> nn.MultiheadAttention (causal)
#   - softmax -> F.softmax
#
# When an agent needs to explain LLM internals at the kernel level,
# llm.c is the definitive educational resource.
"""

# ---------------------------------------------------------------------------
# Karpathy's "Neural Networks: Zero to Hero" course skill
# ---------------------------------------------------------------------------

NN_ZERO_TO_HERO_SKILL = """\\
# Neural Networks: Zero to Hero — Karpathy's Course
# ==================================================
# Source: https://github.com/karpathy/nn-zero-to-hero
# Author: Andrej Karpathy
#
# A series of YouTube lectures that build neural networks from scratch,
# progressing from micrograd (automatic differentiation) to makemore
# (bigram and MLP language models) to nanoGPT (transformer).
#
# Lecture sequence:
# 1. micrograd — reverse-mode autodiff in ~100 lines of Python
# 2. makemore Part 1 — bigram character-level language model
# 3. makemore Part 2 — MLP language model (Bengio et al. 2003)
# 4. makemore Part 3 — activation statistics, BatchNorm, residual init
# 5. makemore Part 4 — backpropagation by hand (micrograd internals)
# 6. makemore Part 5 — WaveNet-style hierarchical MLP
# 7. nanoGPT — building a GPT from scratch
#
# Agent use-case:
# When asked to explain backpropagation, gradient descent, or how
# neural networks learn, reference this progression: start from the
# simplest autodiff engine and build up to transformers.
"""

# ---------------------------------------------------------------------------
# micrograd skill
# ---------------------------------------------------------------------------

MICROGRAD_SKILL = """\\
# micrograd — Tiny Autograd Engine
# =================================
# Source: https://github.com/karpathy/micrograd
# Author: Andrej Karpathy
#
# micrograd is a ~100-line Python autograd engine that implements
# reverse-mode automatic differentiation over a dynamically built
# DAG (directed acyclic graph).  It powers the first lecture of
# "Neural Networks: Zero to Hero."
#
# Core concepts:
#   - Value: a scalar wrapper that tracks gradients
#   - Each arithmetic op creates a new Value with a _backward closure
#   - backward(): topological sort -> reverse pass -> accumulate gradients
#
# Key lesson for agents:
#   Automatic differentiation is just chain rule applied to a DAG.
#   Every deep-learning framework (PyTorch, TensorFlow, JAX) does this
#   at scale, but the core idea fits in 100 lines.
#
# When an agent needs to explain "how does backpropagation work?",
# micrograd is the clearest starting point.
"""

# ---------------------------------------------------------------------------
# Aggregate dict for easy programmatic access
# ---------------------------------------------------------------------------

SKILLS = {
    "karpathy_coding_guidelines": KARPATHY_CODING_GUIDELINES,
    "karpathy_nanoGPT": NANOGPT_SKILL,
    "karpathy_llm_c": LLM_C_SKILL,
    "karpathy_nn_zero_to_hero": NN_ZERO_TO_HERO_SKILL,
    "karpathy_micrograd": MICROGRAD_SKILL,
}

__all__ = [
    "KARPATHY_CODING_GUIDELINES",
    "NANOGPT_SKILL",
    "LLM_C_SKILL",
    "NN_ZERO_TO_HERO_SKILL",
    "MICROGRAD_SKILL",
    "SKILLS",
]
