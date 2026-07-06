\
"""
Prompt-engineering skills — distilled from top prompt-engineering repositories
and community best practices on GitHub.

Sources:
  - ckelsoe/prompt-architect (27 frameworks across 7 intent categories)
  - codejunkie99/prompting-skill (production prompt structure + anti-hallucination)
  - Community prompt-engineering guides (CO-STAR, RISEN, RTF, CoT, CoD frameworks)

Usage
-----
>>> from ilovetools.skills import prompt_engineering_skills
>>> "CO-STAR" in prompt_engineering_skills.PROMPT_FRAMEWORKS
True
>>> "anti-hallucination" in prompt_engineering_skills.ANTI_HALLUCINATION_SKILL.lower()
True
>>> "production" in prompt_engineering_skills.PRODUCTION_PROMPT_STRUCTURE.lower()
True
"""

# ---------------------------------------------------------------------------
# Prompt Frameworks (27 frameworks from prompt-architect)
# ---------------------------------------------------------------------------

PROMPT_FRAMEWORKS = """\
# Prompt Engineering Frameworks
# ==============================
# Source: https://github.com/ckelsoe/prompt-architect (MIT)
# 27 research-backed frameworks across 7 intent categories.
#
## Intent Categories
# 1. **Creative** — storytelling, brainstorming, content generation
# 2. **Analytical** — data analysis, reasoning, evaluation
# 3. **Instructional** — tutorials, guides, how-to content
# 4. **Conversational** — dialogue, roleplay, chatbots
# 5. **Code** — code generation, debugging, review
# 6. **Summarisation** — condensing, extracting key points
# 7. **Transformation** — reformatting, translating, converting
#
## Key Frameworks
#
### CO-STAR
# Context -> Objective -> Style -> Tone -> Audience -> Response format
# Best for: structured content generation with specific audience targeting.
#
### RISEN
# Role -> Instruction -> Steps -> End state -> Narrowing
# Best for: multi-step tasks with a clear definition of done.
#
### RISE
# Role -> Instruction -> Steps -> End state
# Best for: simpler tasks that still need role assignment and clear steps.
#
### RTF (Role-Task-Format)
# Role -> Task -> Format
# Best for: quick, direct prompts where brevity matters.
#
### Chain-of-Thought (CoT)
# "Think step by step before answering."
# Best for: math, logic, multi-step reasoning.
#
### Chain-of-Density (CoD)
# Iteratively increase information density in summaries.
# Best for: creating dense, information-rich summaries.
#
### TIDD-EC
# Task -> Intent -> Details -> Domain -> Examples -> Constraints
# Best for: domain-specific tasks with strict constraints.
#
## Framework Selection Guide
# - Quick task? -> RTF
# - Multi-step? -> RISEN or RISE
# - Audience-specific? -> CO-STAR
# - Reasoning needed? -> CoT
# - Dense summary? -> CoD
# - Domain-specific? -> TIDD-EC
"""

# ---------------------------------------------------------------------------
# Anti-Hallucination Skill (from codejunkie99/prompting-skill)
# ---------------------------------------------------------------------------

ANTI_HALLUCINATION_SKILL = """\
# Anti-Hallucination Prompting Skill
# ===================================
# Source: https://github.com/codejunkie99/prompting-skill (MIT)
#
# Techniques to reduce LLM hallucinations in production prompts.
#
## Five Core Rules
# 1. **Ground in facts**: Provide verifiable context before asking for output.
# 2. **Lock the output format**: Specify exact JSON schema, markdown structure,
#    or template.  Reject outputs that deviate.
# 3. **Use few-shot examples**: Show 2-3 input->output pairs to anchor behaviour.
# 4. **Explicitly permit "I don't know"**: Tell the model it's acceptable to
#    say it lacks information rather than fabricating.
# 5. **Constrain the scope**: Limit the response to the provided context;
#    instruct the model not to use external knowledge for factual claims.
#
## Anti-Hallucination Checklist
# - [ ] Is the source context provided and referenced?
# - [ ] Is the output format locked (schema/template)?
# - [ ] Are there few-shot examples?
# - [ ] Can the model say "I don't know"?
# - [ ] Is the scope constrained to provided context?
# - [ ] Are entity names, dates, and numbers verified?
# - [ ] Is there a verification step (self-check or external)?
#
## Example Pattern
# "Based ONLY on the following document, answer the question.
#  If the answer is not in the document, say 'I don't know.'
#  Document: {context}
#  Question: {question}
#  Format: {{'answer': str, 'confidence': float, 'source_quote': str}}"
"""

# ---------------------------------------------------------------------------
# Production Prompt Structure (from codejunkie99/prompting-skill)
# ---------------------------------------------------------------------------

PRODUCTION_PROMPT_STRUCTURE = """\
# Production Prompt Structure
# ============================
# Source: https://github.com/codejunkie99/prompting-skill (MIT)
#
# A reliable template for production-grade LLM prompts.
#
## Structure
# 1. **System role**: Define who the LLM is and what it does.
# 2. **Context**: Provide all necessary background information.
# 3. **Task**: State the specific task clearly and unambiguously.
# 4. **Constraints**: List what the LLM must NOT do.
# 5. **Format**: Specify the exact output format.
# 6. **Examples**: 2-3 few-shot examples (input -> output).
# 7. **Verification**: A self-check instruction.
#
## Template
# ```
# SYSTEM: You are a {role}. Your job is to {job_description}.
#
# CONTEXT:
# {relevant_context}
#
# TASK: {specific_task}
#
# CONSTRAINTS:
# - Do not {constraint_1}
# - Do not {constraint_2}
# - If unsure, say "I don't know"
#
# FORMAT:
# {output_format_specification}
#
# EXAMPLES:
# Input: {example_input_1}
# Output: {example_output_1}
#
# Input: {example_input_2}
# Output: {example_output_2}
#
# VERIFICATION:
# Before responding, verify: {verification_criteria}
# ```
#
## Agent Use-Case
# When an agent needs to construct a prompt for an LLM call, use this
# structure as the base template and fill in each section.
"""

# ---------------------------------------------------------------------------
# Aggregate dict
# ---------------------------------------------------------------------------

SKILLS = {
    "prompt_frameworks": PROMPT_FRAMEWORKS,
    "anti_hallucination": ANTI_HALLUCINATION_SKILL,
    "production_prompt_structure": PRODUCTION_PROMPT_STRUCTURE,
}

__all__ = [
    "PROMPT_FRAMEWORKS",
    "ANTI_HALLUCINATION_SKILL",
    "PRODUCTION_PROMPT_STRUCTURE",
    "SKILLS",
]
