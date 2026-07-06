\
"""
Community agent-skills — distilled from the most-starred open-source
agent-skills repositories on GitHub.

Sources:
  - addyosmani/agent-skills (~69k stars, MIT) — 24 production-grade
    engineering skills for AI coding agents.
  - openai/skills (~23k stars) — Codex agent skills catalog.
  - JuliusBrussee/caveman (~82k stars, MIT) — token-efficient output skill.
  - nidhinjs/prompt-master (~10k stars, MIT) — prompt profiles for 20+ AI tools.
  - alirezarezvani/claude-skills (~5.2k stars, MIT) — 354 Claude Code skills.

Usage
-----
>>> from ilovetools.skills import agent_skills
>>> "spec" in agent_skills.AGENT_SKILL_SPEC.lower()
True
>>> "token" in agent_skills.CAVEMAN_SKILL.lower()
True
>>> "prompt" in agent_skills.PROMPT_MASTER_SKILL.lower()
True
"""

# ---------------------------------------------------------------------------
# Agent Skills Specification (from addyosmani/agent-skills + anthropics/skills)
# ---------------------------------------------------------------------------

AGENT_SKILL_SPEC = """\
# Agent Skills Specification (Community)
# =======================================
# Sources: addyosmani/agent-skills, anthropics/skills, openai/skills
# License: MIT / Apache 2.0
#
# An "agent skill" is a self-contained, reusable instruction set that an
# AI agent can discover, load, and execute to perform a specific task.
#
## Core Components
# 1. **SKILL.md** — frontmatter (name, description) + markdown body (instructions)
# 2. **scripts/** — deterministic executable code (optional)
# 3. **references/** — documentation loaded on demand (optional)
# 4. **assets/** — static resources like templates or images (optional)
#
## Skill Lifecycle
#   discovery -> activation (match user intent to skill description)
#   -> loading (read SKILL.md + bundled files) -> execution -> result
#
## Writing Good Skill Descriptions
# - 50-150 words describing what the skill does and when to use it.
# - Be proactive: explicitly mention common user intents.
# - Avoid vague terms like "helps with" — say what it *does*.
# - Include 1-2 example triggers.
#
## Verification Gates (from addyosmani/agent-skills)
# Each skill should define verification steps that confirm the task is done:
# - Tests pass
# - Linting passes
# - Output matches expected format
# - No regressions introduced
#
## Anti-Rationalisation Table
# Skills should include a table mapping common agent shortcuts to the
# correct behaviour, preventing the agent from rationalising skipping steps.
"""

# ---------------------------------------------------------------------------
# Caveman skill — token-efficient output (JuliusBrussee/caveman)
# ---------------------------------------------------------------------------

CAVEMAN_SKILL = """\
# Caveman — Token-Efficient Output Skill
# =======================================
# Source: https://github.com/JuliusBrussee/caveman (~82k stars, MIT)
#
# Caveman is a skill/plugin that compresses agent spoken output to reduce
# token usage without losing fidelity.  Code, commands, and error outputs
# are preserved byte-for-byte; only the prose is compressed.
#
## Core Rules
# 1. Respond in concise "caveman-speak" — short, direct phrases.
# 2. NEVER alter code blocks, commands, file paths, or error messages.
# 3. Drop filler words ("I will now", "Let me", "Sure!").
# 4. Use bullet points instead of paragraphs.
# 5. One sentence per thought. No compound sentences.
#
## Example
# Normal: "I'll now search through the repository to find all the files
#   that match the pattern you specified and then list them for you."
# Caveman: "Searching repo. Listing matches below."
#
## When to Use
# - Long coding sessions where token budget matters.
# - CI/CD agent pipelines with cost constraints.
# - Any context where verbose prose wastes tokens without adding value.
#
## When NOT to Use
# - User-facing explanations requiring detail.
# - Documentation generation.
# - Onboarding or teaching scenarios.
"""

# ---------------------------------------------------------------------------
# Prompt Master skill — prompt profiles for 20+ AI tools
# ---------------------------------------------------------------------------

PROMPT_MASTER_SKILL = """\
# Prompt Master — Tool-Specific Prompt Profiles
# ==============================================
# Source: https://github.com/nidhinjs/prompt-master (~10k stars, MIT)
#
# Prompt Master provides ready-made prompt profiles ("skills") for 20+ AI
# tools and a Universal Fingerprint method for unseen tools.
#
## Supported Tools
# Claude, ChatGPT, Gemini, o1/o3, MiniMax, Cursor, Claude Code,
# GitHub Copilot, Midjourney, DALL-E, Stable Diffusion, Runway,
# ElevenLabs, Zapier, Make, and more.
#
## Universal Fingerprint (4-Question Method)
# For any AI tool not in the profile list, answer these four questions:
# 1. **Contract**: What format does the tool expect? (markdown, JSON, XML?)
# 2. **Verbosity**: Does it prefer concise or detailed prompts?
# 3. **Grounding**: Does it need explicit context or can it infer?
# 4. **Scope**: What are the tool's hard limits? (token cap, file types?)
#
## Profile Structure
# Each tool profile contains:
# - Optimal prompt format (structure, sections, delimiters)
# - Verbosity tuning (how much detail to include)
# - Grounding strategy (how much context to provide)
# - Chain-of-thought control (when to use CoT vs direct answers)
# - File scope (what file types/sizes the tool handles)
#
## Agent Use-Case
# When an agent needs to generate prompts for a specific tool, load the
# matching profile and format the prompt accordingly.  For unknown tools,
# apply the Universal Fingerprint method.
"""

# ---------------------------------------------------------------------------
# OpenAI Codex skills catalog reference
# ---------------------------------------------------------------------------

CODEX_SKILLS_SKILL = """\
# OpenAI Codex Skills Catalog
# ============================
# Source: https://github.com/openai/skills (~23k stars)
# Note: Deprecated in favor of openai/plugins, but skills remain available.
#
# Codex skills are reusable, task-specific instruction sets that the Codex
# agent can discover and execute.  They are organised into three tiers:
#
## Tiers
# 1. **System skills** (.system/) — auto-installed with Codex, always available.
# 2. **Curated skills** (.curated/) — reviewed and maintained, install via:
#      $skill-installer <skill-name>
# 3. **Experimental skills** (.experimental/) — community contributions:
#      $skill-installer install <skill-name> from .experimental
#
## Example Skills
# - gh-address-comments: Address GitHub PR review comments systematically.
# - create-plan: Generate a structured execution plan for a complex task.
# - test-runner: Run and interpret test suites with actionable feedback.
#
## Skill Structure
# Each skill folder contains:
#   SKILL.md (frontmatter + instructions)
#   scripts/ (optional executable code)
#   LICENSE.txt (per-skill license)
#
## Agent Integration
# After installing a skill, restart Codex to apply.  Skills are invoked
# via slash commands or natural language matching the skill description.
"""

# ---------------------------------------------------------------------------
# Claude Code engineering skills (from addyosmani/agent-skills)
# ---------------------------------------------------------------------------

ENGINEERING_SKILLS_SKILL = """\
# Production-Grade Engineering Skills for AI Agents
# =================================================
# Source: https://github.com/addyosmani/agent-skills (~69k stars, MIT)
#
# 24 production-grade engineering skills packaged as AI coding-agent
# workflows that enforce senior-engineering practices across the
# development lifecycle.
#
## Lifecycle Skills (8 entry points)
# 1. **spec** — Write specifications before coding.
# 2. **plan** — Break down work into ordered, verifiable steps.
# 3. **implement** — Write code following the spec and plan.
# 4. **test** — Write and run tests; verify coverage.
# 5. **review** — Structured code review with checklist.
# 6. **refactor** — Safe, incremental refactoring.
# 7. **debug** — Systematic debugging workflow.
# 8. **ship** — Pre-deployment checklist and release process.
#
## Meta-Skill
# A "meta-skill" that helps the agent choose the right lifecycle skill
# based on the current task and project state.
#
## Key Design Principles
# - Each skill has verification gates (tests must pass before proceeding).
# - Anti-rationalisation tables prevent the agent from skipping steps.
# - Skills are composable — the output of one feeds into the next.
# - Slash commands map to lifecycle stages for easy invocation.
#
## Agent Use-Case
# When an agent is working on a production codebase, these skills enforce
# discipline: no coding without a spec, no shipping without tests passing,
# no refactoring without a plan.
"""

# ---------------------------------------------------------------------------
# Aggregate dict
# ---------------------------------------------------------------------------

SKILLS = {
    "agent_skill_spec": AGENT_SKILL_SPEC,
    "caveman_token_efficiency": CAVEMAN_SKILL,
    "prompt_master": PROMPT_MASTER_SKILL,
    "codex_skills": CODEX_SKILLS_SKILL,
    "engineering_skills": ENGINEERING_SKILLS_SKILL,
}

__all__ = [
    "AGENT_SKILL_SPEC",
    "CAVEMAN_SKILL",
    "PROMPT_MASTER_SKILL",
    "CODEX_SKILLS_SKILL",
    "ENGINEERING_SKILLS_SKILL",
    "SKILLS",
]
