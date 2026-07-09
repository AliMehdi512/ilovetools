"""
OpenAI Codex agent skills.

This module packages the most useful agent-skill guidelines from
OpenAI's Codex skills repository (github.com/openai/skills) and the
broader OpenAI agent-ecosystem into readable Python string constants
that LLMs can load at runtime.

Included skills
---------------
* **openai_codex_skills** — The Codex agent skill catalog overview,
  covering .system, .curated, and .experimental skill categories.
* **openai_plugin_guide** — How to build skill-only plugins for Codex.
* **openai_agent_skills_spec** — The Agent Skills standard specification
  (SKILL.md format, YAML frontmatter, discovery rules).
* **openai_code_review_skill** — Automated code-review skill for
  Codex agents (PR analysis, style checks, security scanning).
* **openai_testing_skill** — Test generation and execution skill
  (pytest scaffold, coverage analysis, edge-case discovery).
* **openai_debugging_skill** — Systematic debugging workflow.

Usage
-----
>>> from ilovetools.skills import openai_skills
>>> "Codex" in openai_skills.OPENAI_CODEX_SKILLS
True
>>> "SKILL.md" in openai_skills.OPENAI_AGENT_SKILLS_SPEC
True
>>> isinstance(openai_skills.SKILLS, dict)
True
>>> len(openai_skills.SKILLS) >= 5
True
"""

OPENAI_CODEX_SKILLS = """\
# OpenAI Codex Agent Skills Catalog
# ==================================
# Source: github.com/openai/skills (MIT, ~23k stars)
# Status: Deprecated - current guidance moved to OpenAI Plugins repo.
#
# Codex skills are reusable instructions, scripts, and resources that
# AI agents can discover and use to perform tasks.  They live in three
# tiers:
#
# .system/        - Built-in skills shipped with Codex (always available).
# .curated/       - Officially maintained skills, installable via the
#                    built-in skill-installer command.
# .experimental/  - Community-contributed skills that may be unstable.
#
# Installation:
#   codex skill install <skill-name>
#   # Restart Codex for changes to take effect.
#
# Creating a custom skill:
#   1. Create a directory with a SKILL.md file.
#   2. Add YAML frontmatter: name, description, version, author.
#   3. Include any supporting scripts or resources.
#   4. Place it under .curated/ or .experimental/ and install.
#
# Skill discovery:
#   Codex scans .system/, .curated/, and .experimental/ directories.
#   Skills are auto-discovered when their SKILL.md is found.
#
# Best practices:
#   - Keep skills small and focused (one task per skill).
#   - Include usage examples in SKILL.md.
#   - Test skills with `codex skill test <skill-name>`.
#   - Version your skills using semver in the frontmatter.
"""

OPENAI_PLUGIN_GUIDE = """\
# Building Skill-Only Plugins for OpenAI Codex
# ==============================================
# Source: OpenAI Plugins repository (successor to openai/skills)
#
# A skill-only plugin is a package that provides agent skills without
# any runtime tool integration.  It is the simplest way to distribute
# reusable agent instructions.
#
# Plugin structure:
#   my-plugin/
#   +-- plugin.json          # Plugin manifest
#   +-- skills/
#   |   +-- my-skill/
#   |   |   +-- SKILL.md     # Skill definition
#   |   |   +-- helpers.py   # Optional supporting scripts
#   |   +-- another-skill/
#   |       +-- SKILL.md
#   +-- README.md
#
# plugin.json format:
#   {
#     "name": "my-plugin",
#     "version": "1.0.0",
#     "description": "A collection of coding skills",
#     "skills": ["skills/my-skill", "skills/another-skill"]
#   }
#
# SKILL.md frontmatter:
#   ---
#   name: my-skill
#   description: Analyzes Python code for common anti-patterns
#   version: 1.0.0
#   author: developer@example.com
#   tags: [python, code-quality, analysis]
#   ---
#
# Installation:
#   codex plugin install ./my-plugin
#   codex plugin list
#   codex plugin uninstall my-plugin
#
# Tips:
#   - Skills in a plugin are namespaced by the plugin name.
#   - Use `codex plugin validate ./my-plugin` before publishing.
#   - Publish to the community registry with `codex plugin publish`.
"""

OPENAI_AGENT_SKILLS_SPEC = """\
# Agent Skills Standard Specification
# =====================================
# Source: agentskills.io / OpenAI / Anthropic joint specification
#
# The Agent Skills standard defines a portable format for packaging
# reusable instructions that any AI agent (Claude, Codex, Cursor,
# Windsurf, etc.) can discover and use.
#
# ## File Format
# Each skill is a directory containing at minimum a SKILL.md file.
#
# ## SKILL.md Structure
#   ---
#   name: skill-name              # Required, kebab-case
#   description: Short summary    # Required, max 200 chars
#   version: 1.0.0                # Required, semver
#   author: name <email>          # Optional
#   tags: [tag1, tag2]            # Optional, for discovery
#   dependencies: [other-skill]   # Optional, skill prerequisites
#   compatible: [claude, codex]   # Optional, agent compatibility list
#   ---
#
#   # Skill Title
#
#   ## Purpose
#   What this skill does and when to use it.
#
#   ## Instructions
#   Step-by-step guidance for the agent.
#
#   ## Examples
#   Concrete usage examples.
#
#   ## Limitations
#   Known constraints or edge cases.
#
# ## Discovery Rules
# 1. Agents scan recursively for SKILL.md files.
# 2. Skills are identified by the `name` field in frontmatter.
# 3. If two skills have the same name, the one in the deeper directory wins.
# 4. Skills with `compatible` lists are only loaded by listed agents.
# 5. Skills with unmet `dependencies` are skipped with a warning.
#
# ## Portability
# The same SKILL.md works across:
#   - Claude Code (anthropics/skills)
#   - OpenAI Codex (openai/skills -> plugins)
#   - Cursor (.cursor/rules/)
#   - Windsurf
#   - OpenCode
#   - Google Antigravity
#
# ## Best Practices
# - Keep instructions imperative and concise.
# - Use numbered steps for procedures.
# - Include at least one worked example.
# - Specify limitations explicitly.
# - Version skills independently.
"""

OPENAI_CODE_REVIEW_SKILL = """\
# Automated Code Review Skill for AI Agents
# ===========================================
# Source: OpenAI Codex curated skills
#
# This skill guides an AI agent through a systematic code review
# process for pull requests and commits.
#
# ## Review Checklist
# 1. **Syntax and Style**
#    - Check for PEP 8 compliance (Python) / language-appropriate style.
#    - Verify consistent naming conventions.
#    - Flag dead code and unused imports.
#
# 2. **Logic and Correctness**
#    - Trace through control flow for edge cases.
#    - Check for off-by-one errors, null/None dereferences.
#    - Verify error handling completeness (try/except coverage).
#    - Look for race conditions in concurrent code.
#
# 3. **Security**
#    - Scan for hardcoded secrets / credentials.
#    - Check for injection vulnerabilities (SQL, command, path).
#    - Verify input validation on public APIs.
#    - Flag unsafe deserialization.
#
# 4. **Performance**
#    - Identify O(n^2) or worse algorithms where O(n) is possible.
#    - Check for unnecessary memory allocations.
#    - Look for missing database query optimization.
#    - Flag synchronous I/O in async contexts.
#
# 5. **Tests**
#    - Verify new code has corresponding tests.
#    - Check edge cases are covered.
#    - Ensure test names are descriptive.
#    - Look for flaky test patterns (time-dependent, network-dependent).
#
# 6. **Documentation**
#    - Check docstrings on public functions/classes.
#    - Verify README updates for new features.
#    - Flag outdated comments.
#
# ## Output Format
# Present findings as:
#   [SEVERITY] file:line - description
#   Severity: CRITICAL | WARNING | INFO | SUGGESTION
#
# ## Agent Behavior
# - Always run the full checklist, even for small PRs.
# - Provide actionable suggestions, not just complaints.
# - Acknowledge what was done well.
# - Prioritize findings by severity.
"""

OPENAI_TESTING_SKILL = """\
# Test Generation and Execution Skill for AI Agents
# ==================================================
# Source: OpenAI Codex curated skills
#
# This skill enables an AI agent to generate, run, and analyze tests
# for Python codebases using pytest.
#
# ## Test Generation Process
# 1. **Analyze the target code**
#    - Read the function/class signature and docstring.
#    - Identify all code paths (branches, exceptions, loops).
#    - Note type hints for generating type-appropriate inputs.
#
# 2. **Generate test cases**
#    - Happy path: typical valid inputs with expected outputs.
#    - Edge cases: empty inputs, boundary values, None/empty strings.
#    - Error cases: invalid types, out-of-range values, missing args.
#    - Integration: interactions with dependencies (use mocks).
#
# 3. **Write test file**
#    - Name: test_<module>.py
#    - Use pytest fixtures for setup/teardown.
#    - Use parametrize for multiple input combinations.
#    - Include descriptive test names: test_<function>_<scenario>.
#
# 4. **Run tests**
#    - Execute: pytest tests/test_<module>.py -v
#    - Capture stdout, stderr, and exit code.
#    - If tests fail, analyze traceback, fix code or test, re-run.
#
# 5. **Coverage analysis**
#    - Run: pytest --cov=<module> --cov-report=term-missing
#    - Identify uncovered lines.
#    - Add tests for uncovered branches.
#    - Target: >= 90% line coverage for new code.
#
# ## Test File Template
#   import pytest
#   from <module> import <target>
#
#   class TestTarget:
#       def test_typical_usage(self):
#           assert <target>(valid_input) == expected
#
#       @pytest.mark.parametrize("input,expected", [
#           ("case1", "result1"),
#           ("case2", "result2"),
#       ])
#       def test_parametrized(self, input, expected):
#           assert <target>(input) == expected
#
#       def test_raises_on_invalid(self):
#           with pytest.raises(ValueError):
#               <target>(invalid_input)
#
# ## Agent Behavior
# - Always run tests after generation; never assume they pass.
# - If a test fails due to a bug in the code, fix the code.
# - If a test fails due to a bug in the test, fix the test.
# - Report results clearly: X passed, Y failed, Z skipped.
"""

OPENAI_DEBUGGING_SKILL = """\
# Systematic Debugging Skill for AI Agents
# ==========================================
# Source: OpenAI Codex curated skills + community best practices
#
# This skill provides a structured debugging workflow for AI agents
# encountering runtime errors or unexpected behavior.
#
# ## Debugging Workflow
# 1. **Reproduce the error**
#    - Capture the exact command, input, and environment.
#    - Minimize the reproduction case (remove unrelated code).
#    - Verify the error is deterministic (or note intermittency).
#
# 2. **Read the traceback**
#    - Identify the exception type and message.
#    - Locate the failing line in the traceback.
#    - Trace back through the call stack to the root cause.
#    - Note any library frames vs. user code frames.
#
# 3. **Form a hypothesis**
#    - Based on the error type, propose 1-3 likely causes.
#    - Rank hypotheses by probability and ease of verification.
#
# 4. **Verify the hypothesis**
#    - Add a print/assert statement at the failure point.
#    - Or write a minimal test that reproduces the issue.
#    - Check variable types and values at the failure point.
#
# 5. **Fix the root cause**
#    - Address the underlying issue, not just the symptom.
#    - Make the smallest possible change that fixes the issue.
#    - Add a regression test to prevent reintroduction.
#
# 6. **Verify the fix**
#    - Run the reproduction case - error should be gone.
#    - Run the full test suite - no new failures.
#    - Run the regression test - should pass.
#
# ## Common Error Patterns
# - **TypeError**: Wrong type passed to a function. Check type hints.
# - **KeyError/AttributeError**: Missing dict key or object attribute.
#   Check for None values and use .get() or hasattr().
# - **IndexError**: List index out of range. Check len() before indexing.
# - **ImportError**: Module not found or circular import. Check sys.path.
# - **RecursionError**: Infinite recursion. Check base case.
#
# ## Agent Behavior
# - Never guess at a fix without understanding the root cause.
# - Always reproduce before fixing.
# - Always add a regression test after fixing.
# - Report: root cause, fix applied, tests added.
"""

SKILLS = {
    "openai_codex_skills": OPENAI_CODEX_SKILLS,
    "openai_plugin_guide": OPENAI_PLUGIN_GUIDE,
    "openai_agent_skills_spec": OPENAI_AGENT_SKILLS_SPEC,
    "openai_code_review_skill": OPENAI_CODE_REVIEW_SKILL,
    "openai_testing_skill": OPENAI_TESTING_SKILL,
    "openai_debugging_skill": OPENAI_DEBUGGING_SKILL,
}

SKILL_INFO = {
    "openai_codex_skills": {
        "source": "openai/skills (MIT, ~23k stars)",
        "description": "Overview of the Codex agent skill catalog (.system, .curated, .experimental).",
    },
    "openai_plugin_guide": {
        "source": "OpenAI Plugins repo (MIT)",
        "description": "How to build, package, and distribute skill-only plugins for Codex.",
    },
    "openai_agent_skills_spec": {
        "source": "agentskills.io / OpenAI / Anthropic (MIT)",
        "description": "The Agent Skills standard: SKILL.md format, YAML frontmatter, discovery rules.",
    },
    "openai_code_review_skill": {
        "source": "openai/skills .curated (MIT)",
        "description": "Systematic code-review checklist for PR analysis (style, logic, security, perf, tests, docs).",
    },
    "openai_testing_skill": {
        "source": "openai/skills .curated (MIT)",
        "description": "Test generation, execution, and coverage analysis workflow using pytest.",
    },
    "openai_debugging_skill": {
        "source": "openai/skills .curated + community (MIT)",
        "description": "Structured debugging workflow: reproduce, read traceback, hypothesize, verify, fix, test.",
    },
}

__all__ = [
    "OPENAI_CODEX_SKILLS",
    "OPENAI_PLUGIN_GUIDE",
    "OPENAI_AGENT_SKILLS_SPEC",
    "OPENAI_CODE_REVIEW_SKILL",
    "OPENAI_TESTING_SKILL",
    "OPENAI_DEBUGGING_SKILL",
    "SKILLS",
    "SKILL_INFO",
]
