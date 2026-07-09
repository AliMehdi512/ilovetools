\
"""
ilovetools.skills — LLM / AI agent skills packaged as importable Python modules.

This subpackage bundles the most influential open-source agent-skill
repositories on GitHub into readable Python string constants that LLMs
can load at runtime and use as system-prompt instructions or reference
material.

Included skill collections
---------------------------
* **karpathy_skills** — Andrej Karpathy's coding-agent guidelines,
  nanoGPT, llm.c, nn-zero-to-hero, and micrograd skill texts.
* **anthropic_skills** — Anthropic's skill-creator guide, PDF skill,
  and code-review skill from the anthropics/skills repository.
* **agent_skills** — Community agent-skills spec, caveman token-efficiency,
  prompt-master, Codex skills, and engineering skills from the most-starred
  agent-skills repos.
* **prompt_engineering_skills** — 27 prompt frameworks, anti-hallucination
  checklist, and production prompt structure.

Quick Start
-----------
>>> from ilovetools.skills import list_skills, get_skill
>>> available = list_skills()
>>> 'karpathy_nanoGPT' in available
True
>>> skill_text = get_skill('karpathy_nanoGPT')
>>> 'nanoGPT' in skill_text
True
>>> get_skill('nonexistent_skill')
Traceback (most recent call last):
    ...
KeyError: "Skill 'nonexistent_skill' not found. Available skills: ...
>>> len(list_skills()) >= 10
True
"""

from . import karpathy_skills
from . import anthropic_skills
from . import agent_skills
from . import prompt_engineering_skills
from . import community_skills
from . import microsoft_skills
from . import openai_skills
from . import awesome_agent_skills

# Build the unified skill registry from all sub-modules
_SKILL_REGISTRY = {}
_SKILL_REGISTRY.update(karpathy_skills.SKILLS)
_SKILL_REGISTRY.update(anthropic_skills.SKILLS)
_SKILL_REGISTRY.update(agent_skills.SKILLS)
_SKILL_REGISTRY.update(prompt_engineering_skills.SKILLS)
_SKILL_REGISTRY.update(community_skills.SKILLS)
_SKILL_REGISTRY.update(microsoft_skills.SKILLS)
_SKILL_REGISTRY.update(openai_skills.SKILLS)
_SKILL_REGISTRY.update(awesome_agent_skills.SKILLS)


def list_skills():
    """Return a sorted list of all available skill names.

    Returns
    -------
    list[str]
        Alphabetically sorted names of every skill that can be retrieved
        via :func:`get_skill`.

    Examples
    --------
    >>> from ilovetools.skills import list_skills
    >>> names = list_skills()
    >>> isinstance(names, list)
    True
    >>> all(isinstance(n, str) for n in names)
    True
    >>> 'karpathy_coding_guidelines' in names
    True
    """
    return sorted(_SKILL_REGISTRY.keys())


def get_skill(name):
    """Retrieve the full text content of a named skill.

    Parameters
    ----------
    name : str
        The skill name as returned by :func:`list_skills`.

    Returns
    -------
    str
        The full skill content (markdown-formatted text).

    Raises
    ------
    KeyError
        If *name* does not match any available skill.

    Examples
    --------
    >>> from ilovetools.skills import get_skill
    >>> text = get_skill('karpathy_micrograd')
    >>> 'micrograd' in text
    True
    >>> text = get_skill('caveman_token_efficiency')
    >>> 'token' in text.lower()
    True
    """
    if name not in _SKILL_REGISTRY:
        available = ", ".join(list_skills())
        raise KeyError(
            f"Skill '{name}' not found. Available skills: {available}"
        )
    return _SKILL_REGISTRY[name]


def skill_info():
    """Return a dict mapping each skill name to its source and description.

    Returns
    -------
    dict[str, dict[str, str]]
        Each value has keys ``source`` and ``description``.

    Examples
    --------
    >>> from ilovetools.skills import skill_info
    >>> info = skill_info()
    >>> 'karpathy_nanoGPT' in info
    True
    >>> 'source' in info['karpathy_nanoGPT']
    True
    """
    return {
        "karpathy_coding_guidelines": {
            "source": "forrestchang/andrej-karpathy-skills (MIT)",
            "description": "Four Karpathy-inspired coding-agent principles.",
        },
        "karpathy_nanoGPT": {
            "source": "karpathy/nanoGPT (MIT)",
            "description": "Minimal GPT training boilerplate reference.",
        },
        "karpathy_llm_c": {
            "source": "karpathy/llm.c (MIT)",
            "description": "LLM training in pure C/CUDA — educational reference.",
        },
        "karpathy_nn_zero_to_hero": {
            "source": "karpathy/nn-zero-to-hero",
            "description": "Neural networks course from micrograd to nanoGPT.",
        },
        "karpathy_micrograd": {
            "source": "karpathy/micrograd (MIT)",
            "description": "Tiny autograd engine — how backprop works.",
        },
        "anthropic_skill_creator": {
            "source": "anthropics/skills (Apache 2.0)",
            "description": "How to create agent skills (SKILL.md format).",
        },
        "anthropic_pdf": {
            "source": "anthropics/skills (source-available)",
            "description": "PDF document creation and manipulation skill.",
        },
        "anthropic_code_review": {
            "source": "anthropics/skills (Apache 2.0)",
            "description": "Structured code review checklist and process.",
        },
        "agent_skill_spec": {
            "source": "addyosmani/agent-skills + anthropics/skills (MIT/Apache 2.0)",
            "description": "Agent skills specification — structure and lifecycle.",
        },
        "caveman_token_efficiency": {
            "source": "JuliusBrussee/caveman (MIT)",
            "description": "Token-efficient output skill for cost-conscious agents.",
        },
        "prompt_master": {
            "source": "nidhinjs/prompt-master (MIT)",
            "description": "Tool-specific prompt profiles for 20+ AI tools.",
        },
        "codex_skills": {
            "source": "openai/skills",
            "description": "OpenAI Codex skills catalog and structure.",
        },
        "engineering_skills": {
            "source": "addyosmani/agent-skills (MIT)",
            "description": "24 production-grade engineering skills for AI agents.",
        },
        "prompt_frameworks": {
            "source": "ckelsoe/prompt-architect (MIT)",
            "description": "27 prompt-engineering frameworks across 7 intent categories.",
        },
        "anti_hallucination": {
            "source": "codejunkie99/prompting-skill (MIT)",
            "description": "Anti-hallucination prompting checklist and techniques.",
        },
        "production_prompt_structure": {
            "source": "codejunkie99/prompting-skill (MIT)",
            "description": "Production-grade prompt structure template.",
        },
        "voltagent_awesome_skills": {
            "source": "VoltAgent/awesome-agent-skills (MIT)",
            "description": "1,000+ curated agent skills from Anthropic, Google, Vercel, Stripe, and more.",
        },
        "claude_community_skills": {
            "source": "alirezarezvani/claude-skills (MIT)",
            "description": "345+ Claude Code skills across 18 categories (marketing, engineering, DevOps, etc.).",
        },
        "luokai_mega_collection": {
            "source": "luokai0/ai-agent-skills-by-luo-kai",
            "description": "8,966+ skill files across 21 domains — world's largest agent skills collection.",
        },
        "seb1n_universal_skills": {
            "source": "seb1n/awesome-ai-agent-skills",
            "description": "90+ self-contained universal SKILL.md skills for autonomous agents.",
        },
        "mouadja_categorized_skills": {
            "source": "mouadja02/skills",
            "description": "716 agent skills across 31 categories with YAML frontmatter.",
        },
        "microsoft_azure_skills": {
            "source": "microsoft/agent-skills (MIT)",
            "description": "174+ Azure SDK skills: deploy, validate, diagnose, cost-manage.",
        },
        "microsoft_foundry_skills": {
            "source": "microsoft/agent-skills (MIT)",
            "description": "Microsoft Foundry hosted agent infrastructure and orchestration.",
        },
        "microsoft_agents_template": {
            "source": "microsoft/agent-skills (MIT)",
            "description": "AGENTS.md template for project-level agent configuration.",
        },
        "microsoft_mcp_config": {
            "source": "microsoft/agent-skills (MIT)",
            "description": "Azure and Foundry MCP endpoint configurations.",
        },
        # --- openai_skills ---
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
        # --- awesome_agent_skills ---
        "awesome_git_workflow": {
            "source": "addyosmani/agent-skills (MIT)",
            "description": "Git branching, conventional commits, and PR workflow guidelines.",
        },
        "awesome_refactoring": {
            "source": "addyosmani/agent-skills + VoltAgent/awesome-agent-skills (MIT)",
            "description": "Safe refactoring patterns: extract function/class, polymorphism, parameter objects.",
        },
        "awesome_api_design": {
            "source": "VoltAgent/awesome-agent-skills (MIT)",
            "description": "REST and GraphQL API design: naming, methods, status codes, versioning, security.",
        },
        "awesome_error_handling": {
            "source": "addyosmani/agent-skills + alirezarezvani/claude-skills (MIT)",
            "description": "Error handling patterns: specific catching, custom exceptions, context managers, retry.",
        },
        "awesome_doc_generation": {
            "source": "VoltAgent/awesome-agent-skills + mouadja02/skills (MIT)",
            "description": "Documentation generation: docstrings, README, API reference, changelog standards.",
        },
        "awesome_performance_profiling": {
            "source": "addyosmani/agent-skills + community (MIT)",
            "description": "Profiling workflow, tools (cProfile, line_profiler), and common optimizations.",
        },
    }


__all__ = [
    "karpathy_skills",
    "anthropic_skills",
    "agent_skills",
    "prompt_engineering_skills",
    "community_skills",
    "microsoft_skills",
    "openai_skills",
    "awesome_agent_skills",
    "list_skills",
    "get_skill",
    "skill_info",
]
