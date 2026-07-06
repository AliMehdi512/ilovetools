\
"""
Anthropic Claude agent skills — distilled from the anthropics/skills repository.

This module packages the core skill-creation guidelines and representative
skill content from Anthropic's public skills repository
(https://github.com/anthropics/skills, ~158k stars, Apache 2.0) into
readable Python string constants that LLMs can load at runtime.

The repository demonstrates how to build self-contained "skills" — folders
with a SKILL.md metadata file, optional scripts, references, and assets —
that Claude and other agents can discover and execute.

Usage
-----
>>> from ilovetools.skills import anthropic_skills
>>> "SKILL.md" in anthropic_skills.SKILL_CREATOR_GUIDE
True
>>> "frontmatter" in anthropic_skills.SKILL_CREATOR_GUIDE
True
>>> "pdf" in anthropic_skills.PDF_SKILL.lower()
True
"""

# ---------------------------------------------------------------------------
# Skill Creator Guide — how to write a SKILL.md
# ---------------------------------------------------------------------------

SKILL_CREATOR_GUIDE = """\
# How to Create an Agent Skill (Anthropic Style)
# ===============================================
# Source: https://github.com/anthropics/skills (skill-creator/SKILL.md)
# License: Apache 2.0
#
# A "skill" is a self-contained folder that an AI agent can discover and
# execute.  Each skill has a SKILL.md with YAML frontmatter (metadata)
# and a markdown body (instructions).

## Folder Structure
-------------------
skill-name/
├── SKILL.md          # Required: frontmatter + instructions
├── scripts/          # Optional: deterministic executable code
├── references/       # Optional: docs loaded on demand
└── assets/           # Optional: static resources (images, templates)

## SKILL.md Frontmatter
-----------------------
```yaml
---
name: my-skill-name
description: |
  A ~100-word description of what the skill does and when to trigger it.
  This text guides the agent's activation logic — be specific and proactive.
---
```

## Writing the Body
-------------------
1. **Title**: Start with a clear H1 title.
2. **When to trigger**: Describe the user intents that should activate this skill.
3. **Instructions**: Step-by-step guidance the agent follows when the skill is active.
4. **Examples**: Show input → output patterns to illustrate expected behaviour.
5. **Guidelines**: Constraints, best practices, and edge-case handling.

## Best Practices
-----------------
- Keep SKILL.md under 500 lines; move details to references/.
- Write a slightly proactive description to avoid under-triggering.
- Include concrete examples — they improve activation accuracy.
- Reference bundled files with relative paths from the skill folder.
- Use a hierarchical structure (## headings) if approaching the line limit.
- Test the skill with diverse phrasings to ensure reliable triggering.

## Skill Categories (from anthropics/skills)
--------------------------------------------
- **Creative & Design**: art direction, brand guidelines, design systems
- **Development & Technical**: code review, deployment, debugging
- **Enterprise & Communication**: email drafting, meeting notes, reports
- **Document Skills**: docx, pdf, pptx, xlsx creation and editing
"""

# ---------------------------------------------------------------------------
# PDF Skill — document creation reference
# ---------------------------------------------------------------------------

PDF_SKILL = """\
# PDF Document Creation Skill
# ============================
# Source: https://github.com/anthropics/skills (skills/pdf)
# License: Source-available (see repo for details)
#
# This skill enables an agent to create, edit, and manipulate PDF documents
# programmatically using Python libraries (reportlab, PyMuPDF, fpdf2).
#
## Capabilities
# - Generate PDFs from structured content (text, tables, images)
# - Extract text and metadata from existing PDFs
# - Merge, split, and watermark PDF files
# - Fill PDF form fields
# - Convert HTML/markdown to PDF
#
## Recommended Libraries
# - reportlab: Programmatic PDF generation (canvas, platypus)
# - PyMuPDF (fitz): Read, render, and edit existing PDFs
# - fpdf2: Lightweight PDF creation
# - weasyprint: HTML/CSS → PDF conversion
#
## Workflow
# 1. Determine the document structure (sections, headings, tables)
# 2. Choose the appropriate library based on complexity
# 3. Generate content with proper styling (fonts, margins, spacing)
# 4. Add page numbers, headers, footers if needed
# 5. Save and verify the output
#
## Example (reportlab)
# >>> from reportlab.lib.pagesizes import letter
# >>> from reportlab.pdfgen import canvas
# >>> c = canvas.Canvas("output.pdf", pagesize=letter)
# >>> c.drawString(100, 750, "Hello, World!")
# >>> c.save()
"""

# ---------------------------------------------------------------------------
# Code Review Skill
# ---------------------------------------------------------------------------

CODE_REVIEW_SKILL = """\
# Code Review Skill
# ==================
# Source: https://github.com/anthropics/skills (skills/code-review)
# License: Apache 2.0
#
# This skill guides an agent through a structured code review process,
# ensuring thoroughness and consistency across reviews.
#
## Review Checklist
# 1. **Correctness**: Does the code do what it claims? Check logic, edge cases.
# 2. **Security**: Input validation, auth checks, secret handling, injection risks.
# 3. **Performance**: Algorithmic complexity, unnecessary allocations, N+1 queries.
# 4. **Readability**: Naming, comments, function length, single responsibility.
# 5. **Testing**: Are tests included? Do they cover edge cases? Are they meaningful?
# 6. **Style**: Consistency with project conventions, linting compliance.
# 7. **Documentation**: Docstrings, README updates, API docs if interfaces changed.
#
## Review Process
# 1. Read the PR description and linked issue for context.
# 2. Review the diff holistically before line-by-line.
# 3. Categorise feedback: blocking, should-fix, nitpick, suggestion.
# 4. Provide actionable suggestions with code examples where possible.
# 5. Acknowledge what's done well, not just problems.
# 6. Summarise the review verdict: approve, request changes, or comment.
#
## Tone
# - Be constructive and specific.
# - Explain *why*, not just *what*.
# - Distinguish preferences from best practices.
"""

# ---------------------------------------------------------------------------
# Aggregate dict
# ---------------------------------------------------------------------------

SKILLS = {
    "anthropic_skill_creator": SKILL_CREATOR_GUIDE,
    "anthropic_pdf": PDF_SKILL,
    "anthropic_code_review": CODE_REVIEW_SKILL,
}

__all__ = [
    "SKILL_CREATOR_GUIDE",
    "PDF_SKILL",
    "CODE_REVIEW_SKILL",
    "SKILLS",
]
