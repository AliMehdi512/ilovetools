"""
Community agent-skills — additional collections from the most-starred
open-source agent-skills repositories on GitHub.

Sources:
  - VoltAgent/awesome-agent-skills (~1.5k+ skills, MIT) — curated collection
    of 1000+ official and community-built agent skills from teams like
    Anthropic, Google, Vercel, Stripe, Cloudflare, Sentry, Expo, Hugging Face.
  - alirezarezvani/claude-skills (~5.2k stars, MIT) — 345+ Claude Code skills
    across 18 categories (marketing, engineering, DevOps, security, finance, etc.).
  - luokai0/ai-agent-skills-by-luo-kai — 8,966+ skill files across 21 domains.
  - seb1n/awesome-ai-agent-skills — 90+ self-contained universal SKILL.md skills.
  - mouadja02/skills — 716 agent skills across 31 categories.

Usage
-----
>>> from ilovetools.skills import community_skills
>>> "skill" in community_skills.VOLTAGENT_AWESOME_SKILLS.lower()
True
>>> "claude" in community_skills.CLAUDE_COMMUNITY_SKILLS.lower()
True
>>> "domain" in community_skills.LUOKAI_MEGA_COLLECTION.lower()
True
>>> "universal" in community_skills.SEB1N_UNIVERSAL_SKILLS.lower()
True
>>> "category" in community_skills.MOUADJA_CATEGORIZED_SKILLS.lower()
True
"""

# ---------------------------------------------------------------------------
# VoltAgent / awesome-agent-skills
# ---------------------------------------------------------------------------

VOLTAGENT_AWESOME_SKILLS = """\
# VoltAgent awesome-agent-skills — Curated Collection
# ====================================================
# Source: VoltAgent/awesome-agent-skills (GitHub, MIT)
# Stars: ~1.5k+  |  Skills: 1,000+
#
# A curated collection of 1,000+ official and community-built Agent Skills
# for AI coding assistants. Skills are contributed by engineering teams at
# Anthropic, Google Labs, Vercel, Stripe, Cloudflare, Netlify, Sentry, Expo,
# Hugging Face, Figma, and the broader community.
#
# ## Compatibility
# Claude Code, OpenAI Codex, Gemini CLI, Cursor, GitHub Copilot, OpenCode,
# Windsurf, and more.
#
# ## Skill Categories (by team)
#
# ### Official Claude Skills (Anthropic)
# - docx: Create and manipulate Word documents
# - template: Skill creation template and best practices
# - PDF processing and manipulation
# - Code review with structured checklists
#
# ### Skills by Vercel
# - Project setup and deployment workflows
# - Next.js best practices and optimization
# - Edge function patterns
#
# ### Skills by Cloudflare
# - Workers deployment and testing
# - D1 database management
# - R2 storage operations
#
# ### Skills by Stripe
# - Payment integration patterns
# - Webhook handling and verification
# - Subscription management
#
# ### Skills by Google Labs
# - Gemini CLI integration
# - Google Cloud deployment patterns
# - Vertex AI model orchestration
#
# ### Skills by Supabase
# - Database schema design
# - Row-level security policies
# - Real-time subscription setup
#
# ### Skills by Hugging Face
# - Model fine-tuning workflows
# - Dataset preparation and tokenization
# - Inference endpoint management
#
# ### Skills by Expo
# - React Native project setup
# - EAS build and submit
# - OTA update deployment
#
# ### Skills by Sentry
# - Error monitoring setup
# - Performance tracing
# - Release tracking integration
#
# ### Skills by Netlify
# - Static site deployment
# - Edge functions and redirects
# - Form handling and identity
#
# ### Skills by HashiCorp
# - Terraform infrastructure as code
# - Consul service mesh
# - Vault secrets management
#
# ### Skills by Trail of Bits
# - Security auditing
# - Smart contract review
# - Threat modeling
#
# ### Community Skills
# - n8n automation workflows
# - Context engineering patterns
# - Testing and development helpers
# - Data analysis and visualization
#
# ## Skill Quality Standards
# Each skill must include:
# 1. SKILL.md with frontmatter (name, description, icon)
# 2. Clear instructions with examples
# 3. Verification gates (how to confirm the skill worked)
# 4. Optional scripts/ for deterministic code execution
# 5. Optional references/ for on-demand documentation
#
# ## How to Use
# Copy the skill directory into your agent's skills path:
# - Claude Code: ~/.claude/skills/
# - Codex: ~/.codex/skills/
# - Cursor: .cursor/skills/
# - Gemini CLI: ~/.gemini/skills/
# - GitHub Copilot: .github/copilot/skills/
"""

# ---------------------------------------------------------------------------
# alirezarezvani / claude-skills
# ---------------------------------------------------------------------------

CLAUDE_COMMUNITY_SKILLS = """\
# alirezarezvani/claude-skills — 345+ Claude Code Skills
# =======================================================
# Source: alirezarezvani/claude-skills (GitHub, MIT)
# Stars: ~5.2k  |  Skills: 345+  |  Categories: 18
#
# The most comprehensive open-source collection of Claude Code skills,
# agent plugins, and coding-agent prompts for 13 AI coding tools.
#
# ## Compatible Tools
# Claude Code, Codex, Gemini CLI, Cursor, Hermes, Mistral Vibe,
# Windsurf, GitHub Copilot, OpenCode, and more.
#
# ## Skill Categories
#
# ### Marketing (27 skills)
# - copywriting, content-strategy, seo-audit, email-sequences
# - content-creation, conversion-optimization, analytics
# - a-b-testing, onboarding, launch-strategy, aso
# - social-media, pricing, brand-strategy, growth-hacking
#
# ### Engineering Team (24 skills)
# - system-design, frontend, backend, fullstack
# - qa, devops, security-operations, code-review
# - aws-architecture, ms365-administration, tdd
# - tech-stack-evaluation, epic-design, api-design
#
# ### Engineering (12 skills)
# - autoresearch, security-auditing, skill-creation
# - mcp-server, dependency-auditing, frontend-design
# - algorithmic-art, docx, nano-pdf, code-refactoring
#
# ### C-Level Advisors (8 skills)
# - cto-advisor, cpo-advisor, cfo-advisor
# - chief-security-officer, chief-data-officer
#
# ### Product Team (10 skills)
# - product-manager, product-analyst, ux-researcher
# - product-strategy, roadmap-planning, user-story-writer
#
# ### Business Growth (4 skills)
# - customer-success-manager, revenue-ops
# - growth-strategist, partnership-manager
#
# ### Regulatory & QM (12 skills)
# - regulatory-affairs-head, gdpr-dsgvo-expert
# - iso-27001-auditor, hipaa-compliance, soc2-auditor
#
# ### Finance (2 skills)
# - financial-analyst, investment-advisor
#
# ### Documentation (6 skills)
# - technical-writer, api-documentation
# - readme-generator, changelog-writer
#
# ### Design (8 skills)
# - ui-designer, ux-designer, design-system
# - accessibility-auditor, design-reviewer
#
# ### Project Management (6 skills)
# - scrum-master, senior-pm, agile-coach
# - sprint-planner, retrospective-facilitator
#
# ## Installation
# /plugin install marketing-skills@claude-code-skills
# /plugin install engineering-skills@claude-code-skills
# /plugin install engineering-advanced-skills@claude-code-skills
# /plugin install product-skills@claude-code-skills
# /plugin install c-level-skills@claude-code-skills
# /plugin install pm-skills@claude-code-skills
# /plugin install ra-qm-skills@claude-code-skills
# /plugin install business-growth-skills@claude-code-skills
# /plugin install finance-skills@claude-code-skills
"""

# ---------------------------------------------------------------------------
# luokai0 / ai-agent-skills-by-luo-kai
# ---------------------------------------------------------------------------

LUOKAI_MEGA_COLLECTION = """\
# luokai0/ai-agent-skills-by-luo-kai — Mega Collection
# =====================================================
# Source: luokai0/ai-agent-skills-by-luo-kai (GitHub)
# Skills: 8,966+  |  Domains: 21
#
# The world's largest open collection of AI agent skills with real working
# code, aggregated from 25+ major registries.
#
# ## Domains Covered
#
# 01. Software Development & Engineering
#     - Code generation, review, refactoring, testing
#     - CI/CD pipeline design, deployment automation
#     - Architecture patterns, system design
#
# 02. Data & Analytics
#     - Data cleaning, transformation, analysis
#     - Statistical modeling, A/B testing
#     - Data visualization, reporting
#
# 03. AI & Machine Learning
#     - Model training, fine-tuning, evaluation
#     - Prompt engineering, RAG pipelines
#     - Agent orchestration, tool use
#
# 04. Web Development
#     - Frontend frameworks (React, Vue, Svelte)
#     - Backend APIs (FastAPI, Express, Django)
#     - Full-stack integration patterns
#
# 05. DevOps & Infrastructure
#     - Container orchestration (Docker, K8s)
#     - Infrastructure as Code (Terraform, Pulumi)
#     - Monitoring, logging, alerting
#
# 06. Security
#     - Vulnerability scanning, penetration testing
#     - Code security audit, dependency checking
#     - Secrets management, access control
#
# 07. Cloud Computing
#     - AWS, GCP, Azure deployment patterns
#     - Serverless architectures
#     - Multi-cloud strategies
#
# 08. Database
#     - Schema design, migration, optimization
#     - SQL/NoSQL query patterns
#     - Data warehousing, ETL pipelines
#
# 09. Mobile Development
#     - iOS (Swift), Android (Kotlin), React Native
#     - Cross-platform patterns (Flutter, Expo)
#     - App store deployment
#
# 10. Documentation
#     - Technical writing, API docs
#     - README generation, changelog automation
#     - Knowledge base management
#
# 11. Testing & QA
#     - Unit, integration, e2e testing
#     - Test generation, coverage analysis
#     - Performance and load testing
#
# 12. Product Management
#     - Roadmap planning, user stories
#     - Sprint management, retrospectives
#     - Product analytics, feature prioritization
#
# 13. Design & UX
#     - UI/UX design patterns
#     - Accessibility auditing
#     - Design system management
#
# 14. Marketing & Growth
#     - SEO, content strategy, social media
#     - Email campaigns, conversion optimization
#     - Analytics and attribution
#
# 15. Finance & Business
#     - Financial modeling, forecasting
#     - Budget planning, cost optimization
#     - Compliance and regulatory
#
# 16. Legal & Compliance
#     - Contract review, GDPR compliance
#     - Privacy policy generation
#     - Terms of service drafting
#
# 17. Education & Training
#     - Course creation, tutorial generation
#     - Interactive learning content
#     - Assessment and quiz design
#
# 18. Research & Analysis
#     - Literature review, data collection
#     - Hypothesis testing, peer review
#     - Research paper writing
#
# 19. Automation & Workflows
#     - Task automation, batch processing
#     - API integration, webhook handling
#     - Scheduled job management
#
# 20. Content Creation
#     - Blog writing, video scripting
#     - Social media content, newsletters
#     - Podcast production
#
# 21. Operations & Support
#     - Customer support automation
#     - Incident response, on-call procedures
#     - Service desk management
#
# ## How to Use
# Each skill is a folder with SKILL.md + optional scripts/ and references/.
# Copy the skill into your agent's skills directory and restart the agent.
"""

# ---------------------------------------------------------------------------
# seb1n / awesome-ai-agent-skills
# ---------------------------------------------------------------------------

SEB1N_UNIVERSAL_SKILLS = """\
# seb1n/awesome-ai-agent-skills — Universal Ready-to-Use Skills
# ==============================================================
# Source: seb1n/awesome-ai-agent-skills (GitHub)
# Skills: 90+  |  Format: Self-contained SKILL.md
#
# A collection of 90+ self-contained, universal SKILL.md skills for
# autonomous agents. Not a link directory — each skill is a complete,
# ready-to-use instruction set with examples and verification steps.
#
# ## Skill Categories
#
# ### Code Quality & Review
# - code-review: Structured peer review with checklist
# - refactoring: Safe refactoring patterns and verification
# - code-smells: Identify and fix common code smells
# - technical-debt: Assess and prioritize debt reduction
#
# ### Testing
# - test-generation: Generate comprehensive test suites
# - test-strategy: Design testing approaches for features
# - mutation-testing: Verify test quality with mutations
# - property-testing: Property-based test design
#
# ### Security
# - security-audit: Comprehensive security review
# - vulnerability-scan: Automated vulnerability detection
# - dependency-check: Audit third-party dependencies
# - secrets-scan: Detect leaked secrets in code
#
# ### Documentation
# - api-docs: Generate API documentation from code
# - readme: Create comprehensive README files
# - architecture-docs: Document system architecture
# - changelog: Automated changelog generation
#
# ### DevOps
# - ci-cd-pipeline: Design and optimize CI/CD pipelines
# - docker-optimization: Optimize Docker images
# - k8s-deployment: Kubernetes deployment patterns
# - infrastructure-as-code: IaC best practices
#
# ### Data & Analytics
# - data-cleaning: Data quality assessment and cleaning
# - data-pipeline: ETL pipeline design
# - sql-optimization: Query performance tuning
# - data-visualization: Chart and dashboard design
#
# ### AI & ML
# - model-evaluation: ML model assessment
# - prompt-engineering: Advanced prompt design
# - rag-pipeline: Retrieval-augmented generation setup
# - fine-tuning: Model fine-tuning workflows
#
# ### Project Management
# - sprint-planning: Sprint scope and task estimation
# - risk-assessment: Project risk identification
# - retrospective: Agile retrospective facilitation
# - roadmap: Product roadmap creation
#
# ### Content & Communication
# - technical-writing: Clear technical documentation
# - blog-post: SEO-optimized blog content
# - email-campaign: Marketing email sequences
# - presentation: Slide deck creation
#
# ## Skill Structure
# Each SKILL.md contains:
# 1. Frontmatter: name, description, triggers
# 2. Instructions: step-by-step guidance
# 3. Examples: concrete before/after examples
# 4. Verification: how to confirm the skill worked
# 5. Edge cases: known limitations and handling
#
# ## Platform Compatibility
# Claude Code, Codex, Gemini CLI, Cursor, Copilot, Windsurf, OpenCode
"""

# ---------------------------------------------------------------------------
# mouadja02 / skills
# ---------------------------------------------------------------------------

MOUADJA_CATEGORIZED_SKILLS = """\
# mouadja02/skills — 716 Categorized Agent Skills
# ================================================
# Source: mouadja02/skills (GitHub)
# Skills: 716  |  Categories: 31
#
# A curated collection of 716 AI agent skills across 31 categories,
# each as a reusable instruction package with YAML frontmatter and
# an agent-readable body in SKILL.md files.
#
# ## Categories (31 total)
#
# 1.  Code Generation & Completion
# 2.  Code Review & Quality
# 3.  Testing & QA
# 4.  Debugging & Troubleshooting
# 5.  Refactoring & Optimization
# 6.  Documentation Generation
# 7.  API Design & Development
# 8.  Database Design & Queries
# 9.  Security & Vulnerability Analysis
# 10. DevOps & Deployment
# 11. Cloud Infrastructure
# 12. Containerization & Orchestration
# 13. CI/CD Pipelines
# 14. Monitoring & Logging
# 15. Performance Optimization
# 16. Data Analysis & Visualization
# 17. Machine Learning & AI
# 18. Natural Language Processing
# 19. Computer Vision
# 20. Web Scraping & Automation
# 21. File Processing & Conversion
# 22. Text Processing & NLP
# 23. Image Processing
# 24. Audio & Video Processing
# 25. Configuration Management
# 26. Project Planning & Management
# 27. Technical Writing
# 28. Research & Analysis
# 29. UI/UX Design Assistance
# 30. Code Translation & Migration
# 31. Error Handling & Recovery
#
# ## Skill Format
# Each skill follows the SKILL.md open standard:
# ---
# name: skill-name
# description: Brief description of what the skill does
# category: category-name
# triggers:
#   - "user asks to..."
#   - "when working with..."
# ---
# # Skill Name
# ## Instructions
# [Step-by-step guidance]
# ## Examples
# [Concrete examples]
# ## Verification
# [How to confirm success]
#
# ## Usage
# Copy skill folders into your agent platform's skills directory.
# Compatible with Claude Code, Codex, Gemini CLI, Cursor, and more.
"""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SKILLS = {
    "voltagent_awesome_skills": VOLTAGENT_AWESOME_SKILLS,
    "claude_community_skills": CLAUDE_COMMUNITY_SKILLS,
    "luokai_mega_collection": LUOKAI_MEGA_COLLECTION,
    "seb1n_universal_skills": SEB1N_UNIVERSAL_SKILLS,
    "mouadja_categorized_skills": MOUADJA_CATEGORIZED_SKILLS,
}

__all__ = [
    "VOLTAGENT_AWESOME_SKILLS",
    "CLAUDE_COMMUNITY_SKILLS",
    "LUOKAI_MEGA_COLLECTION",
    "SEB1N_UNIVERSAL_SKILLS",
    "MOUADJA_CATEGORIZED_SKILLS",
    "SKILLS",
]
