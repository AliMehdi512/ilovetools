"""
Microsoft agent-skills — distilled from the microsoft/agent-skills repository.

This module packages skill content from Microsoft's public agent-skills
repository (https://github.com/microsoft/agent-skills, ~2.6k stars, MIT)
into readable Python string constants that LLMs can load at runtime.

The repository provides 174+ domain-specific skills tied to Azure SDK
and Microsoft Foundry development, including Azure deployment workflows,
MCP configurations, and multi-harness agent templates.

Usage
-----
>>> from ilovetools.skills import microsoft_skills
>>> "Azure" in microsoft_skills.MICROSOFT_AZURE_SKILLS
True
>>> "Foundry" in microsoft_skills.MICROSOFT_FOUNDRY_SKILLS
True
>>> "AGENTS.md" in microsoft_skills.MICROSOFT_AGENTS_TEMPLATE
True
>>> "MCP" in microsoft_skills.MICROSOFT_MCP_CONFIG
True
"""

# ---------------------------------------------------------------------------
# Microsoft Azure Skills — deployment, validation, diagnostics
# ---------------------------------------------------------------------------

MICROSOFT_AZURE_SKILLS = """\
# Microsoft Azure Agent Skills
# ============================
# Source: https://github.com/microsoft/agent-skills (MIT)
# Stars: ~2,647 | Skills: 174+
#
# Azure-specific agent skills for deployment, validation, diagnostics,
# cost management, and resource lookup workflows.
#
# ## Skill Categories
#
# ### azure-prepare
# Pre-flight checks: validate subscription, resource provider registration,
# quota availability, and naming conventions before deployment.
#
# ### azure-validate
# Validate ARM/Bicep templates for schema compliance, best practices,
# and security baselines before submitting a deployment.
#
# ### azure-deploy
# Execute infrastructure-as-code deployments via az CLI or ARM templates.
# Handles resource group creation, parameter resolution, and deployment
# tracking with rollback on failure.
#
# ### azure-upgrade
# In-place upgrade workflows for Azure resources (AKS clusters, App Service
# plans, databases) with compatibility checks and staged rollout.
#
# ### azure-diagnostics
# Diagnostic skill that collects logs, metrics, and health probes for
# Azure resources. Generates a structured diagnostic report with
# recommended remediation steps.
#
# ### azure-cost
# Cost analysis skill: queries Azure Cost Management APIs, identifies
# underutilised resources, and suggests optimisation actions (right-size,
# reserved instances, auto-shutdown schedules).
#
# ### azure-resource-lookup
# Search and discover Azure resources by name, tag, type, or resource
# group. Returns structured metadata including location, SKU, and tags.
#
# ## Agent Use-Case
# When an agent is working with Azure infrastructure, these skills provide
# structured workflows that enforce guardrails (validation before deploy,
# diagnostics on failure, cost checks before provisioning).
"""

# ---------------------------------------------------------------------------
# Microsoft Foundry Skills — agent hosting and orchestration
# ---------------------------------------------------------------------------

MICROSOFT_FOUNDRY_SKILLS = """\
# Microsoft Foundry Agent Skills
# ===============================
# Source: https://github.com/microsoft/agent-skills (MIT)
#
# Microsoft Foundry provides hosted agent infrastructure with built-in
# toolboxes, MCP endpoints, and resource management.
#
# ## foundry-hosted-agents
# Provision and manage hosted agent infrastructure on Microsoft Foundry.
# Includes configuration for compute, memory, and tool access policies.
#
# ## foundry-router
# Intent-to-skill router that maps user queries to the appropriate
# sub-skills or discovery surfaces. Integrates with Foundry MCP for
# real-time tool execution.
#
# ## foundry-mcp
# MCP (Managed Cloud Platform) configuration for Foundry workflows.
# Provides endpoints for code execution, AI search, file search, and
# OpenAPI tool integration.
#
# ## Multi-Harness Support (APM)
# Cross-platform agent installation support for GitHub Copilot,
# Claude Code, Cursor, and other coding agents via a unified APM
# (Agent Package Manager) interface.
#
# ## Agent Use-Case
# When an agent needs to deploy or manage Azure-hosted AI agents,
# Foundry skills provide the infrastructure layer: provisioning,
# routing, MCP configuration, and cross-platform compatibility.
"""

# ---------------------------------------------------------------------------
# AGENTS.md Template — Microsoft's agent configuration format
# ---------------------------------------------------------------------------

MICROSOFT_AGENTS_TEMPLATE = """\
# AGENTS.md — Microsoft Agent Configuration Template
# ===================================================
# Source: https://github.com/microsoft/agent-skills (MIT)
#
# AGENTS.md is a project-level configuration file that defines agent
# behaviour, available skills, and integration points.
#
# ## Structure
#
# ```markdown
# # Agent: <name>
#
# ## Description
# <what this agent does>
#
# ## Skills
# - skill-1: <description>
# - skill-2: <description>
#
# ## Tools
# - tool-1: <description>
#
# ## Constraints
# - <what the agent must not do>
#
# ## MCP Endpoints
# - endpoint-1: <url>
# ```
#
# ## Best Practices
# 1. Keep AGENTS.md at the repository root for automatic discovery.
# 2. List skills with brief descriptions so the agent can self-select.
# 3. Define constraints explicitly to prevent scope creep.
# 4. Version the file alongside code changes.
# 5. Use it alongside CLAUDE.md / SKILL.md for layered configuration.
#
# ## Agent Use-Case
# When setting up a new agent project, copy this template to AGENTS.md
# at the repo root and fill in the sections. The agent runtime reads
# this file at startup to configure its behaviour.
"""

# ---------------------------------------------------------------------------
# MCP Configuration — Azure and Foundry MCP endpoints
# ---------------------------------------------------------------------------

MICROSOFT_MCP_CONFIG = """\
# Microsoft MCP Configuration
# =============================
# Source: https://github.com/microsoft/agent-skills (MIT)
#
# MCP (Model Context Protocol) configurations for Azure and Foundry
# integrations. These configs enable agents to interact with Azure
# resources and Foundry services via structured tool calls.
#
# ## Azure MCP Server
# Provides live Azure tooling for actionable operations:
# - Resource management (create, read, update, delete)
# - Subscription and tenant queries
# - Log analytics and diagnostic queries
# - Cost management and billing queries
#
# ## Foundry MCP
# Microsoft Foundry workflows via MCP:
# - Agent provisioning and lifecycle management
# - Toolbox configuration (available tools per agent)
# - Code execution sandbox
# - AI Search integration for document retrieval
# - File Search for workspace-scoped file discovery
#
# ## Configuration Example (.mcp.json)
# ```json
# {
#   "mcpServers": {
#     "azure": {
#       "command": "npx",
#       "args": ["-y", "@azure/mcp-server"],
#       "env": {
#         "AZURE_SUBSCRIPTION_ID": "<subscription-id>"
#       }
#     },
#     "foundry": {
#       "command": "npx",
#       "args": ["-y", "@foundry/mcp-server"],
#       "env": {
#         "FOUNDRY_ENDPOINT": "<endpoint-url>"
#       }
#     }
#   }
# }
# ```
#
# ## Agent Use-Case
# When an agent needs to interact with Azure or Foundry, load this
# configuration to understand available MCP endpoints and their
# capabilities. The agent can then make structured tool calls to
# manage cloud resources, run code, or search documents.
"""

# ---------------------------------------------------------------------------
# Aggregate dict for easy programmatic access
# ---------------------------------------------------------------------------

SKILLS = {
    "microsoft_azure_skills": MICROSOFT_AZURE_SKILLS,
    "microsoft_foundry_skills": MICROSOFT_FOUNDRY_SKILLS,
    "microsoft_agents_template": MICROSOFT_AGENTS_TEMPLATE,
    "microsoft_mcp_config": MICROSOFT_MCP_CONFIG,
}

__all__ = [
    "MICROSOFT_AZURE_SKILLS",
    "MICROSOFT_FOUNDRY_SKILLS",
    "MICROSOFT_AGENTS_TEMPLATE",
    "MICROSOFT_MCP_CONFIG",
    "SKILLS",
]
