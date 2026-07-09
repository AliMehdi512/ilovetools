"""
Curated awesome agent skills from the community.

This module packages the most practical and widely-used agent skills
from the top community skill repositories on GitHub, including:

* **addyosmani/agent-skills** (~24 production engineering skills)
* **VoltAgent/awesome-agent-skills** (1000+ curated skills)
* **alirezarezvani/claude-skills** (330+ skills, 70+ commands)
* **mouadja02/skills** (725 skills across 35 categories)

Each skill below is a distilled, high-impact instruction set that
LLM agents can load at runtime as system-prompt context.

Included skills
---------------
* **awesome_git_workflow** — Git branching, commit, and PR workflow skill.
* **awesome_refactoring** — Safe code refactoring patterns and checklist.
* **awesome_api_design** — REST/GraphQL API design guidelines.
* **awesome_error_handling** — Comprehensive error handling patterns.
* **awesome_doc_generation** — Documentation generation skill.
* **awesome_performance_profiling** — Performance profiling and optimization.

Usage
-----
>>> from ilovetools.skills import awesome_agent_skills
>>> "Git" in awesome_agent_skills.AWESOME_GIT_WORKFLOW
True
>>> "refactor" in awesome_agent_skills.AWESOME_REFACTORING.lower()
True
>>> isinstance(awesome_agent_skills.SKILLS, dict)
True
>>> len(awesome_agent_skills.SKILLS) >= 6
True
"""

AWESOME_GIT_WORKFLOW = """\
# Git Workflow Skill for AI Agents
# ==================================
# Source: addyosmani/agent-skills + community best practices
#
# This skill guides an AI agent through proper Git workflows including
# branching, committing, and pull request creation.
#
# ## Branching Strategy
# - main/master: Always deployable. Never commit directly.
# - develop: Integration branch for features.
# - feature/<name>: New features (branched from develop).
# - fix/<name>: Bug fixes (branched from develop or main).
# - hotfix/<name>: Urgent production fixes (branched from main).
# - release/<version>: Release preparation.
#
# ## Commit Conventions (Conventional Commits)
# Format: type(scope): description
#
# Types:
#   feat:     New feature
#   fix:      Bug fix
#   docs:     Documentation only
#   style:    Formatting (no code change)
#   refactor: Code restructuring (no feature/fix)
#   perf:     Performance improvement
#   test:     Adding/updating tests
#   chore:    Build/tooling/maintenance
#   ci:       CI/CD changes
#
# Rules:
#   - Use imperative mood: "add" not "added"
#   - Keep subject line <= 72 characters
#   - Body: explain what and why (not how)
#   - Reference issues: "Closes #123", "Refs #456"
#
# ## Pull Request Process
# 1. Rebase on target branch before opening PR.
# 2. Write a clear PR title and description.
# 3. Link related issues.
# 4. Ensure CI passes.
# 5. Request review from relevant team members.
# 6. Address review feedback with new commits (not force-push).
# 7. Squash-merge if many small commits.
#
# ## Agent Behavior
# - Always check `git status` before operations.
# - Never force-push to shared branches (main, develop).
# - Write meaningful commit messages.
# - Run tests before committing.
# - Keep PRs small and focused (< 400 lines when possible).
"""

AWESOME_REFACTORING = """\
# Safe Code Refactoring Skill for AI Agents
# ===========================================
# Source: addyosmani/agent-skills + VoltAgent/awesome-agent-skills
#
# This skill provides a systematic refactoring workflow that preserves
# behavior while improving code structure.
#
# ## Refactoring Principles
# 1. **Never refactor without tests.** Ensure existing tests pass first.
# 2. **Small steps.** Each refactoring should be a single, reviewable commit.
# 3. **Preserve behavior.** The public API must not change.
# 4. **One change at a time.** Don't mix refactoring with feature work.
# 5. **Run tests after each step.** Catch regressions immediately.
#
# ## Common Refactoring Patterns
#
# ### Extract Function
# When a function is too long or has a clear sub-task:
#   1. Identify the code block to extract.
#   2. Create a new function with a descriptive name.
#   3. Pass needed variables as parameters.
#   4. Replace the original code with a function call.
#   5. Run tests.
#
# ### Extract Class
# When a class has too many responsibilities:
#   1. Identify a cohesive set of methods and attributes.
#   2. Create a new class for that responsibility.
#   3. Move the methods and attributes.
#   4. Update references in the original class.
#   5. Run tests.
#
# ### Replace Conditional with Polymorphism
# When you have type-checking conditionals:
#   1. Create a base class with an abstract method.
#   2. Create subclasses for each type.
#   3. Move the conditional logic into each subclass.
#   4. Replace conditional with a method call on the base class.
#   5. Run tests.
#
# ### Introduce Parameter Object
# When a function takes too many parameters:
#   1. Create a dataclass/NamedTuple for the parameters.
#   2. Replace individual parameters with the object.
#   3. Update all call sites.
#   4. Run tests.
#
# ## Code Smells to Watch For
# - Long methods (> 30 lines)
# - Large classes (> 200 lines or > 10 methods)
# - Duplicate code blocks
# - Deep nesting (> 3 levels)
# - Long parameter lists (> 4 params)
# - Feature envy (method uses another class more than its own)
# - Data clumps (same group of data passed together everywhere)
#
# ## Agent Behavior
# - Always run tests before AND after refactoring.
# - Make one refactoring change per commit.
# - Use descriptive commit messages: "refactor: extract X from Y".
# - Never change behavior during refactoring.
# - If tests fail after refactoring, revert and try a smaller step.
"""

AWESOME_API_DESIGN = """\
# API Design Guidelines Skill for AI Agents
# ===========================================
# Source: VoltAgent/awesome-agent-skills + community standards
#
# This skill provides REST and GraphQL API design best practices
# for AI agents building or reviewing APIs.
#
# ## REST API Design
#
# ### Resource Naming
# - Use nouns, not verbs: /users not /getUsers
# - Plural for collections: /users, /orders
# - Use sub-resources for relationships: /users/{id}/orders
# - Keep URLs lowercase with hyphens: /user-profiles
#
# ### HTTP Methods
# - GET: Read (safe, idempotent, cacheable)
# - POST: Create (not idempotent)
# - PUT: Full update (idempotent)
# - PATCH: Partial update (not guaranteed idempotent)
# - DELETE: Remove (idempotent)
#
# ### Status Codes
# - 200: OK (successful GET/PUT/PATCH)
# - 201: Created (successful POST)
# - 204: No Content (successful DELETE)
# - 400: Bad Request (client error)
# - 401: Unauthorized (not authenticated)
# - 403: Forbidden (authenticated but not allowed)
# - 404: Not Found
# - 409: Conflict (duplicate resource)
# - 422: Unprocessable Entity (validation error)
# - 429: Too Many Requests (rate limited)
# - 500: Internal Server Error
#
# ### Response Format
#   {
#     "data": { ... },
#     "meta": {
#       "page": 1,
#       "per_page": 20,
#       "total": 100
#     },
#     "errors": [
#       {
#         "code": "VALIDATION_ERROR",
#         "message": "Email is required",
#         "field": "email"
#       }
#     ]
#   }
#
# ### Versioning
# - Use URL versioning: /v1/users
# - Or header versioning: Accept: application/vnd.api.v1+json
# - Never make breaking changes without a new version.
#
# ## GraphQL API Design
# - Use camelCase for field names.
# - Provide descriptions for every type and field.
# - Use connections for paginated lists (Relay spec).
# - Implement input types for mutations.
# - Use enums for fixed-value fields.
# - Deprecate fields with @deprecated directive.
#
# ## Security
# - Always use HTTPS.
# - Authenticate with JWT or OAuth 2.0.
# - Validate all input on the server.
# - Rate-limit public endpoints.
# - Use CORS headers appropriately.
# - Never expose internal IDs (use UUIDs or slugs).
#
# ## Agent Behavior
# - Design APIs that are consistent and predictable.
# - Document every endpoint with examples.
# - Include error responses in documentation.
# - Think about pagination from the start.
# - Consider caching strategy for GET endpoints.
"""

AWESOME_ERROR_HANDLING = """\
# Comprehensive Error Handling Skill for AI Agents
# ==================================================
# Source: addyosmani/agent-skills + alirezarezvani/claude-skills
#
# This skill provides patterns for robust error handling in Python
# applications.
#
# ## Core Principles
# 1. **Fail fast.** Detect errors as early as possible.
# 2. **Be specific.** Catch specific exceptions, not bare `except:`.
# 3. **Don't swallow.** Never catch an exception and do nothing.
# 4. **Provide context.** Include relevant data in error messages.
# 5. **Clean up.** Use context managers (with) for resource management.
# 6. **Log, don't print.** Use logging, not print() for errors.
#
# ## Patterns
#
# ### Specific Exception Catching
#   # BAD
#   try:
#       result = do_something()
#   except:
#       pass
#
#   # GOOD
#   try:
#       result = do_something()
#   except FileNotFoundError as e:
#       logger.error(f"Config file missing: {e}")
#       raise
#   except json.JSONDecodeError as e:
#       logger.error(f"Invalid JSON in config: {e}")
#       raise
#
# ### Custom Exceptions
#   class AppError(Exception):
#       '''Base exception for the application.'''
#       pass
#
#   class ValidationError(AppError):
#       def __init__(self, field: str, message: str):
#           self.field = field
#           self.message = message
#           super().__init__(f"{field}: {message}")
#
#   class NotFoundError(AppError):
#       def __init__(self, resource: str, identifier: str):
#           self.resource = resource
#           self.identifier = identifier
#           super().__init__(f"{resource} not found: {identifier}")
#
# ### Context Managers
#   @contextmanager
#   def safe_operation(resource):
#       try:
#           resource.acquire()
#           yield resource
#       except Exception:
#           logger.exception("Operation failed")
#           raise
#       finally:
#           resource.release()
#
# ### Retry with Backoff
#   @retry(exponential_backoff(max_retries=3))
#   def fetch_data(url):
#       response = requests.get(url, timeout=10)
#       response.raise_for_status()
#       return response.json()
#
# ### Validation at Boundaries
#   def process_user(data: dict) -> User:
#       if not isinstance(data, dict):
#           raise TypeError("data must be a dict")
#       if "email" not in data:
#           raise ValidationError("email", "Email is required")
#       if not is_valid_email(data["email"]):
#           raise ValidationError("email", "Invalid email format")
#       return User(**data)
#
# ## Agent Behavior
# - Always handle specific exceptions, never bare except.
# - Create custom exception hierarchies for the application.
# - Use context managers for resource cleanup.
# - Log errors with full context (what, why, how to reproduce).
# - Re-raise exceptions after logging unless you're truly handling them.
# - Validate input at function/class boundaries.
"""

AWESOME_DOC_GENERATION = """\
# Documentation Generation Skill for AI Agents
# ==============================================
# Source: VoltAgent/awesome-agent-skills + mouadja02/skills
#
# This skill guides an AI agent in generating high-quality documentation
# for Python codebases.
#
# ## Documentation Layers
# 1. **Docstrings** - Every public function, class, and module.
# 2. **README.md** - Project overview, installation, usage, examples.
# 3. **API Reference** - Auto-generated from docstrings (Sphinx/MkDocs).
# 4. **Tutorials** - Step-by-step guides for common workflows.
# 5. **CHANGELOG.md** - Versioned list of changes.
#
# ## Docstring Standards (Google Style)
#   def process_data(data: list[dict], *, batch_size: int = 100) -> list[dict]:
#       '''Process a list of records in batches.
#
#       Args:
#           data: List of record dictionaries to process.
#           batch_size: Number of records per batch (default 100).
#
#       Returns:
#           Processed list of record dictionaries.
#
#       Raises:
#           ValueError: If data is empty or batch_size < 1.
#           TypeError: If data contains non-dict items.
#
#       Examples:
#           >>> result = process_data([{"x": 1}], batch_size=10)
#           >>> len(result)
#           1
#       '''
#
# ## README Structure
#   # Project Name
#   Brief one-line description.
#
#   ## Features
#   - Key feature 1
#   - Key feature 2
#
#   ## Installation
#   pip install project-name
#
#   ## Quick Start
#   from project import main_function
#   result = main_function("input")
#
#   ## Documentation
#   Link to full docs.
#
#   ## License
#   MIT
#
# ## Agent Behavior
# - Write docstrings for ALL public functions/classes/modules.
# - Include type hints in signatures.
# - Provide at least one usage example per public function.
# - Document all raised exceptions.
# - Keep descriptions concise but complete.
# - Update README when adding new features.
# - Use Google-style or NumPy-style docstrings consistently.
"""

AWESOME_PERFORMANCE_PROFILING = """\
# Performance Profiling and Optimization Skill for AI Agents
# ==========================================================
# Source: addyosmani/agent-skills + community best practices
#
# This skill guides an AI agent through profiling and optimizing
# Python code for performance.
#
# ## Profiling Workflow
# 1. **Measure first.** Never optimize without measuring.
# 2. **Profile to find bottlenecks.** Use cProfile or line_profiler.
# 3. **Optimize the hottest path.** Focus on the top 20% of time.
# 4. **Re-measure.** Verify the improvement.
# 5. **Benchmark.** Compare before/after with timing.
#
# ## Profiling Tools
#
# ### cProfile (built-in)
#   import cProfile
#   cProfile.run('my_function()', sort='cumulative')
#
# ### timeit (built-in)
#   import timeit
#   timeit.timeit('my_function()', number=10000, globals=globals())
#
# ### line_profiler
#   from line_profiler import LineProfiler
#   lp = LineProfiler()
#   lp.add_function(my_function)
#   lp.runcall(my_function)
#   lp.print_stats()
#
# ### memory_profiler
#   from memory_profiler import profile
#   @profile
#   def my_function():
#       ...
#
# ## Common Optimizations
#
# ### Algorithmic
# - Replace O(n^2) loops with O(n) using sets/dicts.
# - Use bisect for sorted-list insertions.
# - Use heapq for priority queues instead of sorting.
# - Cache expensive computations (functools.lru_cache).
#
# ### Data Structures
# - Use sets for membership testing (O(1) vs O(n) for lists).
# - Use deque for queue operations (O(1) appendleft/popleft).
# - Use defaultdict to avoid key-existence checks.
# - Use array.array for homogeneous numeric data.
#
# ### I/O
# - Batch database queries instead of N+1 patterns.
# - Use connection pooling for databases.
# - Read files in chunks, not all at once.
# - Use asyncio for concurrent I/O operations.
#
# ### Memory
# - Use generators instead of lists for large sequences.
# - Use __slots__ for classes with many instances.
# - Avoid unnecessary copies (use views, not copies).
# - Release large objects explicitly with del.
#
# ## Agent Behavior
# - Always profile before optimizing.
# - Focus on the hottest 20% of code.
# - Measure improvement after each change.
# - Prefer algorithmic improvements over micro-optimizations.
# - Document performance characteristics in docstrings.
# - Add performance regression tests for critical paths.
"""

SKILLS = {
    "awesome_git_workflow": AWESOME_GIT_WORKFLOW,
    "awesome_refactoring": AWESOME_REFACTORING,
    "awesome_api_design": AWESOME_API_DESIGN,
    "awesome_error_handling": AWESOME_ERROR_HANDLING,
    "awesome_doc_generation": AWESOME_DOC_GENERATION,
    "awesome_performance_profiling": AWESOME_PERFORMANCE_PROFILING,
}

SKILL_INFO = {
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
    "AWESOME_GIT_WORKFLOW",
    "AWESOME_REFACTORING",
    "AWESOME_API_DESIGN",
    "AWESOME_ERROR_HANDLING",
    "AWESOME_DOC_GENERATION",
    "AWESOME_PERFORMANCE_PROFILING",
    "SKILLS",
    "SKILL_INFO",
]
