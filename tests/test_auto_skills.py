\
"""
Tests for ilovetools.skills subpackage.
"""

import pytest
from ilovetools.skills import (
    list_skills,
    get_skill,
    skill_info,
    karpathy_skills,
    anthropic_skills,
    agent_skills,
    prompt_engineering_skills,
    community_skills,
    microsoft_skills,
    openai_skills,
    awesome_agent_skills,
)


class TestSkillModulesImport:
    def test_karpathy_skills_importable(self):
        assert karpathy_skills is not None
        assert hasattr(karpathy_skills, "SKILLS")
        assert isinstance(karpathy_skills.SKILLS, dict)
        assert len(karpathy_skills.SKILLS) >= 5

    def test_anthropic_skills_importable(self):
        assert anthropic_skills is not None
        assert hasattr(anthropic_skills, "SKILLS")
        assert isinstance(anthropic_skills.SKILLS, dict)
        assert len(anthropic_skills.SKILLS) >= 3

    def test_agent_skills_importable(self):
        assert agent_skills is not None
        assert hasattr(agent_skills, "SKILLS")
        assert isinstance(agent_skills.SKILLS, dict)
        assert len(agent_skills.SKILLS) >= 5

    def test_prompt_engineering_skills_importable(self):
        assert prompt_engineering_skills is not None
        assert hasattr(prompt_engineering_skills, "SKILLS")
        assert isinstance(prompt_engineering_skills.SKILLS, dict)
        assert len(prompt_engineering_skills.SKILLS) >= 3

    def test_community_skills_importable(self):
        assert community_skills is not None
        assert hasattr(community_skills, "SKILLS")
        assert isinstance(community_skills.SKILLS, dict)
        assert len(community_skills.SKILLS) >= 5

    def test_microsoft_skills_importable(self):
        assert microsoft_skills is not None
        assert hasattr(microsoft_skills, "SKILLS")
        assert isinstance(microsoft_skills.SKILLS, dict)
        assert len(microsoft_skills.SKILLS) >= 4

    def test_openai_skills_importable(self):
        assert openai_skills is not None
        assert hasattr(openai_skills, "SKILLS")
        assert isinstance(openai_skills.SKILLS, dict)
        assert len(openai_skills.SKILLS) >= 5

    def test_awesome_agent_skills_importable(self):
        assert awesome_agent_skills is not None
        assert hasattr(awesome_agent_skills, "SKILLS")
        assert isinstance(awesome_agent_skills.SKILLS, dict)
        assert len(awesome_agent_skills.SKILLS) >= 6


class TestListSkills:
    def test_returns_list(self):
        result = list_skills()
        assert isinstance(result, list)

    def test_returns_sorted(self):
        result = list_skills()
        assert result == sorted(result)

    def test_all_strings(self):
        for name in list_skills():
            assert isinstance(name, str)
            assert len(name) > 0

    def test_contains_expected_skills(self):
        names = list_skills()
        expected = [
            "karpathy_coding_guidelines",
            "karpathy_nanoGPT",
            "karpathy_llm_c",
            "karpathy_nn_zero_to_hero",
            "karpathy_micrograd",
            "anthropic_skill_creator",
            "anthropic_pdf",
            "anthropic_code_review",
            "agent_skill_spec",
            "caveman_token_efficiency",
            "prompt_master",
            "codex_skills",
            "engineering_skills",
            "prompt_frameworks",
            "anti_hallucination",
            "production_prompt_structure",
            "voltagent_awesome_skills",
            "claude_community_skills",
            "luokai_mega_collection",
            "seb1n_universal_skills",
            "mouadja_categorized_skills",
            "microsoft_azure_skills",
            "microsoft_foundry_skills",
            "microsoft_agents_template",
            "microsoft_mcp_config",
        ]
        for skill in expected:
            assert skill in names, f"Missing skill: {skill}"

    def test_at_least_10_skills(self):
        assert len(list_skills()) >= 10

    def test_no_duplicates(self):
        names = list_skills()
        assert len(names) == len(set(names))

    def test_at_least_24_skills(self):
        assert len(list_skills()) >= 24


class TestGetSkill:
    @pytest.mark.parametrize("skill_name", [
        "karpathy_coding_guidelines",
        "karpathy_nanoGPT",
        "karpathy_llm_c",
        "karpathy_nn_zero_to_hero",
        "karpathy_micrograd",
        "anthropic_skill_creator",
        "anthropic_pdf",
        "anthropic_code_review",
        "agent_skill_spec",
        "caveman_token_efficiency",
        "prompt_master",
        "codex_skills",
        "engineering_skills",
        "prompt_frameworks",
        "anti_hallucination",
        "production_prompt_structure",
        "voltagent_awesome_skills",
        "claude_community_skills",
        "luokai_mega_collection",
        "seb1n_universal_skills",
        "mouadja_categorized_skills",
        "microsoft_azure_skills",
        "microsoft_foundry_skills",
        "microsoft_agents_template",
        "microsoft_mcp_config",
    ])
    def test_get_skill_returns_nonempty_string(self, skill_name):
        result = get_skill(skill_name)
        assert isinstance(result, str)
        assert len(result) > 50

    def test_get_skill_unknown_raises_keyerror(self):
        with pytest.raises(KeyError, match="nonexistent_skill"):
            get_skill("nonexistent_skill")

    def test_get_skill_empty_string_raises_keyerror(self):
        with pytest.raises(KeyError):
            get_skill("")

    def test_get_skill_content_has_expected_keywords(self):
        assert "Think Before Coding" in get_skill("karpathy_coding_guidelines")
        assert "nanoGPT" in get_skill("karpathy_nanoGPT")
        assert "llm.c" in get_skill("karpathy_llm_c")
        assert "micrograd" in get_skill("karpathy_micrograd")
        assert "SKILL.md" in get_skill("anthropic_skill_creator")
        assert "pdf" in get_skill("anthropic_pdf").lower()
        assert "token" in get_skill("caveman_token_efficiency").lower()
        assert "CO-STAR" in get_skill("prompt_frameworks")
        assert "hallucination" in get_skill("anti_hallucination").lower()
        assert "skill" in get_skill("voltagent_awesome_skills").lower()
        assert "claude" in get_skill("claude_community_skills").lower()
        assert "domain" in get_skill("luokai_mega_collection").lower()
        assert "universal" in get_skill("seb1n_universal_skills").lower()
        assert "category" in get_skill("mouadja_categorized_skills").lower()
        assert "Azure" in get_skill("microsoft_azure_skills")
        assert "Foundry" in get_skill("microsoft_foundry_skills")
        assert "AGENTS.md" in get_skill("microsoft_agents_template")
        assert "MCP" in get_skill("microsoft_mcp_config")

    def test_get_skill_consistent_with_list(self):
        for name in list_skills():
            result = get_skill(name)
            assert isinstance(result, str)
            assert len(result) > 0


class TestSkillInfo:
    def test_returns_dict(self):
        info = skill_info()
        assert isinstance(info, dict)

    def test_has_entry_for_every_skill(self):
        names = set(list_skills())
        info_keys = set(skill_info().keys())
        assert names == info_keys, f"Mismatch: {names ^ info_keys}"

    def test_each_entry_has_source_and_description(self):
        info = skill_info()
        for name, meta in info.items():
            assert "source" in meta, f"Missing 'source' for {name}"
            assert "description" in meta, f"Missing 'description' for {name}"
            assert isinstance(meta["source"], str)
            assert isinstance(meta["description"], str)
            assert len(meta["source"]) > 0
            assert len(meta["description"]) > 0


class TestSkillsFromRootImport:
    def test_import_from_root(self):
        import ilovetools
        assert hasattr(ilovetools, "skills")
        assert "skills" in ilovetools.__all__

    def test_list_skills_from_root(self):
        import ilovetools
        names = ilovetools.skills.list_skills()
        assert len(names) >= 10


class TestNewSkillContent:
    """Test content of newly added openai_skills and awesome_agent_skills."""

    def test_openai_codex_skills_content(self):
        text = get_skill("openai_codex_skills")
        assert "Codex" in text
        assert "skill" in text.lower()

    def test_openai_plugin_guide_content(self):
        text = get_skill("openai_plugin_guide")
        assert "plugin" in text.lower()
        assert "skill" in text.lower()

    def test_openai_agent_skills_spec_content(self):
        text = get_skill("openai_agent_skills_spec")
        assert "SKILL.md" in text
        assert "frontmatter" in text.lower()

    def test_openai_code_review_content(self):
        text = get_skill("openai_code_review_skill")
        assert "review" in text.lower()
        assert "security" in text.lower()

    def test_openai_testing_skill_content(self):
        text = get_skill("openai_testing_skill")
        assert "pytest" in text.lower()
        assert "test" in text.lower()

    def test_openai_debugging_skill_content(self):
        text = get_skill("openai_debugging_skill")
        assert "debug" in text.lower()
        assert "traceback" in text.lower()

    def test_awesome_git_workflow_content(self):
        text = get_skill("awesome_git_workflow")
        assert "Git" in text
        assert "commit" in text.lower()

    def test_awesome_refactoring_content(self):
        text = get_skill("awesome_refactoring")
        assert "refactor" in text.lower()

    def test_awesome_api_design_content(self):
        text = get_skill("awesome_api_design")
        assert "REST" in text or "API" in text

    def test_awesome_error_handling_content(self):
        text = get_skill("awesome_error_handling")
        assert "error" in text.lower()
        assert "exception" in text.lower()

    def test_awesome_doc_generation_content(self):
        text = get_skill("awesome_doc_generation")
        assert "doc" in text.lower()

    def test_awesome_performance_profiling_content(self):
        text = get_skill("awesome_performance_profiling")
        assert "profil" in text.lower() or "performance" in text.lower()

    def test_new_skills_in_skill_info(self):
        info = skill_info()
        for name in ["openai_codex_skills", "awesome_git_workflow"]:
            assert name in info
            assert "source" in info[name]
            assert "description" in info[name]

    def test_new_skills_count_increase(self):
        """Total skills should now be significantly more than the original set."""
        names = list_skills()
        assert len(names) >= 30
