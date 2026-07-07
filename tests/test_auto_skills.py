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
        ]
        for skill in expected:
            assert skill in names, f"Missing skill: {skill}"

    def test_at_least_10_skills(self):
        assert len(list_skills()) >= 10

    def test_no_duplicates(self):
        names = list_skills()
        assert len(names) == len(set(names))


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
