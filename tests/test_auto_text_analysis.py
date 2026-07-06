\
"""
Tests for ilovetools.text.text_analysis module.
"""

import pytest
from ilovetools.text.text_analysis import (
    redact_secrets,
    extract_urls,
    extract_emails,
    strip_markdown,
    slugify,
    text_similarity,
    word_frequency,
    reading_time,
    extract_keywords,
    normalize_whitespace,
)


class TestRedactSecrets:
    def test_github_token(self):
        text = "token: ghp_1234567890abcdefghijklmnopqrstuvwxyz"
        result = redact_secrets(text)
        assert "ghp_" not in result
        assert "[REDACTED]" in result

    def test_openai_key(self):
        text = "key is sk-proj1234567890abcdefghij"
        result = redact_secrets(text)
        assert "sk-" not in result
        assert "[REDACTED]" in result

    def test_slack_token(self):
        text = "xoxb-1234567890-abcdefghij"
        result = redact_secrets(text)
        assert "xoxb-" not in result

    def test_bearer_token(self):
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        result = redact_secrets(text)
        assert "Bearer" not in result or "[REDACTED]" in result

    def test_generic_key_value(self):
        text = "api_key=supersecretkey1234567890"
        result = redact_secrets(text)
        assert "supersecretkey" not in result

    def test_no_secrets(self):
        text = "this is just normal text with no secrets"
        assert redact_secrets(text) == text

    def test_empty_string(self):
        assert redact_secrets("") == ""

    def test_custom_replacement(self):
        text = "token: ghp_1234567890abcdefghijklmnopqrstuvwxyz"
        result = redact_secrets(text, replacement="***")
        assert "***" in result
        assert "ghp_" not in result

    def test_custom_patterns(self):
        text = "my code is 12345"
        custom = [(r"\b\d{5}\b", "ZIP")]
        result = redact_secrets(text, patterns=custom)
        assert "12345" not in result
        assert "[REDACTED]" in result

    def test_private_key_block(self):
        text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA\n-----END RSA PRIVATE KEY-----"
        result = redact_secrets(text)
        assert "MIIEpAIBAAKCAQEA" not in result

    def test_multiple_secrets(self):
        text = "ghp_1234567890abcdefghijklmnopqrstuvwxyz and sk-abcdefghijklmnop1234567890qrstuv"
        result = redact_secrets(text)
        assert result.count("[REDACTED]") == 2


class TestExtractUrls:
    def test_single_url(self):
        urls = extract_urls("Visit https://example.com today!")
        assert urls == ["https://example.com"]

    def test_multiple_urls(self):
        urls = extract_urls("Go to http://a.com and https://b.org/page?q=1")
        assert len(urls) == 2
        assert "http://a.com" in urls
        assert "https://b.org/page?q=1" in urls

    def test_no_urls(self):
        assert extract_urls("No URLs here") == []

    def test_empty_string(self):
        assert extract_urls("") == []

    def test_url_with_path(self):
        urls = extract_urls("Check https://docs.python.org/3/library/re.html")
        assert len(urls) == 1
        assert "docs.python.org" in urls[0]

    def test_url_with_trailing_punctuation(self):
        urls = extract_urls("See https://example.com.")
        assert urls == ["https://example.com"]


class TestExtractEmails:
    def test_single_email(self):
        emails = extract_emails("Contact ali@example.com for info")
        assert emails == ["ali@example.com"]

    def test_multiple_emails(self):
        emails = extract_emails("Send to a@b.com and c@d.org")
        assert len(emails) == 2
        assert "a@b.com" in emails
        assert "c@d.org" in emails

    def test_no_emails(self):
        assert extract_emails("No emails here") == []

    def test_empty_string(self):
        assert extract_emails("") == []

    def test_complex_email(self):
        emails = extract_emails("Send to a.b+c@test.co.uk please")
        assert emails == ["a.b+c@test.co.uk"]

    def test_email_in_sentence(self):
        emails = extract_emails("My email is john.doe123@sub.domain.example.com, reply soon.")
        assert len(emails) == 1
        assert "john.doe123@sub.domain.example.com" in emails[0]


class TestStripMarkdown:
    def test_heading(self):
        assert strip_markdown("# Title") == "Title"
        assert strip_markdown("### Subtitle") == "Subtitle"

    def test_bold(self):
        assert strip_markdown("**bold text**") == "bold text"

    def test_italic(self):
        assert strip_markdown("*italic text*") == "italic text"

    def test_code_block(self):
        text = "```python\nprint('hi')\n```"
        assert "print('hi')" in strip_markdown(text)
        assert "```" not in strip_markdown(text)

    def test_inline_code(self):
        assert strip_markdown("Use `print()` function") == "Use print() function"

    def test_link(self):
        assert strip_markdown("[click here](https://example.com)") == "click here"

    def test_image(self):
        result = strip_markdown("![alt text](image.png)")
        assert "alt text" in result
        assert "image.png" not in result

    def test_blockquote(self):
        assert strip_markdown("> This is a quote") == "This is a quote"

    def test_list_items(self):
        text = "- item 1\n- item 2"
        result = strip_markdown(text)
        assert "item 1" in result
        assert "item 2" in result
        assert "- " not in result

    def test_numbered_list(self):
        text = "1. first\n2. second"
        result = strip_markdown(text)
        assert "first" in result
        assert "second" in result

    def test_empty_string(self):
        assert strip_markdown("") == ""

    def test_combined_markdown(self):
        text = "# Title\n\n**bold** and *italic* text\n\n[link](url)"
        result = strip_markdown(text)
        assert "Title" in result
        assert "bold" in result
        assert "italic" in result
        assert "link" in result
        assert "**" not in result
        assert "*" not in result

    def test_strikethrough(self):
        assert strip_markdown("~~deleted~~") == "deleted"


class TestSlugify:
    def test_basic(self):
        assert slugify("Hello World!") == "hello-world"

    def test_multiple_spaces(self):
        assert slugify("  Multiple   Spaces  ") == "multiple-spaces"

    def test_special_chars(self):
        assert slugify("Hello, World! How are you?") == "hello-world-how-are-you"

    def test_unicode_accents(self):
        assert slugify("Café résumé naïve") == "cafe-resume-naive"

    def test_custom_separator(self):
        assert slugify("Hello World", separator="_") == "hello_world"

    def test_max_length(self):
        result = slugify("This is a very long title", max_length=10)
        assert len(result) <= 10

    def test_empty_string(self):
        assert slugify("") == ""

    def test_only_special_chars(self):
        assert slugify("!@#$%") == ""

    def test_numbers_preserved(self):
        assert slugify("Article 123 Title") == "article-123-title"

    def test_leading_trailing_separators_stripped(self):
        assert slugify("---Hello---") == "hello"


class TestTextSimilarity:
    def test_identical_strings(self):
        assert text_similarity("hello world", "hello world") == pytest.approx(1.0)

    def test_completely_different(self):
        assert text_similarity("abc", "xyz") == 0.0

    def test_empty_string(self):
        assert text_similarity("", "hello") == 0.0
        assert text_similarity("hello", "") == 0.0
        assert text_similarity("", "") == 0.0

    def test_case_insensitive(self):
        score = text_similarity("Python", "python")
        assert 0.0 < score <= 1.0

    def test_partial_similarity(self):
        score = text_similarity("hello world", "hello there")
        assert 0.0 < score < 1.0

    def test_single_char(self):
        score = text_similarity("a", "a")
        assert score == 0.0

    def test_two_chars(self):
        score = text_similarity("ab", "ab")
        assert score == 1.0


class TestWordFrequency:
    def test_basic(self):
        result = word_frequency("the cat sat on the mat the cat")
        assert result == {"the": 3, "cat": 2, "sat": 1, "on": 1, "mat": 1}

    def test_case_insensitive_default(self):
        result = word_frequency("Hello hello WORLD")
        assert result == {"hello": 2, "world": 1}

    def test_case_sensitive(self):
        result = word_frequency("Hello hello WORLD", case_sensitive=True)
        assert result == {"Hello": 1, "hello": 1, "WORLD": 1}

    def test_stop_words(self):
        result = word_frequency("a an the cat", stop_words={"a", "an", "the"})
        assert result == {"cat": 1}

    def test_min_length(self):
        result = word_frequency("hi there friend", min_length=3)
        assert result == {"there": 1, "friend": 1}

    def test_empty_string(self):
        assert word_frequency("") == {}

    def test_sorted_by_count_desc(self):
        result = word_frequency("apple banana apple cherry apple banana")
        items = list(result.items())
        assert items[0] == ("apple", 3)
        assert items[1] == ("banana", 2)
        assert items[2] == ("cherry", 1)

    def test_alphabetical_tiebreak(self):
        result = word_frequency("zebra apple banana")
        items = list(result.items())
        assert items[0][0] == "apple"
        assert items[1][0] == "banana"
        assert items[2][0] == "zebra"


class TestReadingTime:
    def test_200_words(self):
        assert reading_time("word " * 200) == 1.0

    def test_400_words(self):
        assert reading_time("word " * 400) == 2.0

    def test_empty_string(self):
        assert reading_time("") == 0.0

    def test_short_text_min_1_minute(self):
        assert reading_time("one two three") == 1.0

    def test_seconds_unit(self):
        result = reading_time("word " * 200, unit="seconds")
        assert result == 60.0

    def test_custom_wpm(self):
        result = reading_time("word " * 300, words_per_minute=300)
        assert result == 1.0

    def test_large_text(self):
        result = reading_time("word " * 1000)
        assert result == 5.0


class TestExtractKeywords:
    def test_basic(self):
        result = extract_keywords("machine learning learning is the future of machine intelligence", num_keywords=3)
        assert "machine" in result
        assert "learning" in result
        assert len(result) == 3

    def test_frequency_ordering(self):
        result = extract_keywords("data data data science science models", num_keywords=3)
        assert result[0] == "data"
        assert result[1] == "science"
        assert result[2] == "models"

    def test_empty_string(self):
        assert extract_keywords("") == []

    def test_only_stop_words(self):
        result = extract_keywords("the and or but is was", min_length=3)
        assert result == []

    def test_min_length_filter(self):
        result = extract_keywords("cat dog bird fish", min_length=4)
        assert "bird" in result
        assert "fish" in result
        assert "cat" not in result
        assert "dog" not in result

    def test_custom_stop_words(self):
        result = extract_keywords("apple banana cherry", stop_words={"apple", "banana"})
        assert result == ["cherry"]

    def test_fewer_available_than_requested(self):
        result = extract_keywords("only one word", num_keywords=10)
        assert len(result) <= 3

    def test_alphabetical_tiebreak(self):
        result = extract_keywords("alpha beta gamma", num_keywords=3)
        assert result == ["alpha", "beta", "gamma"]


class TestNormalizeWhitespace:
    def test_multiple_spaces(self):
        assert normalize_whitespace("  hello   world  ") == "hello world"

    def test_tabs(self):
        assert normalize_whitespace("hello\t\tworld") == "hello world"

    def test_multiple_newlines(self):
        assert normalize_whitespace("a\n\n\n\nb") == "a\n\nb"

    def test_collapse_newlines_false(self):
        result = normalize_whitespace("a\n\n\n\nb", collapse_newlines=False)
        assert "a\n\n\n\nb" in result

    def test_empty_string(self):
        assert normalize_whitespace("") == ""

    def test_mixed_whitespace(self):
        result = normalize_whitespace("  hello \t \n  world  \n\n\n  foo  ")
        assert result == "hello\nworld\n\nfoo"

    def test_only_whitespace(self):
        assert normalize_whitespace("   \n\n  \t  ") == ""
