"""
Tests for ilovetools.utils.hash_utils module.

Covers string/bytes hashing, file/directory hashing, HMAC, verification,
algorithm identification, deduplication, short hashes, and batch file hashing.
"""

import os
import pathlib
import tempfile
import pytest

from ilovetools.utils.hash_utils import (
    hash_string,
    hash_bytes,
    hash_multiple,
    hash_file,
    hash_directory,
    hash_files,
    hmac_digest,
    verify_hash,
    verify_file,
    identify_algorithm,
    find_duplicates,
    deduplicate_by_hash,
    short_hash,
    DEFAULT_ALGORITHM,
    SUPPORTED_ALGORITHMS,
    DEFAULT_CHUNK_SIZE,
)


class TestHashString:
    def test_basic_sha256(self):
        digest = hash_string("hello world", algorithm="sha256")
        assert len(digest) == 64
        assert digest == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"

    def test_md5(self):
        assert hash_string("hello", algorithm="md5") == "5d41402abc4b2a76b9719d911017c592"

    def test_empty_string(self):
        digest = hash_string("")
        assert len(digest) == 64
        assert digest == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_bytes_input(self):
        assert hash_string(b"hello", algorithm="md5") == hash_string("hello", algorithm="md5")

    def test_bytearray_input(self):
        assert hash_string(bytearray(b"hello"), algorithm="md5") == hash_string("hello", algorithm="md5")

    def test_custom_encoding(self):
        # "cafe" with accent encodes differently in utf-8 vs latin-1
        utf8 = hash_string("caf\u00e9", encoding="utf-8")
        latin1 = hash_string("caf\u00e9", encoding="latin-1")
        assert utf8 != latin1

    def test_unsupported_algorithm(self):
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            hash_string("test", algorithm="nonexistent")

    def test_invalid_type(self):
        with pytest.raises(TypeError, match="must be str or bytes"):
            hash_string(123)

    def test_deterministic(self):
        assert hash_string("test") == hash_string("test")

    def test_different_inputs_different_hashes(self):
        assert hash_string("a") != hash_string("b")

    def test_all_supported_algorithms(self):
        for algo in SUPPORTED_ALGORITHMS:
            digest = hash_string("test", algorithm=algo)
            assert len(digest) > 0
            assert all(c in "0123456789abcdef" for c in digest)


class TestHashBytes:
    def test_basic(self):
        assert hash_bytes(b"test", algorithm="sha1") == "a94a8fe5ccb19ba61c4c0873d391e987982fbbd3"

    def test_matches_hash_string(self):
        assert hash_bytes(b"data") == hash_string(b"data")


class TestHashMultiple:
    def test_basic(self):
        results = hash_multiple(["a", "b", "c"], algorithm="md5")
        assert len(results) == 3
        assert results[0] == hash_string("a", algorithm="md5")
        assert results[1] == hash_string("b", algorithm="md5")
        assert results[2] == hash_string("c", algorithm="md5")

    def test_empty_list(self):
        assert hash_multiple([]) == []

    def test_mixed_types(self):
        results = hash_multiple(["text", b"bytes"])
        assert len(results) == 2


class TestHashFile:
    def test_basic(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        digest = hash_file(str(f))
        assert digest == hash_string("hello world")

    def test_pathlib_input(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"data")
        digest = hash_file(f)
        assert digest == hash_string("data")

    def test_large_file(self, tmp_path):
        f = tmp_path / "large.bin"
        data = b"x" * (DEFAULT_CHUNK_SIZE * 3 + 100)
        f.write_bytes(data)
        digest = hash_file(str(f))
        assert digest == hash_string(data)

    def test_custom_chunk_size(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("test data")
        d1 = hash_file(str(f), chunk_size=1024)
        d2 = hash_file(str(f), chunk_size=128)
        assert d1 == d2

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            hash_file("/nonexistent/path/file.txt")

    def test_is_directory(self, tmp_path):
        with pytest.raises(IsADirectoryError):
            hash_file(str(tmp_path))

    def test_different_algorithms(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("test")
        for algo in ["md5", "sha1", "sha256", "sha512", "blake2b"]:
            digest = hash_file(str(f), algorithm=algo)
            assert len(digest) > 0


class TestHashDirectory:
    def test_deterministic(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.txt").write_text("world")
        d1 = hash_directory(str(tmp_path))
        d2 = hash_directory(str(tmp_path))
        assert d1 == d2

    def test_changes_with_content(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        d1 = hash_directory(str(tmp_path))
        (tmp_path / "a.txt").write_text("HELLO")
        d2 = hash_directory(str(tmp_path))
        assert d1 != d2

    def test_changes_with_file_addition(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        d1 = hash_directory(str(tmp_path))
        (tmp_path / "b.txt").write_text("world")
        d2 = hash_directory(str(tmp_path))
        assert d1 != d2

    def test_skip_hidden_by_default(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / ".hidden").write_text("secret")
        d1 = hash_directory(str(tmp_path))
        (tmp_path / ".hidden").write_text("changed")
        d2 = hash_directory(str(tmp_path))
        assert d1 == d2

    def test_include_hidden(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / ".hidden").write_text("secret")
        d1 = hash_directory(str(tmp_path), include_hidden=True)
        (tmp_path / ".hidden").write_text("changed")
        d2 = hash_directory(str(tmp_path), include_hidden=True)
        assert d1 != d2

    def test_ignore_patterns(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "ignore.pyc").write_text("compiled")
        d1 = hash_directory(str(tmp_path), ignore_patterns=["*.pyc"])
        (tmp_path / "ignore.pyc").write_text("changed")
        d2 = hash_directory(str(tmp_path), ignore_patterns=["*.pyc"])
        assert d1 == d2

    def test_nested_directories(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.txt").write_text("nested")
        (tmp_path / "top.txt").write_text("top")
        digest = hash_directory(str(tmp_path))
        assert len(digest) == 64

    def test_not_found(self):
        with pytest.raises(FileNotFoundError):
            hash_directory("/nonexistent/dir")

    def test_not_a_directory(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("test")
        with pytest.raises(NotADirectoryError):
            hash_directory(str(f))


class TestHmacDigest:
    def test_basic(self):
        h = hmac_digest("message", "secret")
        assert len(h) == 64

    def test_deterministic(self):
        assert hmac_digest("msg", "key") == hmac_digest("msg", "key")

    def test_different_keys_different_digests(self):
        assert hmac_digest("msg", "key1") != hmac_digest("msg", "key2")

    def test_different_messages_different_digests(self):
        assert hmac_digest("msg1", "key") != hmac_digest("msg2", "key")

    def test_bytes_input(self):
        assert hmac_digest(b"msg", b"key") == hmac_digest("msg", "key")

    def test_unsupported_algorithm(self):
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            hmac_digest("msg", "key", algorithm="nonexistent")

    def test_sha512(self):
        h = hmac_digest("data", "secret", algorithm="sha512")
        assert len(h) == 128


class TestVerifyHash:
    def test_valid(self):
        digest = hash_string("test")
        assert verify_hash("test", digest) is True

    def test_invalid(self):
        digest = hash_string("test")
        assert verify_hash("tampered", digest) is False

    def test_different_algorithm(self):
        digest = hash_string("test", algorithm="md5")
        assert verify_hash("test", digest, algorithm="md5") is True

    def test_constant_time(self):
        digest = hash_string("x" * 1000)
        assert verify_hash("x" * 1000, digest) is True


class TestVerifyFile:
    def test_valid(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("data")
        digest = hash_file(str(f))
        assert verify_file(str(f), digest) is True

    def test_invalid(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("data")
        assert verify_file(str(f), "0" * 64) is False

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            verify_file("/nonexistent", "0" * 64)


class TestIdentifyAlgorithm:
    def test_md5(self):
        assert identify_algorithm("a" * 32) == "md5"

    def test_sha1(self):
        assert identify_algorithm("a" * 40) == "sha1"

    def test_sha256(self):
        assert identify_algorithm("a" * 64) == "sha256"

    def test_sha512(self):
        assert identify_algorithm("a" * 128) == "sha512"

    def test_sha224(self):
        assert identify_algorithm("a" * 56) == "sha224"

    def test_unknown_length(self):
        with pytest.raises(ValueError, match="does not match"):
            identify_algorithm("short")

    def test_empty_string(self):
        with pytest.raises(ValueError, match="does not match"):
            identify_algorithm("")


class TestFindDuplicates:
    def test_finds_duplicates(self):
        dups = find_duplicates(["a", "b", "a", "c", "b"])
        assert len(dups) == 2
        for h, indices in dups.items():
            assert len(indices) == 2

    def test_no_duplicates(self):
        dups = find_duplicates(["x", "y", "z"])
        assert len(dups) == 0

    def test_empty(self):
        assert find_duplicates([]) == {}

    def test_all_same(self):
        dups = find_duplicates(["same", "same", "same"])
        assert len(dups) == 1
        for indices in dups.values():
            assert len(indices) == 3

    def test_bytes_input(self):
        dups = find_duplicates([b"a", b"a", b"b"])
        assert len(dups) == 1


class TestDeduplicateByHash:
    def test_basic(self):
        result = deduplicate_by_hash(["a", "b", "a", "c", "b", "d"])
        assert result == ["a", "b", "c", "d"]

    def test_empty(self):
        assert deduplicate_by_hash([]) == []

    def test_single(self):
        assert deduplicate_by_hash(["only"]) == ["only"]

    def test_all_duplicates(self):
        assert deduplicate_by_hash(["x", "x", "x"]) == ["x"]

    def test_preserves_order(self):
        result = deduplicate_by_hash(["c", "a", "b", "a", "c"])
        assert result == ["c", "a", "b"]

    def test_no_duplicates(self):
        result = deduplicate_by_hash(["a", "b", "c"])
        assert result == ["a", "b", "c"]


class TestShortHash:
    def test_default_length(self):
        s = short_hash("hello world")
        assert len(s) == 8

    def test_custom_length(self):
        s = short_hash("hello world", length=12)
        assert len(s) == 12

    def test_matches_full_prefix(self):
        full = hash_string("test")
        assert short_hash("test", length=10) == full[:10]

    def test_length_zero(self):
        with pytest.raises(ValueError, match="length must be"):
            short_hash("test", length=0)

    def test_length_too_large(self):
        with pytest.raises(ValueError, match="length must be"):
            short_hash("test", length=1000)

    def test_different_algorithm(self):
        s = short_hash("test", algorithm="md5", length=8)
        assert len(s) == 8
        assert s == hash_string("test", algorithm="md5")[:8]


class TestHashFiles:
    def test_basic(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f1.write_text("one")
        f2 = tmp_path / "b.txt"
        f2.write_text("two")
        result = hash_files([str(f1), str(f2)])
        assert len(result) == 2
        assert result[str(f1)] == hash_string("one")
        assert result[str(f2)] == hash_string("two")

    def test_empty_list(self):
        assert hash_files([]) == {}

    def test_pathlib_inputs(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("data")
        result = hash_files([f])
        assert str(f) in result


class TestConstants:
    def test_default_algorithm(self):
        assert DEFAULT_ALGORITHM == "sha256"

    def test_supported_algorithms_contains_common(self):
        for algo in ["md5", "sha1", "sha256", "sha512", "blake2b"]:
            assert algo in SUPPORTED_ALGORITHMS

    def test_default_chunk_size_positive(self):
        assert DEFAULT_CHUNK_SIZE > 0
