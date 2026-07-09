"""
Content hashing and checksum utilities for developers and LLM agents.

This module provides a unified, type-hinted interface to Python's
``hashlib`` and ``hmac`` modules, adding developer-friendly helpers for:

* Hashing strings, bytes, files, and entire directory trees.
* Keyed hashing (HMAC) for message authentication.
* Constant-time verification to prevent timing attacks.
* Content deduplication via hash-based identity.
* Automatic algorithm selection by digest length.
* Batch hashing of multiple items in a single call.

All functions are pure-Python with zero external dependencies beyond
the standard library.

Quick Start
-----------
>>> from ilovetools.utils.hash_utils import hash_string, hash_file, verify_hash
>>> digest = hash_string("hello world", algorithm="sha256")
>>> len(digest) == 64
True
>>> verify_hash("hello world", digest, algorithm="sha256")
True
>>> verify_hash("hello WORLD", digest, algorithm="sha256")
False

>>> from ilovetools.utils.hash_utils import identify_algorithm
>>> identify_algorithm("a" * 64)
'sha256'
>>> identify_algorithm("b" * 128)
'sha512'
"""

from __future__ import annotations

import hashlib
import hmac
import os
import pathlib
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

# ---------------------------------------------------------------------------
# Constants & algorithm registry
# ---------------------------------------------------------------------------

#: Mapping of supported algorithm names to their expected hex-digest lengths.
_ALGORITHM_DIGEST_LENGTHS: Dict[str, int] = {
    "md5": 32,
    "sha1": 40,
    "sha224": 56,
    "sha256": 64,
    "sha384": 96,
    "sha512": 128,
    "sha3_224": 56,
    "sha3_256": 64,
    "sha3_384": 96,
    "sha3_512": 128,
    "blake2b": 128,
    "blake2s": 64,
}

#: Default algorithm used when none is specified.
DEFAULT_ALGORITHM: str = "sha256"

#: Default chunk size (64 KiB) for streaming file reads.
DEFAULT_CHUNK_SIZE: int = 65536

#: Supported algorithms as a frozenset for quick membership checks.
SUPPORTED_ALGORITHMS: frozenset = frozenset(_ALGORITHM_DIGEST_LENGTHS.keys())


def _get_hasher(algorithm: str = DEFAULT_ALGORITHM) -> "hashlib._Hash":
    """Return a fresh hash object for *algorithm*.

    Parameters
    ----------
    algorithm : str
        One of :data:`SUPPORTED_ALGORITHMS`.

    Returns
    -------
    hashlib._Hash
        A new hash object ready to accept ``.update()`` calls.

    Raises
    ------
    ValueError
        If *algorithm* is not supported.

    >>> h = _get_hasher("sha256")
    >>> h.update(b"test")
    >>> h.hexdigest()[:8]
    '9f86d081'
    """
    algo = algorithm.lower().replace("-", "_")
    if algo not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"Unsupported algorithm '{algorithm}'. "
            f"Supported: {sorted(SUPPORTED_ALGORITHMS)}"
        )
    return hashlib.new(algo)


# ---------------------------------------------------------------------------
# String / bytes hashing
# ---------------------------------------------------------------------------

def hash_string(
    data: Union[str, bytes],
    algorithm: str = DEFAULT_ALGORITHM,
    *,
    encoding: str = "utf-8",
) -> str:
    """Compute the hex digest of a string or bytes value.

    Parameters
    ----------
    data : str | bytes
        The content to hash.  Strings are encoded using *encoding*.
    algorithm : str
        Hash algorithm name (default ``"sha256"``).
    encoding : str
        Text encoding to use when *data* is ``str`` (default ``"utf-8"``).

    Returns
    -------
    str
        Hexadecimal digest string.

    Raises
    ------
    ValueError
        If *algorithm* is not supported.
    TypeError
        If *data* is neither ``str`` nor ``bytes``.

    >>> hash_string("hello", algorithm="md5")
    '5d41402abc4b2a76b9719d911017c592'
    >>> hash_string(b"hello", algorithm="sha256")[:10]
    '2cf24dba5'
    >>> hash_string("")
    'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    """
    if isinstance(data, str):
        raw = data.encode(encoding)
    elif isinstance(data, (bytes, bytearray)):
        raw = bytes(data)
    else:
        raise TypeError(
            f"data must be str or bytes, got {type(data).__name__}"
        )
    hasher = _get_hasher(algorithm)
    hasher.update(raw)
    return hasher.hexdigest()


def hash_bytes(
    data: bytes,
    algorithm: str = DEFAULT_ALGORITHM,
) -> str:
    """Compute the hex digest of raw bytes.

    Convenience wrapper around :func:`hash_string` for byte input.

    >>> hash_bytes(b"test", algorithm="sha1")
    'a94a8fe5ccb19ba61c4c0873d391e987982fbbd3'
    """
    return hash_string(data, algorithm=algorithm)


def hash_multiple(
    items: Iterable[Union[str, bytes]],
    algorithm: str = DEFAULT_ALGORITHM,
    *,
    encoding: str = "utf-8",
) -> List[str]:
    """Hash multiple items in a single call.

    Parameters
    ----------
    items : Iterable[str | bytes]
        Iterable of strings or bytes to hash.
    algorithm : str
        Hash algorithm name.
    encoding : str
        Encoding for string items.

    Returns
    -------
    list[str]
        List of hex digests in the same order as *items*.

    >>> results = hash_multiple(["a", "b", "c"], algorithm="md5")
    >>> len(results)
    3
    >>> results[0] == hash_string("a", algorithm="md5")
    True
    """
    return [hash_string(item, algorithm=algorithm, encoding=encoding) for item in items]


# ---------------------------------------------------------------------------
# File / directory hashing
# ---------------------------------------------------------------------------

def hash_file(
    path: Union[str, pathlib.Path],
    algorithm: str = DEFAULT_ALGORITHM,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> str:
    """Compute the hex digest of a file's contents.

    The file is read in streaming chunks so that arbitrarily large
    files can be hashed without loading them entirely into memory.

    Parameters
    ----------
    path : str | pathlib.Path
        Path to the file to hash.
    algorithm : str
        Hash algorithm name (default ``"sha256"``).
    chunk_size : int
        Read buffer size in bytes (default 64 KiB).

    Returns
    -------
    str
        Hexadecimal digest string.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    IsADirectoryError
        If *path* is a directory (use :func:`hash_directory` instead).

    >>> import tempfile, os
    >>> with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
    ...     _ = f.write(b"hello world")
    ...     tmp = f.name
    >>> digest = hash_file(tmp, algorithm="sha256")
    >>> digest == hash_string("hello world", algorithm="sha256")
    True
    >>> os.unlink(tmp)
    """
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if p.is_dir():
        raise IsADirectoryError(
            f"Path is a directory, use hash_directory() instead: {path}"
        )
    hasher = _get_hasher(algorithm)
    with open(p, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def hash_directory(
    path: Union[str, pathlib.Path],
    algorithm: str = DEFAULT_ALGORITHM,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    include_hidden: bool = False,
    ignore_patterns: Optional[List[str]] = None,
) -> str:
    """Compute a combined hash for all files in a directory tree.

    Files are sorted by relative path for deterministic output.  The
    hash is computed by feeding each file's relative path and content
    into the hasher sequentially, so that directory structure and
    file contents both contribute to the final digest.

    Parameters
    ----------
    path : str | pathlib.Path
        Root directory to hash.
    algorithm : str
        Hash algorithm name (default ``"sha256"``).
    chunk_size : int
        Read buffer size in bytes.
    include_hidden : bool
        If ``False`` (default), hidden files and directories (starting
        with ``.``) are skipped.
    ignore_patterns : list[str] | None
        Glob patterns of file/directory names to skip (e.g.
        ``["__pycache__", "*.pyc"]``).

    Returns
    -------
    str
        Hexadecimal digest representing the entire directory tree.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    NotADirectoryError
        If *path* is not a directory.

    >>> import tempfile
    >>> d = tempfile.mkdtemp()
    >>> open(os.path.join(d, "a.txt"), "w").write("hello")
    5
    >>> open(os.path.join(d, "b.txt"), "w").write("world")
    5
    >>> digest1 = hash_directory(d)
    >>> digest2 = hash_directory(d)
    >>> digest1 == digest2  # deterministic
    True
    """
    import fnmatch

    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    if not p.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")

    ignore = ignore_patterns or []
    hasher = _get_hasher(algorithm)

    all_files: List[pathlib.Path] = []
    for root, dirs, files in os.walk(p):
        # Filter directories in-place for os.walk pruning
        if not include_hidden:
            dirs[:] = [d for d in dirs if not d.startswith(".")]
        dirs[:] = [d for d in dirs if not any(fnmatch.fnmatch(d, pat) for pat in ignore)]
        for fname in files:
            if not include_hidden and fname.startswith("."):
                continue
            if any(fnmatch.fnmatch(fname, pat) for pat in ignore):
                continue
            all_files.append(pathlib.Path(root) / fname)

    all_files.sort(key=lambda f: str(f.relative_to(p)))

    for fpath in all_files:
        rel = str(fpath.relative_to(p)).replace(os.sep, "/")
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\x00")  # separator between path and content
        with open(fpath, "rb") as fh:
            while True:
                chunk = fh.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
        hasher.update(b"\x00")  # separator between files

    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# HMAC / keyed hashing
# ---------------------------------------------------------------------------

def hmac_digest(
    data: Union[str, bytes],
    key: Union[str, bytes],
    algorithm: str = DEFAULT_ALGORITHM,
    *,
    encoding: str = "utf-8",
) -> str:
    """Compute an HMAC hex digest for *data* with *key*.

    Parameters
    ----------
    data : str | bytes
        The message to authenticate.
    key : str | bytes
        The secret key.
    algorithm : str
        Hash algorithm name (default ``"sha256"``).
    encoding : str
        Encoding for string inputs.

    Returns
    -------
    str
        Hexadecimal HMAC digest.

    >>> h = hmac_digest("message", "secret", algorithm="sha256")
    >>> len(h)
    64
    >>> hmac_digest("message", "secret") == hmac_digest("message", "secret")
    True
    >>> hmac_digest("message", "wrong") != hmac_digest("message", "secret")
    True
    """
    raw_data = data.encode(encoding) if isinstance(data, str) else bytes(data)
    raw_key = key.encode(encoding) if isinstance(key, str) else bytes(key)
    algo = algorithm.lower().replace("-", "_")
    if algo not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"Unsupported algorithm '{algorithm}'. "
            f"Supported: {sorted(SUPPORTED_ALGORITHMS)}"
        )
    return hmac.new(raw_key, raw_data, algo).hexdigest()


# ---------------------------------------------------------------------------
# Verification (constant-time)
# ---------------------------------------------------------------------------

def verify_hash(
    data: Union[str, bytes],
    expected_digest: str,
    algorithm: str = DEFAULT_ALGORITHM,
    *,
    encoding: str = "utf-8",
) -> bool:
    """Verify that *data* matches *expected_digest* using constant-time comparison.

    Parameters
    ----------
    data : str | bytes
        The content to verify.
    expected_digest : str
        The expected hex digest.
    algorithm : str
        Hash algorithm name.
    encoding : str
        Encoding for string *data*.

    Returns
    -------
    bool
        ``True`` if the computed digest matches *expected_digest*.

    >>> digest = hash_string("test", algorithm="sha256")
    >>> verify_hash("test", digest, algorithm="sha256")
    True
    >>> verify_hash("tampered", digest, algorithm="sha256")
    False
    """
    actual = hash_string(data, algorithm=algorithm, encoding=encoding)
    return hmac.compare_digest(actual, expected_digest)


def verify_file(
    path: Union[str, pathlib.Path],
    expected_digest: str,
    algorithm: str = DEFAULT_ALGORITHM,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> bool:
    """Verify a file's checksum against *expected_digest* (constant-time).

    >>> import tempfile, os
    >>> with tempfile.NamedTemporaryFile(delete=False) as f:
    ...     _ = f.write(b"data")
    ...     tmp = f.name
    >>> digest = hash_file(tmp)
    >>> verify_file(tmp, digest)
    True
    >>> verify_file(tmp, "0" * 64)
    False
    >>> os.unlink(tmp)
    """
    actual = hash_file(path, algorithm=algorithm, chunk_size=chunk_size)
    return hmac.compare_digest(actual, expected_digest)


# ---------------------------------------------------------------------------
# Algorithm identification
# ---------------------------------------------------------------------------

def identify_algorithm(digest: str) -> str:
    """Guess the hash algorithm from the length of a hex digest.

    Parameters
    ----------
    digest : str
        A hexadecimal digest string.

    Returns
    -------
    str
        The most likely algorithm name.

    Raises
    ------
    ValueError
        If the digest length does not match any known algorithm.

    >>> identify_algorithm("a" * 32)
    'md5'
    >>> identify_algorithm("a" * 64)
    'sha256'
    >>> identify_algorithm("a" * 128)
    'sha512'
    >>> identify_algorithm("short")
    Traceback (most recent call last):
        ...
    ValueError: Digest length 5 does not match any known algorithm
    """
    length = len(digest)
    for algo, expected_len in _ALGORITHM_DIGEST_LENGTHS.items():
        if length == expected_len:
            return algo
    raise ValueError(
        f"Digest length {length} does not match any known algorithm"
    )


# ---------------------------------------------------------------------------
# Content deduplication
# ---------------------------------------------------------------------------

def find_duplicates(
    items: Iterable[Union[str, bytes]],
    algorithm: str = DEFAULT_ALGORITHM,
    *,
    encoding: str = "utf-8",
) -> Dict[str, List[int]]:
    """Find duplicate items by content hash.

    Parameters
    ----------
    items : Iterable[str | bytes]
        Iterable of items to check for duplicates.
    algorithm : str
        Hash algorithm to use.
    encoding : str
        Encoding for string items.

    Returns
    -------
    dict[str, list[int]]
        Mapping of hash digests to lists of indices where that
        content appears.  Only entries with **more than one** index
        (i.e. actual duplicates) are included.

    >>> dups = find_duplicates(["a", "b", "a", "c", "b"])
    >>> for h, indices in dups.items():
    ...     print(len(indices))
    2
    2
    >>> dups = find_duplicates(["x", "y", "z"])
    >>> len(dups)
    0
    """
    groups: Dict[str, List[int]] = {}
    for idx, item in enumerate(items):
        digest = hash_string(item, algorithm=algorithm, encoding=encoding)
        groups.setdefault(digest, []).append(idx)
    return {h: idxs for h, idxs in groups.items() if len(idxs) > 1}


def deduplicate_by_hash(
    items: Iterable[Union[str, bytes]],
    algorithm: str = DEFAULT_ALGORITHM,
    *,
    encoding: str = "utf-8",
) -> List[Union[str, bytes]]:
    """Return items with duplicates removed, preserving first-occurrence order.

    Parameters
    ----------
    items : Iterable[str | bytes]
        Iterable of items to deduplicate.
    algorithm : str
        Hash algorithm to use.
    encoding : str
        Encoding for string items.

    Returns
    -------
    list[str | bytes]
        Deduplicated list in original order.

    >>> deduplicate_by_hash(["a", "b", "a", "c", "b", "d"])
    ['a', 'b', 'c', 'd']
    >>> deduplicate_by_hash([])
    []
    >>> deduplicate_by_hash(["only"])
    ['only']
    """
    seen: set = set()
    result: List[Union[str, bytes]] = []
    for item in items:
        digest = hash_string(item, algorithm=algorithm, encoding=encoding)
        if digest not in seen:
            seen.add(digest)
            result.append(item)
    return result


# ---------------------------------------------------------------------------
# Short hash / fingerprint
# ---------------------------------------------------------------------------

def short_hash(
    data: Union[str, bytes],
    algorithm: str = DEFAULT_ALGORITHM,
    *,
    length: int = 8,
    encoding: str = "utf-8",
) -> str:
    """Return the first *length* characters of a hex digest.

    Useful for creating short, human-readable fingerprints or
    cache keys where collision probability is acceptable.

    Parameters
    ----------
    data : str | bytes
        Content to hash.
    algorithm : str
        Hash algorithm name.
    length : int
        Number of hex characters to keep (default 8).
    encoding : str
        Encoding for string *data*.

    Returns
    -------
    str
        Truncated hex digest.

    Raises
    ------
    ValueError
        If *length* is not positive or exceeds the digest length.

    >>> short_hash("hello world")
    'b94d27b9'
    >>> short_hash("hello world", length=12)
    'b94d27b9934d3e'
    >>> short_hash("test", length=0)
    Traceback (most recent call last):
        ...
    ValueError: length must be between 1 and the digest length
    """
    if length < 1:
        raise ValueError("length must be between 1 and the digest length")
    full = hash_string(data, algorithm=algorithm, encoding=encoding)
    if length > len(full):
        raise ValueError("length must be between 1 and the digest length")
    return full[:length]


# ---------------------------------------------------------------------------
# Batch file hashing
# ---------------------------------------------------------------------------

def hash_files(
    paths: Iterable[Union[str, pathlib.Path]],
    algorithm: str = DEFAULT_ALGORITHM,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Dict[str, str]:
    """Hash multiple files and return a mapping of path -> digest.

    Parameters
    ----------
    paths : Iterable[str | pathlib.Path]
        Iterable of file paths to hash.
    algorithm : str
        Hash algorithm name.
    chunk_size : int
        Read buffer size in bytes.

    Returns
    -------
    dict[str, str]
        Mapping of file path (as string) to hex digest.

    >>> import tempfile, os
    >>> f1 = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    >>> _ = f1.write(b"one"); f1.close()
    >>> f2 = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    >>> _ = f2.write(b"two"); f2.close()
    >>> result = hash_files([f1.name, f2.name])
    >>> len(result)
    2
    >>> result[f1.name] == hash_string("one")
    True
    >>> os.unlink(f1.name); os.unlink(f2.name)
    """
    return {str(p): hash_file(p, algorithm=algorithm, chunk_size=chunk_size) for p in paths}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "DEFAULT_ALGORITHM",
    "SUPPORTED_ALGORITHMS",
    "DEFAULT_CHUNK_SIZE",
    "hash_string",
    "hash_bytes",
    "hash_multiple",
    "hash_file",
    "hash_directory",
    "hash_files",
    "hmac_digest",
    "verify_hash",
    "verify_file",
    "identify_algorithm",
    "find_duplicates",
    "deduplicate_by_hash",
    "short_hash",
]
