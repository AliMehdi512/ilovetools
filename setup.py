from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ilovetools",
    version="0.2.9.post3",
    author="Ali Mehdi",
    author_email="ali.mehdi.dev579@gmail.com",
    description="A comprehensive Python utility library with modular tools for AI/ML, data processing, and daily programming needs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AliMehdi512/ilovetools",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    # Dependency metadata is maintained in pyproject.toml so setuptools'
    # configuration reads the authoritative list from there. Removing
    # `install_requires`/`extras_require` from setup.py avoids warnings
    # about overwriting fields when building with modern build backends.
    keywords="utilities, tools, ai, ml, data-processing, automation, python-library",
    project_urls={
        "Bug Reports": "https://github.com/AliMehdi512/ilovetools/issues",
        "Source": "https://github.com/AliMehdi512/ilovetools",
    },
)
