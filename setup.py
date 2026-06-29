#!/usr/bin/env python
"""Setup script for Fancy LLM Router."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fancy-llm-router",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A production-grade LLM routing and evaluation system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/astevko/fancy_llm_router",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "httpx>=0.25.0",
        "aiohttp>=3.8.0",
        "sqlalchemy>=2.0.0",
        "alembic>=1.12.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "python-multipart>=0.0.6",
        "click>=8.1.0",
        "rich>=13.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0.0",
        "aiosqlite>=0.19.0",
        "tiktoken>=0.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.5.0",
            "types-pyyaml>=6.0.0",
            "types-requests>=2.31.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fancy-llm=fancy_llm_router.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
