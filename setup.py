import os
from setuptools import setup, find_packages

setup(
    name="edgeml-sdk",
    version="1.0.0",
    description="EdgeML — serve, deploy, and observe ML models on edge devices",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="EdgeML",
    author_email="team@edgeml.io",
    url="https://github.com/edgeml-ai/edgeml-python",
    packages=find_packages(),
    install_requires=[
        "psutil>=5.9.0",
        "httpx>=0.24.0",
        "click>=8.0.0",
        "pandas>=1.5.0",
        "pyarrow>=10.0.0",
        "qrcode[pil]>=7.0",
    ],
    extras_require={
        "serve": [
            "fastapi>=0.100.0",
            "uvicorn[standard]>=0.20.0",
        ],
        "mlx": [
            "mlx-lm>=0.10.0",
        ],
        "llama": [
            "llama-cpp-python>=0.2.0",
        ],
        "onnx": [
            "onnxruntime>=1.16.0",
        ],
        "whisper": [
            "pywhispercpp>=1.0.0",
        ],
        "ml": [
            "torch>=2.0.0",
            "numpy>=1.24.0",
            "pandas>=1.5.0",
        ],
        "secagg": [
            "cryptography>=41.0.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.21.0",
            "torch>=2.0.0",
            "numpy>=1.24.0",
            "pandas>=1.5.0",
            "keyring>=23.0.0",
            "cryptography>=41.0.0",
            "fastapi>=0.100.0",
            "uvicorn[standard]>=0.20.0",
            "flwr-datasets>=0.3.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.21.0",
            "torch>=2.0.0",
            "numpy>=1.24.0",
            "pandas>=1.5.0",
            "keyring>=23.0.0",
            "cryptography>=41.0.0",
            "ruff>=0.4.0",
            "fastapi>=0.100.0",
            "uvicorn[standard]>=0.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "edgeml=edgeml.cli:main",
        ],
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
