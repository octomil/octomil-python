from setuptools import setup, find_packages

setup(
    name="edgeml-sdk",
    version="1.0.0",
    description="EdgeML Python SDK - Federated Learning Orchestration",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="EdgeML",
    author_email="team@edgeml.io",
    url="https://github.com/edgeml-ai/edgeml-python",
    packages=find_packages(),
    install_requires=[
        "requests>=2.31.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
