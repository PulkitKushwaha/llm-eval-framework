from setuptools import setup, find_packages
 
setup(
    name="llm-eval-framework",
    version="0.1.0",
    author="Pulkit Kushwaha",
    author_email="pulkitkushwahadev@gmail.com",
    description="A standalone evaluation framework for RAG pipelines and LLM systems",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "ragas>=0.1.7",
        "langchain>=0.1.20",
        "openai>=1.14.0",
        "pandas>=2.2.1",
        "pydantic>=2.6.4",
        "sentence-transformers>=2.6.1",
        "click>=8.1.7",
    ],
    entry_points={
        "console_scripts": [
            "llm-eval=llm_eval.cli:main",
        ],
    },
)
