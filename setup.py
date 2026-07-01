from setuptools import setup, find_packages

setup(
    name="qa-ml-framework",
    version="1.0.0",
    description="ML-powered QA intelligence: test failure prediction and performance anomaly detection",
    author="Your Name",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.0.0",
        "scikit-learn>=1.4.0",
        "numpy>=1.26.0",
        "matplotlib>=3.8.0",
        "PyYAML>=6.0.1",
        "lxml>=5.0.0",
        "imbalanced-learn>=0.12.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "dev": ["pytest>=8.0.0", "pytest-cov>=4.1.0", "seaborn>=0.13.0"],
    },
)
