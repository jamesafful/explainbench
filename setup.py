from setuptools import setup, find_packages

setup(
    name="explainbench",
    version="0.1.1",
    description="A toolkit for interpretable machine learning and fairness auditing",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="James Afful",
    author_email="affulj@iastate.edu",
    url="https://github.com/jamesafful/explainbench",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24,<1.27",
        "pandas==1.5.3",
        "scikit-learn==1.3.2",
        "matplotlib==3.7.1",
        "shap==0.44.1",
        "lime==0.2.0.1",
        "dice-ml==0.11",
        "streamlit==1.27.2",
        "typing-extensions>=4.5.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
