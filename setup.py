from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="Organic Scoring",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    author="Dmytro Bobrenko",
    author_email="dmytro.bobrenko@macrocosmos.ai",
    description="Generic Organic Scoring Bittensor Framework",
    url="https://github.com/macrocosm-os/organic-scoring",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
