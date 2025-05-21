from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="mlp_numpy", 
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    author="Emeric",
    description="Implementation of a MLP from scratch using NumPy",
    python_requires=">=3.7",
)