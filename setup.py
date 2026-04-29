from setuptools import setup, find_packages

setup(
    name="attena",
    version="1.0.0",
    description="AttenA+: Velocity Field Action Attention for Robotic Foundation Models",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
    ],
)
