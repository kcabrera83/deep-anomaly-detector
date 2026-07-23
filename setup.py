from setuptools import setup, find_packages

setup(
    name="deep-anomaly-detector",
    version="1.0.0",
    author="Ing. Kelvin Cabrera",
    description="Deep Learning anomaly detection for oil & gas operations",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "flask>=2.3.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
    ],
)
