from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="Insurance_Premium_Prediction",
    version="1.0.0",
    author="dibyendubiswas1998",
    author_email="dibyendubiswas1998@gmail.com",
    description="Insurance Premium Prediction Web Application",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dibyendubiswas1998/Insurance-Premium-Prediction.git",
    packages=["src"],
    license="GNU",
    python_requires=">=3.8",
)
