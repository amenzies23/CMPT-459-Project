from setuptools import setup, find_packages

# Read the README for the long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()
with open("requirements.txt",'r') as f:
    deps = f.readlines()
setup(
    name="plant_health_analysis",
    version="0.1.0",
    author="Still Trainig",
    author_email="your.email@example.com",
    description="Our CMPT-459 Project Source code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=deps,
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)