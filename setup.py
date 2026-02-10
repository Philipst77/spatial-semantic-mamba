from setuptools import setup, find_packages

setup(
    name="ss_mamba",
    version="1.0.0",
    author="Sina Mansouri, Neelesh Prakash Wadhwani, Philip Stavrev",
    author_email="Smansou3@gmu.edu",
    description="Spatial-Semantic Mamba for Computational Pathology",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/SS-Mamba",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "opencv-python>=4.5.0",
        "Pillow>=9.0.0",
        "tqdm>=4.60.0",
        "pyyaml>=6.0",
    ],
)
