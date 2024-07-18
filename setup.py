from setuptools import setup

setup(
    name="jupy_tools",
    version="0.1.0",
    description="RDKit tooling for Jupyter notebooks.",
    url="https://github.com/apahl/jupy_tools",
    author="Axel Pahl",
    author_email="",
    license="MIT",
    packages=["jupy_tools"],
    install_requires=[
        "rdkit",
        "pillow",
        "pandas",
        "matplotlib",
        "seaborn",
        "cairocffi",
        "networkx",
        "graphviz",
        "scikit-learn",
        "scipy",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
)
