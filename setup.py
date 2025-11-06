from setuptools import setup, find_packages

setup(
    name="fornow",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "jax==0.8.0",
        "jaxlib==0.8.0",
        "numpy==2.3.4",
        "optax==0.2.6",
        "flax==0.12.0",
        "rich==14.2.0",
        "joblib==1.5.2",
    ],
    extras_require={
        "test": [
            "pytest==8.4.2",
        ],
    },
)