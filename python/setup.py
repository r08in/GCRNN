from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='gcrnet',
      version='1.0.3',
      author="Bin Luo",
      author_email="bin.luo2@duke.edu",
      description="A Python library for sparse-input neural networks using group concave regularization",
      long_description=long_description,
      long_description_content_type="text/markdown",
      license='MIT',
      packages=['gcrnet'],
       install_requires=[
        "torch",
        "numpy",
        "six",
        "scipy",
        "scikit-learn",
        "lifelines",
        "matplotlib",
        "joblib",
        "pandas"
    ])