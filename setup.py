import os
from setuptools import setup, find_packages


def fread(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="momAPI",
    version=0.1,
    author="Charles Laroche",
    author_email="charles.laroche@ensae.fr",
    description="",
    url="https://github.com/CharlesLaroche/MOM_API",
    license="",
    install_requires=['numpy', 'torch', 'sklearn'],
    keywords="MOM machine-learning deep-learning nn lasso elasticnet cross-validation svm",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[])
