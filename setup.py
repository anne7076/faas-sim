import os

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements-dev.txt", "r", encoding="utf-8") as fh:
    tests_require = [line.strip() for line in fh if line.strip()]

with open("requirements.txt", "r", encoding="utf-8") as fh:
    install_requires = [line.strip() for line in fh if line.strip()]

setuptools.setup(
    name="faas-sim",
    version="1.0.0.dev1",
    author="Thomas Rausch, Philipp Raith, Alexander Rashed",
    author_email="t.rausch@dsg.tuwien.ac.at, p.raith@dsg.tuwien.ac.at, alexander.rashed@gmail.com",
    description="FaaS simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/edgerun/faas-sim",
    packages=setuptools.find_packages(),
    setup_requires=['wheel'],
    test_suite="tests",
    tests_require=tests_require,
    install_requires=install_requires,
    python_requires='>=3.7',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    entry_points={
    },
)
