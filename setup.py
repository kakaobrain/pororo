# yapf: disable

from setuptools import setup, find_packages

packages = find_packages()
requirements = [
    "torch==1.6.0",
    "torchvision==0.7.0",
    "pillow>=4.1.1",
    "fairseq>=0.10.2",
    "transformers>=4.0.0",
    "sentence_transformers>=0.4.1.2",
    "nltk>=3.5",
    "word2word",
    "wget",
    "joblib",
    "lxml",
    "g2p_en",
    "whoosh",
    "marisa-trie",
    "kss",
    'dataclasses; python_version<"3.7"',
]

VERSION = {}  # type: ignore
with open("pororo/__version__.py", "r") as version_file:
    exec(version_file.read(), VERSION)

setup(
    name="pororo",
    version=VERSION["version"],
    description="Pororo: A Deep Learning based Multilingual Natural Language Processing Library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    url="https://github.com/kakaobrain/pororo",
    author="kakaobrain Team SIGNALS",
    author_email="contact@kakaobrain.com",
    license="Apache-2.0",
    packages=find_packages(include=["pororo", "pororo.*"]),
    install_requires=requirements,
    python_requires=">=3.6.0",
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    package_data={},
    include_package_data=True,
    dependency_links=[],
    zip_safe=False,
)
