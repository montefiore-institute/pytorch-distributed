from setuptools import Command, find_packages, setup

setup(
    name = 'sugar',
    description = 'Syntactic sugar and extensions for PyTorch.',
    url = 'https://github.com/montefiore-ai/pytorch-sugar',
    author = 'Joeri Hermans',
    author_email = 'joeri.hermans@doct.uliege.be',
    classifiers = [
        'Intended Audience :: Developers',
        'Topic :: Utilities',
        'License :: Public Domain',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
    packages = find_packages(exclude=['docs', 'examples'])
)
