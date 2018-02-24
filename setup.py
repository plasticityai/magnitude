from setuptools import find_packages
from distutils.core import setup

setup(
    name='pymagnitude',
    packages=find_packages(exclude=['tests', 'tests.*']),
    version='0.0.15',
    description='A universal Python package for utilizing vector embeddings in a fast, efficient manner.',
    author='Plasticity',
    author_email='support@plasticity.ai',
    url='http://plasticity.ai/api',
    keywords=['pymagnitude', 'magnitude', 'plasticity', 'nlp',
              'word', 'vector', 'embeddings', 'embedding', 'word2vec',
              'gensim', 'alternative', 'machine', 'learning', 'annoy', 
              'index', 'approximate', 'nearest', 'neighbors'],
    license='MIT',
    setup_requires=[
        'numpy >= 1.14.0'
    ],
    install_requires=[
        'pip >= 9.0.1',
        'numpy >= 1.14.0',
        'xxhash >= 1.0.1',
        'fasteners >= 0.14.1',
        'annoy >= 1.11.4',
        'lz4 >= 1.0.0'
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4'
    ],
)