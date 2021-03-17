import setuptools

with open("README.md", 'r') as fh:
    long_description = fh.read()

desc = 'A Python package for simulated quantum computing.'

setuptools.setup(
    name='pypSQUEAK',
    version='2.1.0',
    author='Jason K. Elhaderi',
    author_email='jasonelhaderi@gmail.com',
    description=desc,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/jasonelhaderi/pypsqueak',
    packages=setuptools.find_packages(exclude=[
        'tests',
        'examples',
        'docs',
        'htmlcov'
    ]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy>=1.19.1'],
    python_requires='~=3.8'
)
