from setuptools import setup, find_packages

setup(
    name='generals',
    version='1.0',
    description='Generals.io environment compliant with PettingZoo API standard powered by Numpy.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Matej Straka',
    author_email='strakammm@gmail.com',
    url='https://github.com/strakam/Generals-RL',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.6',
    install_requires=[],
)