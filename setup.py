from setuptools import setup, find_packages

def parse_requirements(file):
    with open(file, 'r') as f:
        return [line.strip() for line in f if line.strip()]

setup(
    name='CMiner',
    version='0.1',
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    entry_points={
        'console_scripts': [
            'CMiner=src.main:main_function',
        ],
    },
)