from setuptools import setup, find_packages

__version__ = "unknown"

# "import" __version__
for line in open("groupdelay/__init__.py"):
    if line.startswith("__version__"):
        exec(line)
        break

setup(
    name="groupdelay",
    version=__version__,
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
    ],
    author="",
    author_email="",
    description="Numerical computation of group delay",
    long_description=open('README.md').read(),
    license="MIT",
    keywords="group-delay".split(),
    url="http://github.com/spatialaudio/group-delay-of-filters",
    platforms='any',
    python_requires='>=3.5',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
    ],
    zip_safe=True,
)
