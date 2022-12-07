# Always prefer setuptools over distutils
import os
import sys
from setuptools import setup, find_packages, Extension
import configparser


def get_numpy_include():
    import numpy
    return numpy.get_include()

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Get base requirements from requirements.txt
with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    install_requires = f.readlines()

# get extra requirements from setup.cfg
parser = configparser.ConfigParser()
parser.read('%s/setup.cfg' % here)
extras_requirements = {k: [vi.strip().split('#') for vi in v.split('\n') if vi]
                       for (k, v) in dict(parser['extras']).items()}
extras_requirements.update({
    'dev': ['check-manifest'],
    'test': ['coverage', 'flake8', 'radon', 'nose'],
})

setup(
    name='bluesky-simulator',  # 'bluesky' package name already taken in PyPI
    use_calver=True,
    setup_requires=['calver', 'numpy'],
    install_requires=install_requires,
    extras_require=extras_requirements,
    author='The BlueSky community',
    license='GNU General Public License v3 (GPLv3)',
    maintainer='Jacco Hoekstra and Joost Ellerbroek',
    description='The Open Air Traffic Simulator',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/TUDelft-CNS-ATM/bluesky',
    classifiers=[
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',

        # Pick your license as you wish
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10'
    ],

    # This field adds keywords for your project which will appear on the
    keywords='atm transport simulation aviation aircraft',
    packages=find_packages(exclude=['contrib', 'docs', 'tests', 'bluesky.resources.graphics', 'bluesky.resources.html', 'bluesky.resources.navdata', 'bluesky.resources.performance']),# Required
    include_package_data=True,
    exclude_package_data={'bluesky': ['resources/graphics/*', 'resources/html/*', 'resources/navdata/*', 'resources/performance/*']},
    package_data={
        'bluesky': ['resources/*']
    },
    entry_points={
        'console_scripts': [
            'bluesky=bluesky.__main__:main',
        ],
    },

    project_urls={
        'Source': 'https://github.com/TUDelft-CNS-ATM/bluesky',
    },
    include_dirs=[get_numpy_include()],
    ext_modules=[Extension('bluesky.tools.cgeo', ['bluesky/tools/src_cpp/cgeo.cpp']),
                Extension('bluesky.traffic.asas.cstatebased', ['bluesky/traffic/asas/src_cpp/cstatebased.cpp'], include_dirs=['bluesky/tools/src_cpp'])]
)
