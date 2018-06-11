# Always prefer setuptools over distutils
from os import path
from setuptools import setup, find_packages


here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='bluesky-simulator',
    version='0.0.1',
    description='The Open Air Traffic Simulator',
    long_description=long_description,
    url='https://github.com/ProfHoekstra/bluesky',
    classifiers=[
        'Development Status :: 3 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',

        # Pick your license as you wish
        'License :: OSI Approved :: GPLv3 License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    # This field adds keywords for your project which will appear on the
    keywords='atm transport simulation aviation aircraft',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),  # Required
    install_requires=open(here + '/requirements.txt', 'r').readlines(),
    extras_require={
        'dev': ['check-manifest'],
        'test': ['coverage', 'flake8', 'radon', 'nose'],
    },
    package_data={
        'scripts': [here + '/BlueSky_pygame.py', here + '/BlueSky_qtgl.py']
    #    'sample': ['package_data.dat'],
    },
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # `pip` to create the appropriate form of executable for the target
    # platform.
    entry_points={
        'console_scripts': [
            'bluesky=BlueSky:main',
        ],
    },

    scripts=[here + '/BlueSky_pygame.py', here + '/BlueSky_qtgl.py'],

    project_urls={
        'Source': 'https://github.com/ProfHoekstra/bluesky',
    },
)
