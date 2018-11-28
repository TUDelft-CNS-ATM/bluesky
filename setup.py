# Always prefer setuptools over distutils
from os import path
from setuptools import setup, find_packages
import glob

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='bluesky-simulator',  # 'bluesky' package name already taken in PyPI
    version='0.0.1.dev2',
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
        'Programming Language :: Python :: 3.7',
    ],

    # This field adds keywords for your project which will appear on the
    keywords='atm transport simulation aviation aircraft',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),  # Required
    install_requires=open(here + '/requirements.txt', 'r').readlines(),
    extras_require={
        'dev': ['check-manifest'],
        'test': ['coverage', 'flake8', 'radon', 'nose'],
    },
    include_package_data=True,
    package_data={
          'data': [f for f in glob.glob('data/**/*') if path.isfile(f)] + ['data/default.cfg']
    },
    #     'scripts': [here + '/BlueSky_pygame.py', here + '/BlueSky_qtgl.py',
    #                 here + '/utils/mkcustomnavdata/makescen.py'] + \
    #                 glob.glob(here + '/utils/Scenario-creator/'),
    #     '': glob.glob(here + '/plugins/*.py'),
    #     '': glob.glob(here + '/data/**/*'),
    # },
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # `pip` to create the appropriate form of executable for the target
    # platform.
    data_files = [('/share/bluesky/data', [f for f in glob.glob('data/**/*') if path.isfile(f)] + ['data/default.cfg']),
                  ('/share/bluesky/plugins', [f for f in glob.glob('plugins/*.py') if path.isfile(f)]),
                  ('/share/bluesky/utils', [f for f in glob.glob('utils/**/*') if path.isfile(f)])],
    entry_points={
        'console_scripts': [
            'bluesky=BlueSky:main',
        ],
    },

    scripts=['BlueSky_pygame.py', 'BlueSky_qtgl.py'],

    project_urls={
        'Source': 'https://github.com/ProfHoekstra/bluesky',
    },
)
