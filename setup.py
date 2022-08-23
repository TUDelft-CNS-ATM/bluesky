# Always prefer setuptools over distutils
from os import path
from setuptools import setup, find_packages
from codecs import open
import shutil
import configparser

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Get base requirements from requirements.txt
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
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

# Temporarily create resources folder
if not path.exists('bluesky/resources'):
    shutil.copytree('data', 'bluesky/resources/data', ignore=shutil.ignore_patterns('cache'))
    shutil.copytree('plugins', 'bluesky/resources/plugins')
    shutil.copytree('scenario', 'bluesky/resources/scenario')
    with open('bluesky/resources/__init__.py', 'w') as f:
        f.write('')

setup(
    name='bluesky-simulator',  # 'bluesky' package name already taken in PyPI
    use_calver=True,
    setup_requires=['calver'],
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
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),  # Required
    include_package_data=True,
    # package_data={
    #       'bluesky.resources': [f for f in glob.glob('data/**/*') if path.isfile(f)] + ['data/default.cfg']
    # },
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # `pip` to create the appropriate form of executable for the target
    # platform.
    # data_files = [('resources/data', [f for f in glob.glob('data/**/*') if path.isfile(f)
    #               and path.basename(f) not in ('world.16384x8192.dds',
    #                                            'world.8192x4096.dds')] + ['data/default.cfg']),
    #               ('resources/plugins', [f for f in glob.glob('plugins/*.py') if path.isfile(f)]),
    #               ('resources/utils', [f for f in glob.glob('utils/**/*') if path.isfile(f)])],
    entry_points={
        'console_scripts': [
            'bluesky=bluesky.__main__:main',
        ],
    },

    project_urls={
        'Source': 'https://github.com/TUDelft-CNS-ATM/bluesky',
    },
)

if path.exists('bluesky/resources'):
    shutil.rmtree('bluesky/resources')