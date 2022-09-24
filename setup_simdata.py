from setuptools import setup, findall
from os import path


# Get the long description from the README file
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='bluesky-simdata',
    use_calver=True,
    setup_requires=['calver'],
    author='The BlueSky community',
    license='GNU General Public License v3 (GPLv3)',
    maintainer='Jacco Hoekstra and Joost Ellerbroek',
    description='The Open Air Traffic Simulator - simulation resources',
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
    packages=['bluesky.resources.performance', 'bluesky.resources.navdata'],  # Required
    package_data={'bluesky.resources.performance': [f.replace('bluesky/resources/performance/', '', 1) for f in findall('bluesky/resources/performance')],
                  'bluesky.resources.navdata':     [f.replace('bluesky/resources/navdata/', '', 1) for f in findall('bluesky/resources/navdata')]},
    project_urls={
        'Source': 'https://github.com/TUDelft-CNS-ATM/bluesky',
    },
    zip_safe=False
)