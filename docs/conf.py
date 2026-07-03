# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Matplotlib is imported by several bluesky modules; force a non-GUI backend
# so autodoc and the command-reference generator work on headless builders.
os.environ.setdefault('MPLBACKEND', 'Agg')

# Make the repository root importable (for autodoc of a source checkout)
# and the docs dir itself (for the _ext extension package).
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.dirname(__file__))

# -- Project information -----------------------------------------------------

project = 'BlueSky'
author = 'The BlueSky community'
copyright = '%Y, Delft University of Technology and the BlueSky community'

try:
    from importlib.metadata import version as _pkg_version
    release = _pkg_version('bluesky-simulator')
    version = '.'.join(release.split('.')[:2])
except Exception:
    release = version = 'dev'

# -- General configuration ---------------------------------------------------

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
    'sphinx_design',
]

myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'fieldlist',
    'substitution',
    'attrs_inline',
]
myst_heading_anchors = 3

autodoc_mock_imports = ['PyQt6', 'pygame', 'OpenGL', 'textual']
autodoc_member_order = 'bysource'

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
}

exclude_patterns = ['legacy', '_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 3,
    'collapse_navigation': False,
}
html_static_path = ['_static']

# -- Link check ---------------------------------------------------------------
# A few external links are known to be flaky or block automated requests;
# skip them rather than failing `make linkcheck` on infrastructure noise.

linkcheck_ignore = [
    r'https://discord\.gg/.*',
    r'https://www\.researchgate\.net/.*',
    r'https://github\.com/.*/(blob|tree)/.*',
]

# -- Command reference generation --------------------------------------------
# Regenerate the stack command reference from the live registry on every
# build, so it always reflects the code being documented. Nothing under
# reference/commands/generated/ is committed (see docs/.gitignore).

from _ext.gen_commands import generate_command_pages

generate_command_pages(os.path.join(os.path.dirname(__file__), 'reference', 'commands', 'generated'))
