# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from importlib.metadata import version as get_version
from datetime import datetime

# add root folder to path so that we can import the package in
# a relative sense
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "XVAMP"
copyright = f"{datetime.now().year}, DLR e.V."
author = "Tobias Köhne"
release: str = get_version("xvamp")
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.graphviz",
    "sphinx_autodoc_typehints",
    "nbsphinx",
    "sphinxcontrib.bibtex",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Other settings ------------------------------------------------

# complain about broken links
nitpicky = True

# intersphinx settings
intersphinx_mapping = {
    "python": ("https://docs.python.org/3.13/", None),
    "numpy": ("https://numpy.org/doc/2.2/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy-1.15.2/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/version/2.3/", None),
    "matplotlib": ("https://matplotlib.org/3.10.1/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}

# allow the reusing of 'Classes' and 'Functions' section labels by
# prefixing the document name
autosectionlabel_prefix_document = True

# typehints settings
typehints_defaults = "comma"

# shorter object names
add_module_names = False

# by default, show all members
autodoc_default_options = {
    "members": True,
    "member-order": "groupwise",
    "class-doc-from": "both",
    "special-members": "__call__",
    "undoc-members": False,
    "show-inheritance": True,
    "exclude-members": "datafolder, tables",
}

# for the bibliography
bibtex_bibfiles = ["refs.bib"]
bibtex_reference_style = "author_year"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
# html_static_path = ["_static"]

# for the notebooks
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
]

# for graphs
graphviz_output_format = "svg"
