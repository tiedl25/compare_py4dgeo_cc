import os
import sys
import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../../src'))
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CompareCC_Py4dGeo'
copyright = '2022, Max Tiedl'
author = 'Max Tiedl'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [ 'sphinx.ext.autodoc', 'sphinx.ext.doctest', 'sphinx.ext.todo', 'sphinx.ext.mathjax', 'sphinx.ext.ifconfig', 'sphinx.ext.viewcode', 'sphinx.ext.githubpages', 'sphinx.ext.napoleon', 'sphinx_rtd_theme',]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
