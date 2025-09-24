# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# Configuration file for the Sphinx documentation builder.
#

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
from unittest.mock import MagicMock

import pytorch_sphinx_theme2

# Add the source directory to Python path so modules can be imported
sys.path.insert(0, os.path.abspath("../../src"))


# Mock problematic modules that might be missing during doc build
class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()


# Simple and reliable mocking approach
MOCK_MODULES = [
    "six",
    "torch",
    "torch.nn",
    "torch.distributed",
    "transformers",
    "huggingface_hub",
    "ray",
    "datasets",
    "accelerate",
    "peft",
    "monarch",
    "monarch._src",
    "monarch._src.actor",
    "monarch._src.actor.allocator",
    "monarch._src.actor.shape",
    "monarch.actor",
    "monarch.tools",
    "monarch.tools.components",
]

for mod_name in MOCK_MODULES:
    try:
        __import__(mod_name)
    except ImportError:
        sys.modules[mod_name] = Mock()

# Try to import the modules to ensure they're available
try:
    import forge
    import forge.controller

    print("Successfully imported forge.controller")
except ImportError as e:
    print(f"Warning: Could not import forge.controller: {e}")


project = "torchforge"
copyright = "2025, PyTorch Contributors"
author = "PyTorch Contributors"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_design",
    "sphinx_sitemap",
    "sphinxcontrib.mermaid",
    "pytorch_sphinx_theme2",
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

html_baseurl = "https://meta-pytorch.org/forge/"  # needed for sphinx-sitemap
sitemap_locales = [None]
sitemap_excludes = [
    "search.html",
    "genindex.html",
]
sitemap_url_scheme = "{link}"

templates_path = [
    "_templates",
    os.path.join(os.path.dirname(pytorch_sphinx_theme2.__file__), "templates"),
]
exclude_patterns = []

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../../src"))
html_theme = "pytorch_sphinx_theme2"
html_theme_path = [pytorch_sphinx_theme2.get_html_theme_path()]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme_options = {
    "navigation_with_keys": False,
    "show_lf_header": False,
    "show_lf_footer": False,
    "analytics_id": "GTM-NPLPKN5G",
    "logo": {
        "text": "Home",
    },
    "icon_links": [
        {
            "name": "X",
            "url": "https://x.com/PyTorch",
            "icon": "fa-brands fa-x-twitter",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/meta-pytorch/forge",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "Discourse",
            "url": "https://discuss.pytorch.org/",
            "icon": "fa-brands fa-discourse",
        },
        {
            "name": "PyPi",
            "url": "https://pypi.org/project/forge/",
            "icon": "fa-brands fa-python",
        },
    ],
    "use_edit_page_button": True,
    "navbar_center": "navbar-nav",
    "canonical_url": "https://meta-pytorch.org/forge/",
    "header_links_before_dropdown": 7,
}

theme_variables = pytorch_sphinx_theme2.get_theme_variables()

html_context = {
    "theme_variables": theme_variables,
    "display_github": True,
    "github_url": "https://github.com",
    "github_user": "meta-pytorch",
    "github_repo": "forge",
    "feedback_url": "https://github.com/meta-pytorch/forge",
    "github_version": "main",
    "doc_path": "docs/source",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
]

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = True
