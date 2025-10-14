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

import pytorch_sphinx_theme2

# Add the source directory to Python path so modules can be imported
sys.path.insert(0, os.path.abspath("../../src/forge"))


# Determine the version path for deployment
def get_version_path():
    """Get the version path based on environment variables or git context."""
    # Check if we're in CI/CD and get the target folder
    github_ref = os.environ.get("GITHUB_REF", "")

    # Convert refs/tags/v1.12.0rc3 into 1.12.
    # Matches the logic in .github/workflows/docs.yml
    if github_ref.startswith("refs/tags/v"):
        import re

        match = re.match(r"^refs/tags/v([0-9]+\.[0-9]+)\..*", github_ref)
        if match:
            return match.group(1) + "/"

    # Default to main for main branch or local development
    return "main/"


# Set base URL based on deployment context
version_path = get_version_path()

project = "torchforge"
copyright = ""
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
    "sphinx_autodoc_typehints",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
]

html_baseurl = (
    f"https://meta-pytorch.org/forge/{version_path}"  # needed for sphinx-sitemap
)
sitemap_locales = [None]
sitemap_excludes = [
    "search.html",
    "genindex.html",
]
sitemap_url_scheme = "{link}"

# Ensure static files use relative paths
html_static_path = ["_static"]

templates_path = [
    "_templates",
    os.path.join(os.path.dirname(pytorch_sphinx_theme2.__file__), "templates"),
]
exclude_patterns = ["tutorials/index.rst", "tutorials/template_tutorial.rst"]

html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["custom.js"]

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
            "url": "https://pypi.org/project/torchforge/",
            "icon": "fa-brands fa-python",
        },
    ],
    "use_edit_page_button": True,
    "navbar_center": "navbar-nav",
    "canonical_url": "https://meta-pytorch.org/forge/",
    "header_links_before_dropdown": 7,
    "show_toc_level": 2,
    "navigation_depth": 3,
}

theme_variables = pytorch_sphinx_theme2.get_theme_variables()

html_context = {
    "theme_variables": theme_variables,
    "display_github": True,
    "github_url": "https://github.com",
    "github_user": "meta-pytorch",
    "github_repo": "forge",
    "feedback_url": "https://github.com/meta-pytorch/forge",
    "colab_branch": "gh-pages",
    "github_version": "main",
    "doc_path": "docs/source",
    "has_sphinx_gallery": True,  # Enable tutorial call-to-action links
}

# For tutorial repository configuration
# Note: github_user and github_repo are combined in the template as "{{ github_user }}/{{ github_repo }}"
# So we keep github_user = "meta-pytorch" and github_repo = "forge" already set above
# and only need to ensure the branch settings are correct
tutorial_repo_config = {
    "github_version": "main",  # This maps to github_branch in the template
    "colab_branch": "gh-pages",
}
html_context.update(tutorial_repo_config)

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
    "substitution",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": False,
    "inherited-members": False,
}

# Autodoc configuration for cleaner signatures
autodoc_preserve_defaults = True  # Preserves default values without expansion
autodoc_typehints = "description"  # Move type hints to description instead of signature
autodoc_typehints_description_target = (
    "documented_params"  # Only add types to documented params
)

# Disable docstring inheritance
autodoc_inherit_docstrings = False
autodoc_typehints = "none"


# Removed suppress_warnings to make the build stricter
# All warnings will now be treated as errors when -W is passed to sphinx-build

# Be strict about references to catch broken links and references
nitpicky = False

# Napoleon settings for Google-style docstrings (from torchtitan and other dependencies)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True


# -- Sphinx Gallery configuration -------------------------------------------
sphinx_gallery_conf = {
    "examples_dirs": "tutorial_sources",  # Path to examples directory
    "gallery_dirs": "tutorials",  # Path to generate gallery
    "filename_pattern": ".*",  # Include all files
    "download_all_examples": False,
    "first_notebook_cell": "%matplotlib inline",
    "plot_gallery": "True",
    "promote_jupyter_magic": True,
    "backreferences_dir": None,
    "show_signature": False,
    "write_computation_times": False,
}


def clean_docstring_indentation(app, what, name, obj, options, lines):
    if name and name.startswith("torchtitan."):
        lines[:] = [line.lstrip() for line in lines]
        if lines and lines[-1].strip():
            lines.append("")


def setup(app):
    app.connect("autodoc-process-docstring", clean_docstring_indentation)
