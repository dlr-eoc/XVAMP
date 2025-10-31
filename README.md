# X-Band Venus Atmosphere Model Permissivity (XVAMP)

Python implementation of the atmospheric permissivity model in X-band as described in:

> Xueyang Duan, Mahta Moghaddam, Daniel Wenkert, Rolando L. Jordan, and Suzanne E. Smrekar.
> _"X Band Model of Venus Atmosphere Permittivity."_
> Radio Science 45, no. 2 (2010).
> <https://doi.org/10.1029/2009RS004169>.

## Installation

Since this code is in active development, no self-contained packages are currently
being produced.

### Minimal

If you only want to use the code, not continue developing it, and are happy
with PyPI handling the installation of the requirements, a simple

```bash
pip install git+https://gitlab.dlr.de/veritas/veritas-shared/xvamp.git
```

is sufficient. This will always install the `main` branch, unless you specify
another tag, branch or commit by appending `@desired-version`. You can
then also update the package by simply calling

```bash
pip install --upgrade git+https://gitlab.dlr.de/veritas/veritas-shared/xvamp.git
```

(again with the optional `@desired-version`).

### Full

The full installation will clone the repository, optionally creating a new
virtual environment for the dependencies.

```bash
# clone repository
git clone https://gitlab.dlr.de/veritas/veritas-shared/xvamp.git
# change into directory
cd xvamp
```

If you want to install XVAMP into an existing virtual environment, manually
install the required dependencies as listed in the [pyproject.toml](pyproject.toml)
file. (The optional dependencies are only necessary if you want to recreate
the Sphinx-based API documentation). Alternatively, you can create a new
conda/mamba environment with the environment file provided in the repository:

```bash
# create the environment, including all dependencies
conda env create -f environment.yml
# activate the environment
conda activate xvamp
```

Finally, install the package using pip from the local source:

```bash
# optional: use git to select a certain tag, branch, or commit
# git checkout desired-version
# install static version into the environment
pip install .
# optional: to install an editable version, instead run
# pip install -e .
```

Installing updates is then simply updating the local repository, and if it's
not an editable install, running `pip install --upgrade .` again.

## Documentation

An API documentation is hosted on Gitlab Pages.

## Quick Start

See [this notebook](docsrc/scripts/quickstart.ipynb).

## Acknowledgments

DLR is grateful to Jessie Duan at JPL for making available the source code of
her reference implementation of her paper, and assisting with any questions that
came up during the development of this package.
