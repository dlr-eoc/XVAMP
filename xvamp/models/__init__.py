"""
Module containing the different joint atmospheric models.
"""

# provide quick access to models
from .model import Model as _Model
from .duan_et_al_2010 import Duan2010 as _Duan2010
from .onboard import OnboardPolynomial as _OnboardPolynomial

# make explicit
Model = _Model
Duan2010 = _Duan2010
OnboardPolynomial = _OnboardPolynomial
