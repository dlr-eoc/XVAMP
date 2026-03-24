Notes on the Implementation
===========================

Reference Model
---------------

The reference model :class:`~xvamp.model.Duan2010` is based on :cite:t:`duan2010`.
To understand what it does and how it is structured, the study should be read first.
The key quantities that we want out of the model are the absorption coefficient and
index of refraction as a function of height, since those can be converted to
the range delay and attenuation, which are needed for the SAR processing.

Structure
^^^^^^^^^

From an implementation perspective, the code is structured as follows:

.. graphviz::

   digraph pre {

      # settings
      splines=ortho;
      node[colorscheme=pastel13, fontname="Segoe UI,sans-serif,serif"];

      # nodes
      cloud[label="Cloud Density\nand Concentration", shape=box];
      tpd[label="Temperature, Pressure\n and Density", shape=box];
      mr[label="Species Mixing\nRatios", shape=box];
      el[label="Electron\nDensity", shape=box];
      polpars[label="Species Polarization\nParameters", shape=box];
      dens[label="Species Densities", shape=box];
      compspecs[
         label="Compute species and cloud\npolarizations and absorptions",
         shape=plaintext
      ];
      pol[label="Polarization", shape=box, style=filled, fillcolor=1];
      abs[label="Absorption", shape=box];
      relpermatmo[label="Real Part of Atmosphere's\nRelative Permittivity", shape=box];
      relpermiono[label="Real Part of Ionosphere's\nRelative Permittivity", shape=box];
      combatmoiono[label="Combine Atmosphere\nand Ionosphere", shape=plaintext];
      realrelperm[label="Real Part of\nRelative Permittivity", shape=box];
      imagrelperm[label="Imaginary Part of\nRelative Permittivity", shape=box];
      relperm[label="Relative\nPermittivitiy", shape=box];
      ref[label="Refraction", shape=box, style=filled, fillcolor=2];

      # edges
      {tpd mr} -> dens;
      {tpd dens polpars cloud} -> compspecs;
      compspecs -> {pol abs};
      pol -> relpermatmo;
      el -> relpermiono;
      {relpermatmo relpermiono} -> combatmoiono;
      combatmoiono -> realrelperm;
      {realrelperm abs} -> imagrelperm;
      {realrelperm imagrelperm} -> relperm;
      relperm -> ref;
   }

This functionality is provided in the instantiation of a :class:`~xvamp.model.Duan2010`
model. All input datasets (folder icons) are part of :mod:`~xvamp.reference` and are
loaded at the package import time.

Once the two key quantities (refraction and absorption) are computed as a function
of altitude, we can compute the quantities actually needed for the SAR processing:

.. graphviz::

   digraph post {
      # settings
      splines=ortho;
      node[colorscheme=pastel13, style=rounded, fontname="Segoe UI,sans-serif,serif"];

      # nodes
      pol[label="Polarization", shape=box, style=filled, fillcolor=1];
      ref[label="Refraction", shape=box, style=filled, fillcolor=2];
      th[label="Terrain\nHeight"];
      ts[label="Satellite\nHeight"];
      la[label="Instrument\nLook Angle"];
      int[label="Integrate through\nProfile", shape=plaintext];
      apprange[label="Apparent\nRange", style=filled, fillcolor=3];
      att[label="Two-Way\nAttenuation", style=filled, fillcolor=3];
      ca[label="Central\nAngle"];
      geomrange[label="Geometric\nRange", style=filled, fillcolor=3];

      # edges
      {th ts la ref pol} -> int;
      int -> {apprange att ca};
      {th ts ca} -> geomrange;
   }

This functionality is provided as part of the
:meth:`~xvamp.model.Model.get_range_attenuation_angle` and
:func:`~xvamp.utils.geometric_range_from_central_angle` functions.

For more information about the usage of the code, please see
:doc:`the Quick Start Notebook </scripts/quickstart>` and the
:doc:`API description </xvamp>` itself. The remainder of this document is focused on
deviations and improvements from the :cite:t:`duan2010` study.

Remark on polarization notation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To prevent confusion when comparing XVAMP to the paper when it comes to the computation
of the polarization, the following is a quick recap of how the polarization terms
are computed for each species and then combined.

We recall eq. (4):

.. math::
   
   P_\text{mix} = \sum_i \Phi_i^* P_i \left( T, \frac{\rho_{r,mix}}{\nu_i^*} \right)

as well as the two main equations used to derive polarizations, the one from
:cite:t:`harvey2005`, eq. (8), using the virial expansion, reproduced partially in
eq. (8) in :cite:t:`duan2010`:

.. math:: \frac{P}{\rho} = A_\epsilon + A_\mu / T + B_\epsilon \rho + C \rho^D

and the one from :cite:t:`pitzer1983`, eq. (14):

.. math:: P_\nu = \rho \frac{4\pi N_0}{3} \left( \alpha_T + \frac{\mu^2 g}{3kT} \right)

where we have substituted :math:`\rho = d/M`.

Now, inserting the virial expansion equation and the definitions for :math:`\Phi_i^*`
and :math:`\rho_{r,mix}` into the mixing equation and then simplifying yields:

.. math::

   P_\text{mix} = \sum_i x_i \rho_{mix} \left(
      A_{\epsilon,i} + A_{\mu,i} / T + B_{\epsilon,i} \rho_i + C \rho_i^D \right)

where :math:`x_i` is the molar fraction and :math:`\rho_{mix}` is the molar density
of the mixture. This can therefore be further simplified by using the molar density
of the species, :math:`\rho_i`:

.. math::

   P_\text{mix} = \sum_i \rho_i \left(
      A_{\epsilon,i} + A_{\mu,i} / T + B_{\epsilon,i} \rho_i + C_i \rho_i^{D_i} \right)
      = \sum_i \rho_i P'_i

here, we have introduced the shorthand notation :math:`P'_i` for later use.
Note how the terms inside the sum are equivalent to the :cite:t:`pitzer1983` model
if we set

.. math:: A_{\epsilon,i} = \frac{4\pi N_0}{3} \alpha_T

.. math:: A_{\mu,i} = \frac{4\pi N_0 g}{9k} \mu^2

.. math:: B_{\epsilon,i} = C_i = 0

which in turn also implies :math:`P_\nu = P`. (This equivalence is the reason the
:class:`~xvamp.model.Duan2010` option ``use_virial_approximation`` has no effect, it
just changes the notation of the parameters.)

For the species where we have detailed parameters according to the :cite:t:`harvey2005`
polarization equation, we should make use of them, which means the overall model
is nonlinear in molar density :math:`\rho`. This is one of the reasons why internally,
XVAMP always directly computes the product of :math:`\rho_i P'_i (\rho_i)`, instead of
first calculating :math:`P'_i (\rho_i)`, and then later scaling it by :math:`\rho_i`.
The other reason is the benefit that the :math:`\rho_i P'_i (\rho_i)` formulation
(compared to the :math:`\Phi_i^* P_i \left( T, \frac{\rho_{r,mix}}{\nu_i^*} \right)`
notation) does not require the knowledge of any characteristic volumes.

On the coding side, both types of describing the polarization parameters are
implemented:

- :cite:t:`harvey2005` formulation: :meth:`~xvamp.model.Duan2010.eq8` to evaluate,
  and :meth:`~xvamp.model.Duan2010.A_epsilon_from_eq8` to estimate :math:`A_\epsilon`
  from a reference polarization.
- :cite:t:`pitzer1983` formulation: :meth:`~xvamp.model.Duan2010.eq14` to evaluate,
  and :meth:`~xvamp.model.Duan2010.alpha_T_from_eq14` to estimate :math:`\alpha_T`
  from a reference polarization.

The polarization parameters are evaluted according to the format they're in inside
:meth:`~xvamp.model.Duan2010.evaluate_polarization_parameters`, and simply summed
together (since their relative importance to the mixture polarization has already
been taken into account) in :meth:`~xvamp.model.Duan2010.sum_polarizations`.

Modifications
^^^^^^^^^^^^^

In the following, the :cite:t:`duan2010` study is simply referred to as the "paper",
and Jessie Duan's Matlab implementation as the "reference code".

"Bottom" of the simulation
''''''''''''''''''''''''''

The reference code is only defined for altitudes of 0 and above, i.e., at the mean
planetary radius of 6051.8 km or more. If delay or attenuation quantities for areas
below the mean radius are desired, the final permittivity profile is extrapolated.
XVAMP instead extrapolates the temperature, pressure, density, and mixing ratio
profiles to negative altitudes instead.

.. note::

   This should have a negligible impact on the final delay or attenuation values.

Cloud polarization and absorption
'''''''''''''''''''''''''''''''''

The paper describes a way to incorporate the effect of the cloud layer in sections
2.1.5 and 2.2.5. The reference code, however, follows a different approach: the one
assuming shell-like droplets described by :cite:t:`cimino1982`. The two approches
differ by one and two orders of magnitude in the polarization and absorption profiles
they yield, respectively, with the paper version being the higher one.
For typical satellite heights, the difference amounts to a sub-millimeter change
in the delay terms, but an approximately 2.7 dB two-way attenuation increase for
the paper version.

The reason the code uses the :cite:t:`cimino1982` model instead is because it
was derived explicitly for the Venus conditions. As such, the Cimino model explains the
observed discrepancy between cloud "mass contents derived from the absorption
coefficient data" and "those measured by [...] the Sounder probe" (section 9).
Therefore, the Cimino model is preferred, and also implemented in XVAMP.
However, the paper version is available as an option (``use_cimino_clouds=False``).

.. important::

   This has a **significant** impact on the final attenuation values,
   and a negligible one on the final delay values.

Polarization and absorption of OCS
''''''''''''''''''''''''''''''''''

*Still needs to be written - the reference code only includes OCS absorption,
whereas XVAMP and the study include both. There are also discrepancies in the
models used (spectral line shapes and broadening coefficients), which is currently
being finalized.*

.. note::

   This should have a negligible impact on the final delay or attenuation values.

Polarization of H2SO4
'''''''''''''''''''''

When computing the the polarization of the gaseous H2SO4, the reference code relied
on some unnecessary assumptions to derive the polarizability :math:`A_\epsilon`.
The present code addresses this by instead following the polarization computation
approach suggested by the study which provides the necessary experimental data,
:cite:t:`kolodner1998`, Section 3.2. This approach is implemented in
:meth:`~xvamp.reference.KolodnerSteffes1998.get_eps_prime_r_and_molar_density`
(to derive the mass density and the real part of the relative permittivity of H2SO4)
and then completed in a general sense in
:meth:`~xvamp.model.Duan2010.get_polarization_parameters` (to compute the
polarization parameters that can then be evaluated at given molar densities and
temperatures of the atmospheric column).

.. note::

   This has no impact on the attenuation (since absorption is computed from a
   different model; Section 7 in :cite:t:`kolodner1998`), but changes the polarization
   profile by about 20%. Since H2SO4 is only a minor contributor to the total
   polarization, this should have a negligible impact on the final delay values.

Limitations
^^^^^^^^^^^

There are many assumptions made in the model, most of which are described in the paper.
Some others, however, come from the implementation. The following is a
**non-exhaustive** list of those.

- There is currently no check being made that all species molar fractions add up to
  one (or equivalently, that the sum of all species mass densities equals the total
  mass density). Furthermore, the cloud mass density is considered "outside" the other
  species mass densities, i.e., it is not derived from H2O or H2SO4 abundances in the
  profile, nor does it change when H2O or H2SO4 changes.
