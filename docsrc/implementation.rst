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
:doc:`the Quickstart Notebook </scripts/quickstart>` and the
:doc:`API description </xvamp>` itself. The remainder of this document is focused on
deviations and improvements from the :cite:t:`duan2010` study.

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Still needs to be written - the reference code only includes OCS absorption,
whereas XVAMP and the study include both. There are also discrepancies in the
models used (spectral line shapes and broadening coefficients), which is currently
being finalized.*

.. note::

   This should have a negligible impact on the final delay or attenuation values.

Polarization of H2SO4
^^^^^^^^^^^^^^^^^^^^^

*Still needs to be written - the reference code has some flaws in the way it
computes the H2SO4 polarization parameters. These are fixed internally, but not
fully finalized yet.*

.. note::

   This should have a negligible impact on the final delay or attenuation values.
