---
title: 'XVAMP: Modeling the Effects of the Venus Atmosphere on X-Band Radar with Python'
tags:
  - Python
  - Venus
  - planetary atmospheres
  - radar
authors:
  - name: Tobias Köhne
    orcid: 0000-0002-8400-7255
    corresponding: true
    affiliation: 1
  - name: Xueyang Duan
    orcid: 0000-0002-0716-5233
    affiliation: 2
affiliations:
 - name: Remote Sensing Technology Institute, German Aerospace Center (DLR), Weßling, Germany
   index: 1
 - name: Jet Propulsion Laboratory, California Institute of Technology, Pasadena, CA, USA
   index: 2
date: 26 July 2026
bibliography: refs.bib
---
<!-- markdownlint-disable MD025 -->

# Summary

One of the scientifically fascinating aspects about Venus is how it is very similar to
Earth in some regards, while being completely dissimilar in others. For example,
it is a rocky planet only marginally smaller than Earth, but it does not have a liquid
water ocean (currently or potentially ever) and its toxic atmosphere is about 92 times
as dense as ours.
The upcoming NASA mission Venus Emissivity, Radio Science, InSAR, Topography, and
Spectroscopy (VERITAS), currently scheduled for a launch in 2032, is a spacecraft
designed to answer some of the key questions that could explain how a planet starting
out similar to Earth in the habitable zone of our Sun could end up so differently
[@smrekar2022].
Among its instruments is the Venus Interferometric Synthetic Aperture Radar (VISAR),
which will image the surface using X-band radar of 7.9 GHz using its two antennas.
Apart from high-resolution, near-global surface imagery, one of VISAR's goals is to
produce a Digital Elevation Model (DEM) two orders of magnitude better than existing
datasets using its interferometric capabilities.
However, the thick atmosphere strongly influences the radar signal, both attenuating the
signal strength as well as bending the path of the electromagnetic waves (similar to
optical waves bending as they pass between air and water). Without proper compensation,
the DEM would be subject to errors on the order of tens to hundreds of meters, far
above the VERITAS requirements.
This study presents the X-Band Venus Atmosphere Model Permissivity (XVAMP) Python
package, which implements the atmospheric model currently used by the VISAR
engineering team. It is based on the atmospheric mixing model by @duan2010 and is the
first publicly-available code for this topic to the knowledge of the authors.
By making this software open-source, we solicit feedback from the scientific community
already during the implementation phase of the mission, such that the model can truly
represent the current state of our knowledge of the Venus atmosphere as it concerns
radar.

# Statement of need

The VERITAS mission requires a model that for a given viewing geometry (angle and
altitude) of the VISAR instrument onboard the spacecraft looking down at the planet
yields the expected signal attenuation and path delay.
The path delay is the difference between the direct geometric distance between the
spacecraft and the surface and the actual distance traveled by the signal after
accounting for the refraction in the atmosphere (which bends the path).
On Venus, this delay can reach hundreds of meters; an error which would get directly
passed onto the extracted DEM.
The knowledge of the path delay is used onboard during the radar image formation process
(which, for radar, requires the knowledge of the instrument velocity relative to the
ground, which can be measured by the spacecraft but is corrupted by the atmosphere)
and on the ground, where the path delay has to be removed when generating the
DEM (which is derived from the phase difference between the images created
simultaneously by the two VISAR antennas).
Such an atmospheric model is strictly more than a model solely containing the
temperature, pressure, density, and species volume fraction at different altitudes such
as the ones provided by @seiff1985 and @keating1985.
This model is also different from General Circulation Models (GCM), which focus on the
physical properties (e.g., wind speed) and include chemical reactions (in order to
assess the lifetime of chemical species); e.g. @lebonnois2010.
In particular, such a model is also more important for an X-band radar such as the one
used by VERITAS compared to radar instruments of lower frequencies (such as the earlier
Magellan mission by NASA, which produced the highest-resolution DEM of Venus to date,
or the upcoming EnVision mission by ESA; both using S-band frequencies of 2.4 and
3.2 GHz, respectively) as the higher frequency is subject to larger attenuation.

# State of the field

Currently, there exists no publicly-available code to model the effect of the Venus
atmosphere on radar signals to our knowledge.
For Earth, the effect in X-band is much smaller, and there is less of a need to model
the impact, since there are plenty of calibration methods available on Earth that are
not possible on Venus (e.g., radar calibration sites using corner reflectors).
The model used in the planning phase of the VERITAS mission is the proprietary,
script-based code from @duan2010, which is difficult to integrate into other
mission tools and to extend as newer datasets get published (or inferred by the
VERITAS mission).
Therefore, the present model is completely rewritten from scratch in order to
become a state-of-the-art atmospheric model which will be used by the engineering
team moving forward.

# Software design

The model of @duan2010 is based on empirical and analytical formulas which relate the
effect of each chemical species in the atmosphere, scaled by their molar fraction and
given the reigning temperature, pressure, and density profiles, to the effect on the
_polarization_, _absorption_, and _refraction_ of the Venus atmosphere.
In addition, but in different manners, the size, density, and composition of the cloud
layer as well as the electron density in the ionosphere are also incorporated.
All of the aforementioned profiles are the key inputs to the ``Duan2010`` model class,
which is at the core of XVAMP. As a convenience for the user, many published profiles
for the different constituents are included as datasets in the ``references`` module.
During instantiation, all profiles are resampled to common altitude levels.
Then, polarization parameters of the different species can either be loaded or
computed (using data from laboratory measurements).
Evaluating the polarization and absorption for each model constituent at each
altitude level yields the complex permittivity profile, which can be converted to
the refractive index.
At this stage, the model provides integration routines through the absorption and
refraction profiles along the curved path the radar wave takes, yielding the total delay
and attenuation.
Finally, geometric considerations in the ``geometry`` module yield the geometric
range and angles as if no atmosphere was present, which are required for the data
calibration.
Throughout the processing steps of the model, units are preserved using the ``astropy``
package [@theastropycollaboration2022] in order to ensure physical correctness and
easier interpretation of the results.

# Research impact statement

The current impact of the presented model is its usage by the VISAR engineering team.
In particular, this model is used to derive a polynomial approximation of the range
delay and attenuation suitable to be stored in and evaluated by the data processor on
board the spacecraft.
It is also used by mission simulation tools during the design and implementation of the
prototype DEM processing toolchain.
Before and after arrival of the spacecraft at Venus, we furthermore welcome feedback
from members of the wider scientific community to propose methodological improvements
and incorporate new datasets the engineering team may not be aware of.
Finally, we expect end users of the scientific data to be released by VERITAS to
use this model if they intend to reprocess the data themselves.

# Acknowledgments

The authors gratefully acknowledge support from the VERITAS project, especially the
helpful discussions with Scott Hensley and Eva Peral.
This work was partially carried out at the Jet Propulsion Laboratory, California
Institute of Technology, under a contract with the National Aeronautics and Space
Administration.

# AI usage disclosure

No generative AI tools were used in the development of this software, the writing
of this manuscript, or the preparation of supporting materials.

# References
