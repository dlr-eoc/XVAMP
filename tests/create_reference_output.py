"""
Script that creates a default model and saves its output.
Can be used to check for unintended consequences when updating model code.
"""

# standard imports
import os
import numpy as np
from astropy.units import Quantity
from astropy.table import QTable

# package import
from xvamp.model import Duan2010
from xvamp.utils.io import write_polarization_parameters
from xvamp.geometry import geometry_from_central_angle, get_cross_track_displacement

# start script
if __name__ == "__main__":

    # make an output folder
    os.makedirs("reference_output")

    # instantiate all-defaults model, only regenerating the polarization
    # parameters to make sure we're not using loaded ones
    model = Duan2010(load_polarization_parameters=False)

    # part 1: save all things already computed in the model
    # combine all Quantities into a saveable table
    quantities = QTable(
        {
            field: attr
            for field, attr in model.__dict__.items()
            if isinstance(attr, Quantity)
        }
    )
    quantities.write("reference_output/quantities.fits")
    # save all QTables directly
    for field, attr in model.__dict__.items():
        if isinstance(attr, QTable):
            attr.write(f"reference_output/{field}.fits")
    # now only the polarization parameters are left
    write_polarization_parameters(
        model.polarization_parameters, "reference_output/polarization_parameters.toml"
    )

    # part 2: evaluate the model for a set of viewing geometries
    # range of test values
    apparent_look_angle = Quantity(np.linspace(28, 32, num=5), "deg")
    height_terrain = Quantity(np.linspace(-6, 16, num=23), "km")
    height_platform = Quantity(220, "km")
    # combine the two varying quantities in a single grid
    grid_terrain, grid_look = np.meshgrid(
        height_terrain.to_value("km"), apparent_look_angle.to_value("rad")
    )
    # get profile-integrated values
    apparent_range, attenuation, central_angle, apparent_incidence_angle = (
        model.get_range_attenuation_angles(
            grid_look.ravel(), grid_terrain.ravel(), height_platform
        )
    )
    # use law of cosines to get geometric quantities
    geometric_range, geometric_look_angle, geometric_incidence_angle = (
        geometry_from_central_angle(
            central_angle, grid_terrain.ravel(), height_platform
        )
    )
    # use it again to get the cross-track displacement
    cross_track_displacement = get_cross_track_displacement(
        np.repeat(apparent_look_angle[:, None], height_terrain.size, axis=1).ravel(),
        central_angle,
        grid_terrain.ravel(),
        height_platform,
    )
    # combine all into a single QTable and save
    profile_values = QTable(
        [
            Quantity(grid_look.ravel(), "rad"),
            Quantity(grid_terrain.ravel(), "km"),
            apparent_range,
            attenuation,
            central_angle,
            apparent_incidence_angle,
            geometric_range,
            geometric_look_angle,
            geometric_incidence_angle,
            cross_track_displacement,
        ]
    )
    profile_values.write("reference_output/profile_values.fits")
