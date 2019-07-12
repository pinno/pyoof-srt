#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
from astropy import units as u
from pyoof import aperture, telgeometry, fit_beam, extract_data_effelsberg

# telescope = [blockage, delta, pr, name]
telescope = dict(
    effelsberg=[
        telgeometry.block_effelsberg,
        telgeometry.opd_effelsberg,
        50. * u.m,  # primary reflector radius
        'effelsberg'
        ],
    manual=[
        telgeometry.block_manual(
            pr=50 * u.m, sr=0 * u.m, a=0 * u.m, L=0 * u.m),
        telgeometry.opd_manual(Fp=30 * u.m, F=387.39435 * u.m),
        50. * u.m,  # primary reflector radius
        'effelsberg partial blockage'
        ]
    )


def fit_beam_effelsberg(pathfits):
    """
    Fits the beam from the OOF holography observations specifically for the
    Effelsberg telescope.
    """

    data_info, data_obs = extract_data_effelsberg(pathfits)

    [name, pthto, obs_object, obs_date, freq, wavel, d_z, meanel] = data_info
    [beam_data, u_data, v_data] = data_obs

    fit_beam(
        data_info=data_info,
        data_obs=[beam_data, u_data, v_data],
        order_max=10,                        # it'll fit from 1 to order_max
        illum_func=aperture.illum_pedestal,  # or illum_gauss
        telescope=telescope['effelsberg'],
        fit_previous=True,                   # True is recommended
        resolution=2 ** 8,                   # standard is 2 ** 8
        box_factor=5,              # box_size = 5 * pr, better pixel resolution
        config_params_file=None,   # default or add path config_file.yaml
        make_plots=False,           # for now testing only the software
        verbose=2,
        work_dir='/scratch/v/vanderli/cassane'
        )


# Example in my machine :)
if __name__ == '__main__':

    fit_beam_effelsberg(
        pathfits='/home/v/vanderli/cassane/data/pyoof/S9mm_3800-3807_3C84_48deg_H6_LON.fits'
        )  # Execute!
