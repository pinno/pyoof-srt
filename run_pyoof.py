#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Authors: Tomas Cassanelli and Andrea Pinna

import sys
import pyoof
from pyoof import aperture, telgeometry

# ---------------------------------------------------------------------------- #

def main():

    # Read configuration file
    config_file = sys.argv[1]
    config = pyoof.get_run_config(config_file)

    # Initialize output directory
    pyoof.init_output_dir(config)

    # Initialize logger
    logger = pyoof.init_logger(config)
    logger.info('Starting "pyoof for SRT"...')

    # Read data from input files
    metadata, observation_data = pyoof.extract_data_srt(config, logger)

    # Telescope definition
    telescope = [telgeometry.block_srt_wo_legs,       # Blockage distribution
                 telgeometry.opd_srt,                 # OPD function
                 config['params']['radius'],          # Primary dish radius
                 config['params']['radiotelescope']]  # Telescope name

    # Aperture function
    aperture_function = aperture.illum_gauss

    # Fit beam
    pyoof.fit_beam(data_info=metadata,
                   data_obs=observation_data,
                   method=config['fit']['optimization_method'],
                   order_max=config['fit']['max_order'],
                   illum_func=aperture_function,
                   telescope=telescope,
                   resolution=config['fit']['pixel_resolution'],
                   box_factor=config['fit']['box_factor'],
                   fit_previous=config['fit']['fit_previous'],
                   make_plots=config['output']['plot_figures'],
                   config_params_file=config['fit']['optimization_variables'],
                   config=config,
                   logger=logger)

# ---------------------------------------------------------------------------- #

if __name__ == '__main__':

    main()

# ---------------------------------------------------------------------------- #
