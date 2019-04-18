#!/usr/bin/env python


import logging
import os
import sys

import pyoof
from pyoof import aperture, telgeometry


if __name__ == '__main__':

    logger = logging.getLogger('pyoof')
    logger.setLevel(logging.DEBUG)

    if os.path.expanduser('~') == '/home/franco':
         data_path = os.path.join(os.path.expanduser('~'), 'pyoof_data')
    else:
         data_path = os.path.join(os.path.expanduser('~'), 'oac/pyoof_data')

    log_file = os.path.join(data_path, 'pyoof.log')    

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info('Starting "pyoof"...')


    if sys.argv[1] == 'SRT':
    
        data_directory = os.path.join(data_path, 'srt_data')
        dz = 0.01153
        frequency = 26.0E+9
        data_info, data_obs = pyoof.extract_data_srt(data_directory, dz, frequency)
        
        # SRT telescope definition
        telescope = [
            telgeometry.block_srt_wo_legs,  # Blockage distribution
            telgeometry.opd_srt,            # OPD function
            32.004,                         # Primary dish radius
            sys.argv[1]                     # Telescope name
        ]
        aperture_function = aperture.illum_gauss
    
    
    elif sys.argv[1] == 'Effelsberg':
    
        # Extracting observation data and important information
        data_file = os.path.join(data_path, 'example0.fits')
        data_info, data_obs = pyoof.extract_data_pyoof(data_file)

        # Effelsberg telescope definition
        telescope = [
            telgeometry.block_effelsberg,  # Blockage distribution
            telgeometry.opd_effelsberg,    # OPD function
            50.,                           # Primary dish radius
            sys.argv[1]                    # Telescope name
        ]
        aperture_function = aperture.illum_pedestal
        
    else:
    
        logger.error('"{}" telescope does not exist!\nTry "SRT" or "Effelsberg"!'.format(sys.argv[1]))
        sys.exit()

    

    pyoof.fit_beam(
        data_info=data_info,                   # information
        data_obs=data_obs,                     # observed beam
        method='trf',                          # opt. algorithm 'trf', 'lm' or 'dogbox'
        order_max=5,                           # it will fit from 1 to order_max
        illum_func=aperture_function,          # or illum_gauss
        telescope=telescope,                   # telescope properties
        resolution=2 ** 8,                     # standard is 2 ** 8
        box_factor=5,                          # box_size = 5 * pr, pixel resolution
        )

