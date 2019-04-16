#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Tomas Cassanelli
import os
from astropy.io import fits
import numpy as np
from astropy.io import ascii
from scipy.constants import c as light_speed
from .aperture import illum_gauss, illum_pedestal

__all__ = [
    'extract_data_pyoof', 'extract_data_srt', 'extract_data_effelsberg',
    'precompute_srt_delta_opd', 'str2LaTeX',
    'store_data_csv', 'uv_ratio', 'illum_strings', 'store_data_ascii'
    ]


def illum_strings(illum_func):
    """
    It assigns string labels to the illumination function. The `~pyoof` package
    has two standard illumination functions, `~pyoof.aperture.illum_pedestal`
    and `~pyoof.aperture.illum_gauss`.

    Parameters
    ----------
    illum_func : `function`
        Illumination function, :math:`E_\\mathrm{a}(x, y)`, to be evaluated
        with ``I_coeff``. The illumination functions available are
        `~pyoof.aperture.illum_pedestal` and `~pyoof.aperture.illum_gauss`.

    Returns
    -------
    illum_name : `str`
        String with the illumination function name.
    taper_name : `str`
        String with the illumination function taper.
    """

    # adding illumination function information
    if illum_func == illum_pedestal:
        illum_name = 'pedestal'
        taper_name = 'c_dB'
    elif illum_func == illum_gauss:
        illum_name = 'gauss'
        taper_name = 'sigma_dB'
    else:
        illum_name = 'manual'
        taper_name = 'taper_dB'

    return illum_name, taper_name


def extract_data_pyoof(pathfits):
    """
    Extracts data from the `~pyoof` default fits file OOF holography
    observations, ready to use for the least squares minimization (see
    `~pyoof.fit_beam`). The fits file has to have the following keys on its
    PrimaryHDU header: ``'FREQ'``, ``'WAVEL'``, ``'MEANEL'``, ``'OBJECT'`` and
    ``'DATE_OBS'``. Besides this three BinTableHDU are required for the data
    itself; ``MINUS OOF``, ``ZERO OOF`` and ``PLUS OOF``. The BinTableHDU
    header has to have the ``'DZ'`` key which includes the radial offset,
    :math:`d_z`. Finally the BinTableHDU has the data files ``'U'``, ``'V'``
    and ``'BEAM'``, which is the :math:`x`- and :math:`y`-axis position in
    radians and the ``'BEAM'`` in a flat array, in mJy.

    Parameters
    ----------
    pathfits : `str`
        Path to the fits file that contains the three beam maps pre-calibrated,
        using the correct PrimaryHDU and the three BinTableHDU (``MINUS OOF``,
        ``ZERO OOF`` and ``PLUS OOF``).

    Returns
    -------
    data_info : `list`
        It contains all extra data besides the beam map. The output
        corresponds to a list,
        ``[name, pthto, obs_object, obs_date, freq, wavel, d_z, meanel]``.
        These are, name of the fits file, paht of the fits file, observed
        object, observation date, frequency, wavelength, radial offset and
        mean elevation, respectively.
    data_obs : `list`
        It contains beam maps and :math:`x`-, and :math:`y`-axis
        (:math:`uv`-plane in Fourier space) data for the least squares
        minimization (see `~pyoof.fit_beam`). The list has the following order
        ``[beam_data, u_data, v_data]``. ``beam_data`` is the three beam
        observations, minus, zero and plus out-of-focus, in a flat array.
        ``u_data`` and ``v_data`` are the beam axes in a flat array.
    """

    hdulist = fits.open(pathfits)  # open fits file, pyoof format
    # path or directory where the fits file is located
    pthto = os.path.split(pathfits)[0]
    # name of the fit file to fit
    name = os.path.split(pathfits)[1][:-5]

    if not all(
            k in hdulist[0].header
            for k in ['FREQ', 'WAVEL', 'MEANEL', 'OBJECT', 'DATE_OBS']
            ):
        raise TypeError('Not all necessary keys found in FITS header.')

    freq = hdulist[0].header['FREQ']
    wavel = hdulist[0].header['WAVEL']
    meanel = hdulist[0].header['MEANEL']
    obs_object = hdulist[0].header['OBJECT']
    obs_date = hdulist[0].header['DATE_OBS']

    beam_data = [hdulist[i].data['BEAM'] for i in range(1, 4)]
    u_data = [hdulist[i].data['U'] for i in range(1, 4)]
    v_data = [hdulist[i].data['V'] for i in range(1, 4)]
    d_z = [hdulist[i].header['DZ'] for i in range(1, 4)]

    data_file = [name, pthto]
    data_info = data_file + [obs_object, obs_date, freq, wavel, d_z, meanel]
    data_obs = [beam_data, u_data, v_data]

    return data_info, data_obs
    
# ---------------------------------------------------------------------------- #
    
def extract_data_srt():
    """
    Read data from files in "srt_data_dir" and generate the FITS file according
    to the format requested by pyoof

    Parameters
    ----------
    
    """

    srt_data_dir = '/home/pinno/oac/pyoof/pyoof/data/srt_data'
    file_names = ['ffmap_-out.grd', 'ffmap_in.grd', 'ffmap_+out.grd']
    d_z = [-0.01153, 0.0, 0.01153]
    frequency = 26.0E+9

    wavelength = light_speed / frequency  # Hz frequency
    
    # Load SRT data
    u_all, v_all, P_all = [], [], []

    for i in range(3):
        data_file = os.path.join(srt_data_dir, file_names[i])
        with open(data_file, 'r') as f:
            data = f.readlines()[9:]
        grid_start, grid_end = [float(k) for k in data[0].split()[0::2]]
        n_u, n_v = [int(k) for k in data[1].split()[0:2]]
        n_values = n_u * n_v
        u = np.linspace(grid_start, grid_end, n_u)
        v = np.linspace(grid_start, grid_end, n_v)
        pr = np.zeros(n_values)
        pi = np.zeros(n_values)
        lines = data[2:]
        for j, line in enumerate(lines):
            pr[j], pi[j] = [float(k) for k in line.split()[0:2]]
        P_tot = np.power(pr,2) + np.power(pi,2)
        
        uu, vv = np.meshgrid(u, v)
        
        u_all.append(uu)
        v_all.append(vv)
        P_all.append(P_tot)

    P_norm = P_all # [P_all[i] / P_all[i].max() for i in range(3)]

    u_to_save = [u_all[i].flatten() for i in range(3)]
    v_to_save = [v_all[i].flatten() for i in range(3)]
    p_to_save = [P_norm[i].flatten() for i in range(3)]

    # Writing default fits file for OOF observations
    table_hdu0 = fits.BinTableHDU.from_columns([
        fits.Column(name='U', format='E', array=u_to_save[0]),
        fits.Column(name='V', format='E', array=v_to_save[0]),
        fits.Column(name='BEAM', format='E', array=p_to_save[0])
        ])

    table_hdu1 = fits.BinTableHDU.from_columns([
        fits.Column(name='U', format='E', array=u_to_save[1]),
        fits.Column(name='V', format='E', array=v_to_save[1]),
        fits.Column(name='BEAM', format='E', array=p_to_save[1])
        ])

    table_hdu2 = fits.BinTableHDU.from_columns([
        fits.Column(name='U', format='E', array=u_to_save[2]),
        fits.Column(name='V', format='E', array=v_to_save[2]),
        fits.Column(name='BEAM', format='E', array=p_to_save[2])
        ])

    # storing data
    if not os.path.exists(os.path.join(srt_data_dir, 'fits_files')):
        os.makedirs(os.path.join(srt_data_dir, 'fits_files'))

    for j in ["%03d" % i for i in range(101)]:
        fits_file = os.path.join(
            srt_data_dir, 'fits_files', 'test_{}.fits'.format(j)
            )
        if not os.path.exists(fits_file):
            print('Writing data to {}'.format(fits_file))

            prihdr = fits.Header()
            prihdr['FREQ'] = frequency
            prihdr['WAVEL'] = wavelength
            prihdr['MEANEL'] = 0
            prihdr['OBJECT'] = 'test_{}'.format(j)
            prihdr['DATE_OBS'] = 'test_{}'.format(j)
            prihdr['COMMENT'] = 'SRT data'
            prihdr['AUTHOR'] = 'Andrea Pinna'
            prihdu = fits.PrimaryHDU(header=prihdr)
            pyoof_fits = fits.HDUList(
                [prihdu, table_hdu0, table_hdu1, table_hdu2]
                )

            for i in range(3):
                pyoof_fits[i + 1].header['DZ'] = d_z[i]

            pyoof_fits[1].name = 'MINUS OOF'
            pyoof_fits[2].name = 'ZERO OOF'
            pyoof_fits[3].name = 'PLUS OOF'

            pyoof_fits.writeto(fits_file)

            break
            
    data_info, data_obs = extract_data_pyoof(fits_file)

    return data_info, data_obs
    

def precompute_srt_delta_opd(data_info, telgeo, resolution, box_factor):

    pr = telgeo[2]
    box_size = pr * box_factor

    x = np.linspace(-box_size, box_size, resolution)
    y = x
    x_grid, y_grid = np.meshgrid(x, y)
    
    # Cassegrain/Gregorian (at focus) telescope
    Fp = 21.0236  # Focus primary reflector m
    F = 149.76  # Total focus Gregorian telescope m
    # F/D = 2.34
    r = np.sqrt(x_grid ** 2 + y_grid ** 2)  # polar coordinates radius
    a = r / (2 * Fp)
    b = r / (2 * F)
    
    # Polynomial fitting
    R = np.asarray([
         [-1.438998E-09, -2.706440E-10, -7.098885E-14],
         [ 1.467218E-07,  2.888850E-08,  7.767377E-12],
         [-5.904197E-06, -1.234467E-06, -3.330234E-10],
         [ 1.188731E-04,  2.717683E-05,  7.038221E-09],
         [-1.239164E-03, -3.357801E-04, -7.544711E-08],
         [ 6.017033E-03,  2.461091E-03,  3.776198E-07],
         [-1.323805E-02, -9.312688E-03, -7.093549E-07],
         [ 5.676379E-03,  8.636208E-04,  2.160941E-07]])
    # Degree of polynomial
    N1 = 8
    N2 = 3

    delta_opd = [[], [], []]
    for i_dz in range(3):
        d_z = data_info[6][i_dz]
    
      
        coeff = np.zeros(N1)
        for k in range(N1-1,-1,-1):
            for i in range(N2-1,-1,-1):
                coeff[k] = coeff[k] + R[k,i] * np.power(d_z, N2-i)

        delta_opd[i_dz] = np.zeros(r.shape)
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                for k in range(N1-1,-1,-1):
                    delta_opd[i_dz][i,j] = delta_opd[i_dz][i,j] + \
                                           coeff[k] * np.power(r[i,j], N1-k)

    return delta_opd


def extract_data_effelsberg(pathfits):
    """
    Extracts data from the Effelsberg OOF holography observations, ready to
    use for the least squares minimization. This function will only work for
    the Effelsberg telescope beam maps.

    Parameters
    ----------
    pathfits : `str`
        Path to the fits file that contains the three beam maps pre-calibrated,
        from the Effelsberg telescope.

    Returns
    -------
    data_info : `list`
        It contains all extra data besides the beam map. The output
        corresponds to a list,
        ``[name, pthto, obs_object, obs_date, freq, wavel, d_z, meanel]``.
        These are, name of the fits file, paht of the fits file, observed
        object, observation date, frequency, wavelength, radial offset and
        mean elevation, respectively.
    data_obs : `list`
        It contains beam maps and :math:`x`-, and :math:`y`-axis
        (:math:`uv`-plane in Fourier space) data for the least squares
        minimization (see `~pyoof.fit_beam`). The list has the following order
        ``[beam_data, u_data, v_data]``. ``beam_data`` is the three beam
        observations, minus, zero and plus out-of-focus, in a flat array.
        ``u_data`` and ``v_data`` are the beam axes in a flat array.
    """

    # Opening fits file with astropy
    try:
        # main fits file with the OOF holography format
        hdulist = fits.open(pathfits)

        # Observation frequency
        freq = hdulist[0].header['FREQ']  # Hz
        wavel = light_speed / freq

        # Mean elevation
        meanel = hdulist[0].header['MEANEL']  # Degrees
        obs_object = hdulist[0].header['OBJECT']  # observed object
        obs_date = hdulist[0].header['DATE_OBS']  # observation date
        d_z = [hdulist[i].header['DZ'] for i in range(1, 4)][::-1]

        beam_data = [hdulist[i].data['fnu'] for i in range(1, 4)][::-1]
        u_data = [hdulist[i].data['DX'] for i in range(1, 4)][::-1]
        v_data = [hdulist[i].data['DY'] for i in range(1, 4)][::-1]

    except FileNotFoundError:
        print('Fits file does not exists in directory: ' + pathfits)
    except NameError:
        print('Fits file does not have the OOF holography format')

    else:
        pass

    # Permuting the position to provide same as main_functions
    beam_data.insert(1, beam_data.pop(2))
    u_data.insert(1, u_data.pop(2))
    v_data.insert(1, v_data.pop(2))
    d_z.insert(1, d_z.pop(2))

    # path or directory where the fits file is located
    pthto = os.path.split(pathfits)[0]
    # name of the fit file to fit
    name = os.path.split(pathfits)[1][:-5]

    data_info = [name, pthto, obs_object, obs_date, freq, wavel, d_z, meanel]
    data_obs = [beam_data, u_data, v_data]

    return data_info, data_obs


def str2LaTeX(python_string):
    """
    Function that solves the underscore problem in a python string to
    :math:`\LaTeX` string.

    Parameters
    ----------
    python_string : `str`
        String that needs to be changed.

    Returns
    -------
    LaTeX_string : `str`
        String with the new underscore symbol.
    """

    string_list = list(python_string)
    for idx, string in enumerate(string_list):
        if string_list[idx] == '_':
            string_list[idx] = '\\_'

    LaTeX_string = ''.join(string_list)

    return LaTeX_string


def store_data_csv(name, name_dir, order, save_to_csv):
    """
    Stores all important information in a csv file after the least squares
    minimization has finished, `~pyoof.fit_beam`. All data will be stores in
    the ``pyoof_out/name`` directory, with ``name`` the name of the fits file.

    Parameters
    ----------
    name : `str`
        File name of the fits file to be optimized.
    name_dir : `str`
        Path to store all the csv files. The files will depend on the order of
        the Zernike circle polynomial.
    order : `int`
        Order used for the Zernike circle polynomial, :math:`n`.
    save_to_csv : `list`
        It contains all data that will be stored. The list must have the
        following order, ``[beam_data, u_data, v_data, res_optim, jac_optim,
        grad_optim, phase, cov_ptrue, corr_ptrue]``.
    """

    headers = [
        'Normalized beam', 'u vector radians', 'v vector radians', 'Residual',
        'Jacobian', 'Gradient', 'Phase primary reflector radians', 'Deformation Error',
        'Variance-Covariance matrix (first row fitted parameters idx)',
        'Correlation matrix (first row fitted parameters idx)'
        ]

    fnames = [
        '/beam_data.csv', '/u_data.csv', '/v_data.csv',
        '/res_n{}.csv'.format(order), '/jac_n{}.csv'.format(order),
        '/grad_n{}.csv'.format(order), '/phase_n{}.csv'.format(order),
        '/error_n{}.csv'.format(order),
        '/cov_n{}.csv'.format(order), '/corr_n{}.csv'.format(order)
        ]

    if order != 1:
        headers = headers[3:]
        fnames = fnames[3:]
        save_to_csv = save_to_csv[3:]

    for fname, header, file in zip(fnames, headers, save_to_csv):
        np.savetxt(
            fname=name_dir + fname,
            X=file,
            header=header + ' ' + name
            )


def store_data_ascii(
    name, name_dir, taper_name, order, params_solution, params_init
        ):
    """
    Stores in an ascii format the parameters found by the least squares
    minimization (see `~pyoof.fit_beam`).

    Parameters
    ----------
    name : `str`
        File name of the fits file to be optimized.
    name_dir : `str`
        Path to store all the csv files. The files will depend on the order of
        the Zernike circle polynomial.
    taper_name : `str`
        Name of the illumination function taper.
    order : `int`
        Order used for the Zernike circle polynomial, :math:`n`.
    params_solution : `~numpy.ndarray`
        Contains the best fitted parameters, the illumination function
        coefficients, ``I_coeff`` and the Zernike circle polynomial
        coefficients, ``K_coeff`` in one array.
    params_init : `~numpt.ndarray`
        Contains the initial parameters used in the least squares minimization
        to start finding the best fitted combination of them.
    """

    n = order
    N_K_coeff = (n + 1) * (n + 2) // 2

    # Making nice table :)
    ln = [(j, i) for i in range(0, n + 1) for j in range(-i, i + 1, 2)]
    L = np.array(ln)[:, 0]
    N = np.array(ln)[:, 1]

    params_names = ['i_amp', taper_name, 'x_0', 'y_0']
    for i in range(N_K_coeff):
        params_names.append('K({}, {})'.format(N[i], L[i]))

    # To store fit information and found parameters in ascii file
    ascii.write(
        table=[params_names, params_solution, params_init],
        output=name_dir + '/fitpar_n{}.csv'.format(n),
        names=['parname', 'parfit', 'parinit'],
        comment='Fitted parameters ' + name
        )


def uv_ratio(u, v):
    """
    Calculates the aspect ratio for the 3 power pattern plots, plus some
    corrections for the text on it. Used in the `function` `~pyoof.plot_beam`
    and `~pyoof.plot_data`

    Parameters
    ----------
    u : `~numpy.ndarray`
        Spatial frequencies from the power pattern, usually in degrees.
    v : `~numpy.ndarray`
        Spatial frequencies from the power pattern, usually in degrees.

    Returns
    -------
    plot_width : `float`
        Width for the power pattern figure.
    plot_height : `float`
        Height for the power pattern figure.
    """

    ratio = (v.max() - v.min()) / (3 * (u.max() - u.min()))
    width = 14
    height = width * (ratio) + 0.2

    return width, height
