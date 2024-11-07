# Out-of-focus holography for the Sardinia Radio Telescope

This repository contains ``pyoof-srt``, a heavily customized fork of ``pyoof`` for the Sardinia Radio Telescope.
The software performs out-of-focus holography on astronomical beam maps.

The software is described in this [INAF Technical Report](https://openaccess.inaf.it/handle/20.500.12386/23075).

## Installation

The following installation process has been tested in a workstation running Ubuntu 20.04 and Python 3.9.

1. Create a virtual environment dedicated to `pyoof-srt`, e.g.:
```
mkdir ~/venvs
cd ~/venvs
python -m venv pyoof
```

2. Somewhere else, clone this repository, e.g.:
```
mkdir ~/code
cd ~/code
git clone https://github.com/pinno/pyoof-srt.git
```

3. `pyoof-srt` will be therefore downloaded in the folder `pyoof-srt` within the `~/code` directory.
4. The installation is completed by activating the above virtual environment and by installing with `pip` the `pyoof-srt` package and its dependencies (AstroPy, NumPy, SciPy, Matplotlib and PyYAML):
```
cd ~/code/pyoof-srt
source ~/venvs/pyoof/bin/activate
pip install -r requirements .
```

5. To enable LaTeX syntax in the plots produced by ``pyoof-srt``, a full installation of TeXLive is required:
```
sudo apt install texlive-full
```

## Usage

Once the configuration file has been compiled (in this case `${RUN_DIRECTORY}/run_config.yaml`), `pyoof-srt` can be run with the following command:
```
python ~/code/pyoof-srt/run_pyoof.py ${RUN_DIRECTORY}/run_config.yaml
```

## Sample files

The repository contains the folder `sample_input_files` which provides sample configuration files and input data:
* `opt_vars.yaml` contains the parameters for the optimization problem;
* `run_config.yaml` contains the input variables required for running the out-of-focus holography with real measurements;
* `synthetic_run_config.yaml` contains the input variables required for running the out-of-focus holography with simulated measurements;
* `real_data` is a directory containing a sample of the required three measurement files;
* `synthetic_data` is a directory containing a sample of the required three simulated measurement files.
