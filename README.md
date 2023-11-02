# OptLevAnalysis
This package contains analysis code used by the Stanford optical levitation lab to search for non-Newtonian interactions at the micron scale.

## Installation
It is highly recommended that you install this inside a virtual environment using `venv` or `conda`. From within the environment, run the following in the `OptLevAnalysis` directory:
```
pip install -e .
```
This will install the package in developer mode, which is currently the only available method of installation.

To uninstall, run:
```
pip uninstall optlevanalysis
```

## Usage
An example notebook that demonstrates how to use the `FileData` and `AggregateData` classes can be found [here](notebooks/example.ipynb).

To run out-of-the-box, se the [`process_dataset`](scripts/process_dataset.py) script. The only required arguments are the path to a folder containing the raw data (including both the `.h5` files and a `config.yaml` file) and a file prefix:
```
cd scripts
python process_dataset.py /path/to/data/ file_prefix
```
This will save the `AggregateData` object to the default folder with a filename generated using the path to the raw data. Some basic figures can then be produced using the [`make_figures`](scripts/make_figures.py) script with the path to the saved `AggregateData` object:
```
python make_figures.py /path/to/aggdat.h5
```
## Related packages
[`opt_lev_analysis`](https://github.com/stanfordbeads/opt_lev_analysis) - the package from which much of this code was originally adapted.

## Authors
Clarke Hardy ([cahardy@stanford.edu](mailto:cahardy@stanford.edu)), adapted from analysis code originally developed by Chas Blakemore and Alexander Rider.