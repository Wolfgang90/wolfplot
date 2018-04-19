# wolfplot
## Purpose
`wolfplot` is a high-level plotting framework for `python` built on top of `matplotlib`. It is primarily designed for the use in process mining projects and focuses on data in `pandas`-datatypes.

## Getting started
In order to install wolfplot clone wolfplot to your machine:
```
git clone https://github.com/Wolfgang90/wolfplot.git
```

Install wolfplot from the package directory `wolfplot` with the command:
```
pip install .
```

You can now use the package from any direcory on your machine.
For importing the `Plot` module of wolfplot type the following import statement:
```
from wolfplot import Plot
```
For importing the `EventLog` module of wolfplot type the following import statement:
```
from wolfplot import EventLog
```

Currently the stylesheet is not implemented in the package. Therefore you need to place the `matplotlibmetarc` in your current working directory and type the following commands:
```
import matplotlib.pyplot as plt
plt.style.use('./matplotlibmetarc')
```

To output plots in Jupyter Notebooks additionally type:
```
%matplotlib inline
```

If you want to upgrade wolfplot execute the following command on the command line from the directory `wolfplot`:
```
git pull origin master
```
Subsequently upgrade the package with:
```
pip install . --upgrade
```

## Package setup


All files required for using `wolfplot` are located in the subfolder `wolfplot/wolfplot`. It contains:
* `matplotlibrc`: `matplotlib` style-sheet for to apply pre-defined style-patterns on `matplotlib` graphs
* `eventlog.py`: `python` file containing a class for importing process mining eventlogs into `pandas` dataframes
* `wolfplot.py`: `python` file containing the plotting class built on top of `matplotlib`
