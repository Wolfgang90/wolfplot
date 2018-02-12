# wolfplot
## Purpose
`wolfplot` is a high-level plotting framework for `python` built on top of `matplotlib`. It is primarily designed for the use in process mining projects

## Getting started
In order do initialize your work environment, clone or download wolfplot onto your machine.

All files required for using `wolfplot` are located in the folder `src`. It contains:
* `matplotlibrc`: `matplotlib` style-sheet for to apply pre-defined style-patterns on `matplotlib` graphs
* `eventlog.py`: `python` file containing a class for importing process mining eventlogs into `pandas` dataframes
* `wolfplot.py`: `python` file containing the plotting class built on top of `matplotlib`

Once downloaded one can create a `.ipynb` file in the `wolfplot` directory, open it from `wolfplot` directory  via the command `jupyter notebook <filename>.ipynb` and start using the framework.

To import all required packages into the notebook, make sure to execute the following in the first code cell:

```from src.eventlog import EventLog
from src.wolfplot import Plot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('src/matplotlibmetarc')

%matplotlib inline
```

For a documentation of the available classes and methods, please reference to the documentation in `eventlog.py` and `wolfplot.py`. 
