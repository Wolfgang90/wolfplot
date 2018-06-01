# wolfplot
![wolfplot-plot](https://github.com/Wolfgang90/wolfplot/tree/master/examples/01_Bar.svg?sanitize=true)
## Purpose
`wolfplot` is a high-level plotting framework for `python` built on top of `matplotlib`. It is primarily designed for the use in process mining projects and focuses on data in `pandas`-datatypes.

## Installing wolfplot
In order to install wolfplot clone wolfplot to your machine:
```
git clone https://github.com/Wolfgang90/wolfplot.git
```

Install wolfplot from the package directory `wolfplot` with the command:
```
pip install .
```

If you want to upgrade wolfplot execute the following command on the command line from the directory `wolfplot`:
```
git pull origin master
```
Subsequently upgrade the package with:
```
pip install . --upgrade
```

## Using wolfplot

### Import wolfplot
For importing the `Plot` module of wolfplot type the following import statement:
```
from wolfplot import Plot
```
For importing the `EventLog` module of wolfplot type the following import statement:
```
from wolfplot import EventLog
```

### Create and save plots
In order to create a plot and save it there are generally three steps.

1. Create an instance of `Plot()`, eg. `plot = Plot()`. Make sure to hand over the chart attributes like `y_label` to `Plot()`.
2. Create your plot by calling the method of the desired plot type from the instance of `Plot()`, e.g. `fig = plot.plot_bar()`. Make sure to hand over your data to the method.
3. Save the graph by calling the method `save_figure()` from the instance of `Plot()`, like `plot.save_figure()`

### Overview of plot types
| Plot type        | Call           |
| ------------- |:-------------|
| Barplot      | `.plot_bar()` |
| Hbarplot      | `.plot_hbar()` |
| Histogramplot | `.plot_hist()` |
| Lineplot | `.plot_line_distinct()` |
| Scatterplot | `.plot_scatter()` |
| Boxplot | `.plot_boxplot()` |

### Example plots
