# wolfplot
![wolfplot-plot](https://github.com/Wolfgang90/wolfplot/blob/master/examples/05_Scatter.png "Example plot")
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
In order to create and save a plot, there are generally three steps after having imported the `Plot` module.

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
#### Barplot
![Barplot](https://github.com/Wolfgang90/wolfplot/blob/master/examples/01_Bar.png "Barplot")
Example code:
```
plot = Plot(padding=(0.27, 0.1, 0.95, 0.95),
            y_label = "Document type [#]", 
            y_format_type = "number_de", 
            y_lim = (0, 400),
            x_rotation = 45,
            x_horizontal_alignment = 'right',
            y_tick_width = 50,
            y_minor_tick_width = 25,
            x_label="Document type", 
            grid_ticktype='minor',
            grid_direction='y',
           )

fig = plot.plot_bar(data = eventlog.doctype)

plot.save_figure('./examples/01_Bar.png')
```

#### Hbarplot
![Hbarplot](https://github.com/Wolfgang90/wolfplot/blob/master/examples/02_Hbar.png "Hbarplot")
Example code:
```
plot = Plot(padding=(0.1, 0.25, 0.95, 0.95),
            x_label = "Document type [#]", 
            x_format_type = "number_de", 
            x_lim = (0, 400),
            y_rotation = 0,
            y_horizontal_alignment = 'right',
            x_tick_width = 50,
            x_minor_tick_width = 25,
            y_label="Document type", 
            grid_ticktype='minor',
            grid_direction='x',
           )

fig = plot.plot_hbar(data = eventlog.doctype)

plot.save_figure('./examples/02_Hbar.png')
```

#### Histogramplot
![Histogramplot](https://github.com/Wolfgang90/wolfplot/blob/master/examples/03_Histogram.png "Histogramplot")
Example code:
```
plot = Plot(padding=(0.1, 0.12, 0.95, 0.95),
            x_label = "Parcel size [Acres]", 
            x_tick_width = 100,
            y_label="Parcels [#]", 
            y_format_type = "number_de",            
            y_lim = (0, 20000),
            y_tick_width = 5000,
            y_minor_tick_width = 1000,
            grid_ticktype='minor',
            bins = (0,650,25)
           )

fig = plot.plot_hist(data = casetabel.area)

plot.save_figure('./examples/03_Histogram.png')
```

#### Lineplot
![Lineplot](https://github.com/Wolfgang90/wolfplot/blob/master/examples/04_Linechart.png "Lineplot")
Example code:
```
plot = Plot(x_label = "Year", 
            x_rotation = 45,
            x_horizontal_alignment = 'right',
            y_label="Cases [#]", 
            y_format_type = "number_de",            
            y_lim = (0, 5000),
            y_tick_width = 500,
            y_minor_tick_width = 250,
            padding=(0.14, 0.1, 0.95, 0.95)
           )

fig = plot.plot_line_distinct(data = casetabel_year_dep.sort_values(by = 'year'), 
                              by_column_name = "department",
                              key_column_names="year", 
                              values_column_names = "cases"
                             )

plot.save_figure('./examples/04_Linechart.png')
```

#### Scatterplot
![Scatterplot](https://github.com/Wolfgang90/wolfplot/blob/master/examples/05_Scatter.png "Scatterplot")
Example code:
```
plot = Plot(padding=(0.1, 0.13, 0.95, 0.95),
            x_label = "Parcel size [Acres]",
            x_lim = (0,500),
            x_tick_width = 50,
            x_rotation = 0,
            x_horizontal_alignment = 'center',
            x_format_type = "number_de",  
            y_lim = (0,200000),
            y_label="Payment [€]", 
            y_format_type = "number_de"
           )


fig = plot.plot_scatter(data = casetabel, x_column_names = 'area', 
                        y_column_names = 'payment_actual0', 
                        regression=False, s=10, alpha = 0.3)

plot.save_figure('./examples/05_Scatter.png')
```

#### Boxplot
![Boxplot](https://github.com/Wolfgang90/wolfplot/blob/master/examples/06_Boxplot.png "Boxplot")
Example code:
```
plot = Plot(x_label = "year",
            x_lim = (0,30),
            x_tick_width = 2,
            x_rotation = 0,
            x_horizontal_alignment = 'center',
            y_lim = (0,120),
            y_label="Parcel size [Acres]", 
            y_format_type = "number_de"           
           )


fig = plot.plot_boxplot(data = casetabel, 
                        by_column_name = 'year',
                        column_names = 'area')

plot.save_figure('./wolfplot_examples/06_Boxplot.png')
```
