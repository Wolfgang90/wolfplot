import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import math
import datetime




class Plot:    
    def __init__(self, 
                 width = 6, 
                 height = 4.5, 
                 x_label = "", 
                 x_lim=None, 
                 x_tick_width = None, 
                 x_minor_tick_width = None,              
                 y_label="", 
                 y_lim=None, 
                 y_tick_width = None, 
                 y_minor_tick_width = None,

                 
                 
                 x_format_type = None,
                 x_rotation=45, 
                 x_horizontal_alignment="right", 
                 y_format_type = None,
                 y_rotation=0, 
                 y_horizontal_alignment="right",
                 
                 grid_ticktype="major",
                 grid_direction="y",
                 bins = None):
        
        """
            Input parameters:
            width (type: float): width of overall figure (default: 6)
            height (type: float): height of overall figure (default: 4.5)
            x_label(type: string): label displayed on the x-axis (default: empty string)
            x_lim (type: tuple): limits for x-axis in the format (lower limit, upper limit)
            x_tick_width (type: float): width of the major x-tick labels (default: None)
            x_minor_tick_width (type: float): width of the minor x-tick labels, which should be a possible divider 
                                              of the major tick width (default: None)

            y_label(type: string): label displayed on the y-axis (default: empty string)
            y_lim (type: tuple): limits for x-axis in the format (lower limit, upper limit)
            y_tick_width (type: float): width of the major y-tick labels (default: None)
            y_minor_tick_width (type: float): width of the minor y-tick labels, which should be a possible divider 
                                              of the major tick width (default: None)
        
            x_format_type (type: string): format for the x-axis tick labels (possible values: 'number_de')
            x_rotation (type: int): rotation for the x-axis tick labels (default: 45)
            xhorizontalalignement(type: string): alignement of the x-axis tick labels 
                                                 (possible: 'right', 'center', 'left'; default: right)
            y_format_type (type: string): format for the y-axis tick labels (possible values: 'number_de')
            y_rotation (type: int): rotation for the y-axis tick labels (default: 45)
            yhorizontalalignement(type: string): alignement of the y-axis tick labels 
                                                 (possible: 'right', 'center', 'left'; default: right)
            
            grid_ticktype (type: string): describes whether grid lines are displayed on the major or the minor ticks 
                                          (possible: 'major', 'minor'; default: major)
            grid_direction (type: string): describes whether grid lines are displayed on x, y or both ticks 
                                          (possible: 'x','y','xy'; default: 'y')
            bins (type: tuple(<lower end>,<higher end>,<bin width>)):
    
        """
        
        
        # Initialize variables
        self.width = width
        self.height = height
        self.x_label = x_label
        self.x_format_type = x_format_type
        self.x_lim = x_lim
        self.x_tick_width = x_tick_width
        self.x_minor_tick_width = x_minor_tick_width
        self.x_rotation = x_rotation
        self.x_horizontal_alignment = x_horizontal_alignment
        self.y_label = y_label
        self.y_format_type = y_format_type
        self.y_lim = y_lim
        self.y_tick_width = y_tick_width
        self.y_minor_tick_width =y_minor_tick_width
        self.y_rotation = y_rotation
        self.y_horizontal_alignment = y_horizontal_alignment
        self.grid_direction = grid_direction
        self.grid_ticktype = grid_ticktype       

        if bins:
            self.x_lim = (bins[0],bins[1])
            self.x_minor_tick_width = bins[2]

            self.bins = np.arange(bins[0],bins[1]+1,bins[2]).tolist()

            # Determine number of y-values and create vector representing y-Axis
            self.buckets = np.arange(bins[0],bins[1],bins[2]).tolist()     

        
        plt.style.use('src/matplotlibmetarc')
        
        
    def _linear_regression(self,x,y):

        x_max = x.max()
        x_min = x.min()

        x_pred = np.arange(x_min, x_max+1, (x_max-x_min)/20)


        x = np.array(x[:, np.newaxis], dtype='float64')
        y = np.array(y[:, np.newaxis], dtype='float64')
        x_pred = np.array(x_pred[:,np.newaxis], dtype='float64')
        
        # Create linear regression object
        regr = linear_model.LinearRegression()

        # Train the model using the full set
        regr.fit(x, y)
        
        # Make predictions using the full set
        y_pred = regr.predict(x_pred)
        
        return x_pred,y_pred



    def _create_figure_and_axes(self, fig_kind = "single", n_data = 1, axes_per_row=4):
        """
            Input parameters:
            fig_kind (type: string): Describes whether data should be plotted on one or on multiple graphs
            number_of_axes (int): Only relevant for multiple figure, describes number of axes
            axes_per_row (int): Describes number of axes per graph row
        """
        if fig_kind=="single":
            self.fig, ax = plt.subplots(1,1,figsize=(self.width,self.height))
            n_plots = 1

        if fig_kind == "multiple":            
            self.fig, ax = plt.subplots(int(n_data/axes_per_row)+1,axes_per_row, figsize=(self.width,self.height))
            n_plots = (int(n_data/axes_per_row)+1) * axes_per_row         


            # Set subplot axis without content to invisible
            gap = n_plots - n_data
            if gap:
                for i in range(gap):
                    ax[int(n_data/axes_per_row)][axes_per_row-i-1].axis('off')

        return ax
            

        


    def _set_title(self, ax,title):

        # Plot chart title
        ax.set_title(title,y=1.00, loc='left')
        return ax


    def _set_axis(self, ax, axis_pos=None, axis_ticklabels=None, axis = None, axis_plot = True):

        if axis_plot:

            # Set axis label
            exec("ax.set_" + axis + "label(self." + axis + "_label)")
        
            # Set position of tick labels
            if axis_ticklabels:
                exec("ax.set_" + axis + "ticklabels(axis_ticklabels)")
            # Set 1000-seperator
            if eval("self." + axis + "_format_type"):

                exec("ax." + axis + "axis.set_major_formatter(self._format_ticker(axis = '" + axis + "'))")

            # Set orientation and alignment of axis tick labels
            exec("plt.setp(ax.get_" + axis + "ticklabels(), rotation=self." + axis + "_rotation, horizontalalignment=self." + axis + "_horizontal_alignment)")

        elif not axis_plot:
            exec("ax.get_" + axis + "axis().set_ticklabels([])")



        # Set position of axis ticks (required for axis with ordinal data)
        if axis_pos:
            exec("ax.set_" + axis + "ticks(axis_pos)")

        # Set axis lower and upper limit
        if eval("self." + axis + "_lim"):
            exec("ax.set_" + axis + "lim(self." + axis + "_lim[0], self." + axis + "_lim[1], auto=False)")


        # Set axis major tick width
        if eval("self." + axis + "_tick_width"):
            if not eval("self." + axis + "_lim"):
                print(axis + "_lim is missing and required to set " + axis + "_tick_width")
            # Set major ticks
            exec("ax.set_" + axis + "ticks(np.arange(self." + axis + "_lim[0], self." + axis + "_lim[1]+1, self." + axis + "_tick_width))")

        # Set axis minor tick width
        if eval("self." + axis +  "_minor_tick_width"):
            exec("minorLocator = ticker.MultipleLocator(self." + axis + "_minor_tick_width)")
            exec("ax." + axis + "axis.set_minor_locator(minorLocator)")

        return ax



    def _format_ticker(self, axis):
        """
            Number formatting function to be handed over to FuncFormatter
            Input paramters:
            number(type: number-like): The number to be formatted            
        """

        if eval("self." + axis +"_format_type == 'number_de'"):
            major_formatter = ticker.FuncFormatter(lambda x, loc: "{:,}".format(int(x)).replace(",","x").replace(".",",").replace("x","."))

        if eval("self." + axis + "_format_type == 'timedelta'"):
            major_formatter = ticker.FuncFormatter(lambda x, loc: str(datetime.timedelta(seconds=x)))

        return major_formatter
    


    def _create_bar(self, ax, keys, values, data_label = None):

        # Determine number of y-values and create vector representing y-Axis
        x_pos = np.arange(len(keys)).tolist()

        # Set x and y axis
        ax = self._set_axis(ax, axis_pos=x_pos, axis_ticklabels = keys.tolist(), axis="x")  
        ax = self._set_axis(ax, axis="y") 
        
        # Set grid lines
        ax.grid(axis=self.grid_direction,which=self.grid_ticktype)
           
        # set lengths of y-tick parameters to 0
        ax.tick_params(axis='x', length=0)

        # Create data part of the chart
        ax.bar(x_pos, values,1, label = data_label)
        
        return ax




    def _create_hbar(self, ax, keys, values, data_label = None):

        # Determine number of y-values and create vector representing y-Axis
        y_pos = np.arange(len(keys)).tolist()


        # Set x and y axis
        ax = self._set_axis(ax, axis = "x")
        ax = self._set_axis(ax, axis_pos=y_pos, axis_ticklabels = keys.tolist(), axis = "y")
        
        # Set grid lines
        ax.grid(axis=self.grid_direction,which=self.grid_ticktype)

        # set lengths of y-tick parameters to 0
        ax.tick_params(axis='y', length=0)

        # Create data part of the chart
        ax.barh(y_pos, values,1, label = data_label)

        return ax


    def _create_histogram(self,ax,x):
        
        # Set x and y axis
        ax = self._set_axis(ax, axis_pos=self.buckets, axis="x")
        ax = self._set_axis(ax, axis = "y")
        
        # Set grid lines
        ax.grid(axis=self.grid_direction,which=self.grid_ticktype) 
        
        # Create data part of the chart
        ax.hist(x, self.bins)
        
        return ax
    
    def _create_scatter(self, ax, x, y, data_label = None, title = False, y_axis_plot = True, regression = False):
        
        if title:
            ax = self._set_title(ax, data_label)
       
        ax = self._set_axis(ax, axis = "x")
        
        ax = self._set_axis(ax, axis_plot = y_axis_plot, axis="y")
        
        # Set grid lines
        ax.grid(axis=self.grid_direction,which=self.grid_ticktype)        
        
        ax.scatter(x ,y, label=data_label, s = 3, linewidth = 0.0)
        
        if regression:
            x_reg, y_reg = self._linear_regression(x,y)
            ax.plot(x_reg,y_reg)
        
        return ax


    def _create_boxplot(self, ax, x_data, data_label, display_info=True):

        # Determine number of y-values and create vector representing y-Axis
        x_pos = np.arange(len(data_label)).tolist()

        # Set x and y axis
        ax = self._set_axis(ax,axis_pos=x_pos, axis_ticklabels = data_label, axis = "x")
        ax = self._set_axis(ax, axis = "y")

        # Set grid lines
        ax.grid(axis=self.grid_direction,which=self.grid_ticktype)
           
        # set lengths of y-tick parameters to 0
        ax.tick_params(axis='x', length=0)

        ax.boxplot(x_data)
        x_data = np.swapaxes(x_data,0,1)


        if display_info:
            for i,_ in enumerate(data_label):
                median = float(np.median(x_data[i]))
                mean = float(np.mean(x_data[i]))
                ax.text(i+1, ((ax.get_ylim())[1])*1.01, "Mean:\n" + str(round(mean,1)) +  "\nMedian:\n" + str(round(median,1)), horizontalalignment = 'center',fontsize="xx-small", weight = 'semibold')
                

        ax.set_xticklabels(data_label)

        return ax


    def _series_to_arrays(self, pd_series, value_counts = False, ascending = False):

        # Value counts only to be set for bar charts and hbarcharts
        if value_counts:
            pd_series = pd_series.value_counts(ascending=ascending)

        keys = np.expand_dims(pd_series.keys().values, axis = 0)
        values = np.expand_dims(pd_series.values, axis = 0)
        minimum = pd_series.min()
        maximum = pd_series.max()
        return keys, values, minimum, maximum



    def _dataframe_to_arrays(self, pd_data_frame):
        column_titles = pd_data_frame.columns.values
        for i, title in enumerate(column_titles):
            if i == 0:
                data = np.expand_dims(pd_data_frame[title].values, axis = 0)
            else:
                data = np.append(data, np.expand_dims(pd_data_frame[title].values, axis=0), axis=0)

        return column_titles, data

    def _get_columns_from_data(self, data, column_names=None):
        """
            Input parameters:
            data (type: pandas.core.frame.DataFrame or pandas.core.series.Series): DataFrame from which columns should be extracted
            column_names (type: str or tuple): tuple of column names which should be extracted

            Return parameters:
            data_array(type: ndarry): data from the columns in the format [[column 1][column 2][column 3]...]
        """
        # If just one column from a dataframe or the values from a series are required

        if type(column_names) == str:
            data_array = np.expand_dims(data[column_names].values, axis = 0)
        elif type(data) == pd.core.series.Series:
            data_array = np.expand_dims(data.values, axis = 0)
            column_names = data.name
            
        # If multiple columns from a dataframe are required
        elif type(data) == pd.core.frame.DataFrame:
            if type(column_names) != tuple:
                print("For a dataframe from which more than one column should be selected a tuple with the columns to be selected needs to be provided and is not available")
            else:
                # Go through all column names and get the columns
                for i, name in enumerate(column_names):
                    if i == 0:
                        data_array = np.expand_dims(data[name].values, axis = 0)
                    else:
                        data_array = np.append(data_array, np.expand_dims(data[name].values, axis=0), axis=0)

        return data_array, column_names
    


    def _set_axis_maximum(self,maximum):
        maximum_axis = (math.ceil(maximum/(10**(len(str(int(maximum)))-1)))) * (10**(len(str(int(maximum)))-1))
        return maximum_axis



    def plot_bar(self, data, key_column_names = None, values_column_names= None, data_label = None, fig_kind = "single"):
        """
            Input parameters:
            data (type: pd.core.series.Series (keys as labels and values as bars), pd.core.fram.DataFrame): Data structure containing one columns with the keys and one or more columns with values
            key_column_names (type: str or tuple): column names which should be extracted as key
            values_column_names (type: str or tuple): column names which should be extracted as values
            data_label(type: tuple): only required if there are multiple value columns to label each "layer" of the barplot in the legend or for the titles if small multiples are applied
            fig_kind (type: string): Describes whether data should be plotted on one or on multiple graphs

            Return parameters: None

            Comment: multiple value columns are not yet implemented
        """

        if type(data) == pd.core.series.Series:
            key, values, minimum, maximum = self._series_to_arrays(data, value_counts = True)
        else:
            # Get key
            key, key_column_names = self._get_columns_from_data(data, column_names=key_column_names)
            # Get values
            values, values_column_names = self._get_columns_from_data(data, column_names=values_column_names)


        ax = self._create_figure_and_axes(fig_kind = fig_kind, n_data = values.shape[0], axes_per_row = 4)
            

        # method for setting better axis limits could be transfered into axis
        """
        if self.y_lim == None:
            self.y_lim = (0,self._set_axis_maximum(maximum))

        if self.y_tick_width == None:
            self.y_tick_width = self.y_lim[1]/10
        """

        for i,val in enumerate(values):
            if fig_kind == "single":
                n_plots = 1
                if data_label == None:
                    tmp = None
                else:
                    tmp = data_label[i]
                ax = self._create_bar(ax, key[0], values[i], data_label)    

        self.fig.subplots_adjust(bottom=0.2, left=0.2, top=0.95, right=0.95)




    def plot_hbar(self, data, key_column_names = None, values_column_names = None, data_label = None, fig_kind = "single"):
         """
            Input parameters:
            data (type: pd.core.series.Series (keys as labels and values as bars), pd.core.fram.DataFrame): Data structure containing one columns with the keys and one or more columns with values
            key_column_names (type: str or tuple): column names which should be extracted as key
            values_column_names (type: str or tuple): column names which should be extracted as values
            data_label(type: tuple): only required if there are multiple value columns to label each "layer" of the barplot in the legend or for the titles if small multiples are applied
            fig_kind (type: string): Describes whether data should be plotted on one or on multiple graphs

            Return parameters: None

            Comment: multiple value columns are not yet implemented
        """


        if type(data) == pd.core.series.Series:
            key, values, minimum, maximum = self._series_to_arrays(data, value_counts = True, ascending = True)
        else:
            # Get key
            key, key_column_names = self._get_columns_from_data(data, column_names=key_column_names)
            # Get values
            values, values_column_names = self._get_columns_from_data(data, column_names=values_column_names)

        ax = self._create_figure_and_axes(fig_kind = fig_kind, n_data = values.shape[0], axes_per_row = 4)

        """
        if self.y_lim == None:
            self.grid_direction = self.grid_direction
        if self.x_lim == None:
            self.x_lim = (0,self._set_axis_maximum(maximum))


        if self.x_tick_width == None:
            self.x_tick_width = self.x_lim[1]/10
        """
        for i,val in enumerate(values):
            if fig_kind == "single":
                n_plots = 1
                if data_label == None:
                    tmp = None
                else:
                    tmp = data_label[i]
                ax = self._create_hbar(ax, key[0], values[i], data_label)    

        
        self.fig.subplots_adjust(bottom=0.2, left=0.35, top=0.95, right=0.95)



    def plot_hist(self, data, column_names = None):
        """
            Input parameters:
            data (type: pd.core.frame.DataFrame): DataFrame from which value columns are extracted
            x_column_names (type: str or tuple): column names which should be extracted for x-values

            Return parameters: None
        """

        values,_ = self._get_columns_from_data(data, column_names)

        self.fig, ax = plt.subplots(1,1,figsize=(self.width,self.height))
        ax = self._create_histogram(ax,values[0])
        self.fig.subplots_adjust(bottom=0.2, left=0.2, top=0.95, right=0.95)


    def plot_scatter(self, data, x_column_names, y_column_names, data_label = None, fig_kind = "single", regression = False, axes_per_row = 4):
        """
            Input parameters:
            data (type: pd.core.frame.DataFrame): DataFrame from which x and y columns are extracted
            x_column_names (type: str or tuple): column names which should be extracted for x-values
            y_column_names (type: str or tuple): column names which should be extracted for y-values
            data_label(type: tuple): only required if there are multiple x and y columns to label each "layer" of the scatter in the legend or for the titles if small multiples are applied
            fig_kind (type: string): Describes whether data should be plotted on one or on multiple graphs
            regression (type: bool): Describes whether a regression line should be plotted in the graph (default: False)
            axes_per_row(type: int): Number of axes in one row (only applies if fig_kind = 'multiple'

            Return parameters: None
        """
        
        x_values, x_column_names = self._get_columns_from_data(data, column_names=x_column_names)
        y_values, y_column_names = self._get_columns_from_data(data, column_names=y_column_names)
        

        ax = self._create_figure_and_axes(fig_kind = fig_kind, n_data = x_values.shape[0], axes_per_row = axes_per_row)
        

        for i,x_val in enumerate(x_values):
            if fig_kind == "single":
                n_plots = 1
                if data_label == None:
                    tmp = None
                else:
                    tmp = data_label[i]
                ax = self._create_scatter(ax ,x = x_values[i],y = y_values[i], data_label = tmp, regression = regression)

            if fig_kind == "multiple":
                y_axis_plot = True
                if int(i%axes_per_row)!=0:
                    y_axis_plot = False
                
                ax[int(i/axes_per_row)][int(i%axes_per_row)] = self._create_scatter(ax[int(i/axes_per_row)][int(i%axes_per_row)] ,x = x_values[i],y = y_values[i] , data_label = data_label[i], title = True , y_axis_plot = y_axis_plot, regression = regression)

                
        # Fit plot into figure
        if fig_kind == "multiple":
            self.fig.tight_layout()
        elif fig_kind == "single":
            try:
                ax.legend()
            except:
                pass
            self.fig.subplots_adjust(bottom=0.2, left=0.2, top=0.95, right=0.95)



    def plot_boxplot(self, data, column_names = None, display_info=True): 
        """
            data (type: pd.core.frame.DataFrame): DataFrame from which values are extracted
            column_names (type: str or tuple): column names from which values should be extracted
            display_info (type: bool): if True, mean and median for all boxplots will be displayed above the axes
        """

        values, column_names = self._get_columns_from_data(data, column_names)

        values = np.swapaxes(values,0,1)

        column_names = list(column_names)


        self.fig, ax = plt.subplots(1,1,figsize=(self.width,self.height))
        ax = self._create_boxplot(ax, values, data_label=column_names,display_info=display_info)
        if display_info:
            self.fig.subplots_adjust(bottom=0.2, left=0.2, top=0.9, right=0.95)
        else:
            self.fig.subplots_adjust(bottom=0.2, left=0.2, top=0.95, right=0.95)



    def save_figure(self, path):
        """
        Input parameters:
        path (type: string): the path to the destination file (currently feasible filetypes: .svg, .png, .jpg)
        
        """
        name = path[0:path.rfind(".")]
        file = path[path.rfind("."):]      
        if file == ".svg":
            self.fig.savefig(path)
        elif file == ".jpg":
            fig.savefig(path,dpi=500)
        elif file == ".png":
            fig.savefig(path,dpi=500)
        else:
            print("This filetype is currently not selected")
