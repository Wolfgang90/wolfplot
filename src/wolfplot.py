"""
TODOs:
- Boxplots (however I have included a simple example for self building in our style in the example file)
- Graphs with multiple data for: bar, hbar, hist (currently implemented for one data layer (compare example file))
- Write documentation for subfunctions
"""


import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
# Import MultipleLocator from matplotlib ticker for creating tick marks
from matplotlib.ticker import MultipleLocator

import math




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

                 
                 
                 x_number_format = "{:,}",
                 x_rotation=45, 
                 x_horizontal_alignment="right", 
                 y_number_format = "{:,}",
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
        
            x_number_format (type: format): format for the x-axis tick labels (default: {:,})
            x_rotation (type: int): rotation for the x-axis tick labels (default: 45)
            xhorizontalalignement(type: string): alignement of the x-axis tick labels 
                                                 (possible: 'right', 'center', 'left'; default: right)
            y_number_format (type: format): format for the y-axis tick labels (default: {:,})
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
        self.x_number_format = x_number_format
        self.x_lim = x_lim
        self.x_tick_width = x_tick_width
        self.x_minor_tick_width = x_minor_tick_width
        self.x_rotation = x_rotation
        self.x_horizontal_alignment = x_horizontal_alignment
        self.y_label = y_label
        self.y_number_format = y_number_format
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
        
        x = x[:, np.newaxis]
        y = y[:, np.newaxis]
        
        # Create linear regression object
        regr = linear_model.LinearRegression()

        # Train the model using the full set
        regr.fit(x, y)
        
        # Make predictions using the full set
        y_pred = regr.predict(x)
        
        return x,y_pred
        


    def _set_title(self, ax,title):

        # Plot chart title
        ax.set_title(title,y=1.00, loc='left')
        return ax


    def _set_x(self, ax, axis_pos=None, axis_ticklabels=None, axis = "x"):

        # Set axis label
        exec("ax.set_" + axis + "label(self." + axis + "_label)")

        # Set position of axis ticks (required for axis with ordinal data)
        if axis_pos:
            exec("ax.set_" + axis + "ticks(axis_pos)")

        # Set position of tick labels
        if axis_ticklabels:
            exec("ax.set_" + axis + "ticklabels(axis_ticklabels)")

        # Set axis lower and upper limit
        if eval("self." + axis + "_lim"):
            exec("ax.set_" + axis + "lim(self." + axis + "_lim[0], self." + axis + "_lim[1], auto=False)")

        
        # Set 1000-seperator
        if eval("self." + axis + "_number_format"):
            if axis == "x":
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: self.x_number_format.format(int(x)).replace(",","x").replace(".",",").replace("x",".")))
            if axis == "y":
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: self.y_number_format.format(int(x)).replace(",","x").replace(".",",").replace("x",".")))


        # Set axis major tick width
        if eval("self." + axis + "_tick_width"):
            if not eval("self." + axis + "_lim"):
                print(axis + "_lim is missing and required to set " + axis + "_tick_width")
            # Set major ticks
            exec("ax.set_" + axis + "ticks(np.arange(self." + axis + "_lim[0], self." + axis + "_lim[1]+1, self." + axis + "_tick_width))")

        # Set axis minor tick width
        if eval("self." + axis +  "_minor_tick_width"):
            exec("minorLocator = MultipleLocator(self." + axis + "_minor_tick_width)")
            exec("ax." + axis + "axis.set_minor_locator(minorLocator)")


        # Set orientation and alignment of axis tick labels
        exec("plt.setp(ax.get_" + axis + "ticklabels(), rotation=self." + axis + "_rotation, horizontalalignment=self." + axis + "_horizontal_alignment)")

        return ax
    

    def _set_y(self, ax, y_pos=None,y_ticklabels=None, y_axis = True):
        if y_axis:
            ax.set_ylabel(self.y_label)
        if y_pos:
            ax.set_yticks(y_pos)

        if y_ticklabels and y_axis:
            ax.set_yticklabels(y_ticklabels)

        if self.y_lim:
            ax.set_ylim(self.y_lim[0], self.y_lim[1], auto=False)

        # Set 1000-seperator
        if self.y_number_format and y_axis:
            ax.yaxis.set_major_formatter(plt.FuncFormatter((lambda x, loc: self.y_number_format.format(int(x)).replace(",","x").replace(".",",").replace("x","."))))
            

        if self.y_tick_width:
            if not self.y_lim:
                print("y_lim is missing and required to set y_tick_width")
            # Set major x-ticks
            ax.set_yticks(np.arange(self.y_lim[0], self.y_lim[1]+1, self.y_tick_width))

        if self.y_minor_tick_width:
            minorLocator = MultipleLocator(self.y_minor_tick_width)
            ax.yaxis.set_minor_locator(minorLocator)

        # Set the x-ticklabels to 45 degrees and align them right
        if y_axis:
            plt.setp(ax.get_yticklabels(), rotation=self.y_rotation, horizontalalignment=self.y_horizontal_alignment)
        
        if not y_axis:
            ax.get_yaxis().set_ticklabels([])

        return ax

    def _create_bar(self, ax, keys, values):

         # Set x-number format to None as there are no numbers on the axis
        self.x_number_format = None
        
        # Determine number of y-values and create vector representing y-Axis
        x_pos = np.arange(len(keys)).tolist()

        ax = self._set_x(ax, axis_pos=x_pos, axis_ticklabels = keys.tolist())  

        ax = self._set_y(ax) 
        
        # Set grid lines
        ax.grid(axis=self.grid_direction,which=self.grid_ticktype)
           
        # Remove y-axis spine
        #ax.spines['bottom'].set_visible(False)
        # set lengths of y-tick parameters to 0
        ax.tick_params(axis='x', length=0)

        # Create data part of the chart
        ax.bar(x_pos, values,1)
        
        return ax




    def _create_hbar(self, ax, keys, values):

        # Set y-number format to None as there are no numbers on the axis
        self.y_number_format = None


        # Determine number of y-values and create vector representing y-Axis
        y_pos = np.arange(len(keys)).tolist()


        # Format x-axis    
        ax = self._set_x(ax)


        ax = self._set_y(ax, y_pos=y_pos, y_ticklabels = keys.tolist())
        
        # Set grid lines
        ax.grid(axis=self.grid_direction,which=self.grid_ticktype)

        # Remove y-axis spine
        #ax.spines['left'].set_visible(False)
        # set lengths of y-tick parameters to 0
        ax.tick_params(axis='y', length=0)

        # Create data part of the chart
        ax.barh(y_pos, values,1)

        return ax


    def _create_histogram(self,ax,x):

        ax = self._set_x(ax, axis_pos=self.buckets)

        ax = self._set_y(ax)
        
        # Set grid lines
        ax.grid(axis=self.grid_direction,which=self.grid_ticktype) 
        
        # Create data part of the chart
        ax.hist(x, self.bins)
        
        return ax
    
    def _create_scatter(self, ax, x, y, data_label = None, title = False, y_axis = True, regression = False):
        
        if title:
            ax = self._set_title(ax, data_label)
            
        
       
        ax = self._set_x(ax)
        
        
        ax = self._set_y(ax, y_axis = y_axis)
        
        
        # Set grid lines
        ax.grid(axis=self.grid_direction,which=self.grid_ticktype)        
        
        # Create graph
        ax.scatter(x ,y, label=data_label, s = 3, linewidth = 0.0)
        
        if regression:
            x_reg, y_reg = self._linear_regression(x,y)
            ax.plot(x_reg,y_reg)
            
        
        return ax

    def _series_to_arrays(self, pd_series, ascending = False):
        pd_series = pd_series.value_counts(ascending=ascending)
        keys = pd_series.keys().values
        values = pd_series.values
        minimum = pd_series.min()
        maximum = pd_series.max()
        return keys, values, minimum, maximum

    def _set_axis_maximum(self,maximum):
        maximum_axis = (math.ceil(maximum/(10**(len(str(int(maximum)))-1)))) * (10**(len(str(int(maximum)))-1))
        return maximum_axis


    def plot_bar(self, data):
        """
            Input parameters:
            data (type: pandas.core.series.Series): Keys as labels and values as bars
        """
        keys, values, minimum, maximum = self._series_to_arrays(data)
        if self.y_lim == None:
            self.y_lim = (0,self._set_axis_maximum(maximum))

        if self.y_tick_width == None:
            self.y_tick_width = self.y_lim[1]/10
        self.fig, ax = plt.subplots(1,1,figsize=(self.width,self.height))
        ax = self._create_bar(ax, keys, values)    
        self.fig.subplots_adjust(bottom=0.2, left=0.2, top=0.95, right=0.95)

    def plot_hbar(self, data):
        """
            Input parameters:
            data (type: pandas.core.series.Series): Keys as labels and values as bars
            grid_direction(type:string): x, y, or xy
        """
        keys, values, minimum, maximum = self._series_to_arrays(data, ascending = True)
        self.grid_direction = self.grid_direction
        if self.x_lim == None:
            self.x_lim = (0,self._set_axis_maximum(maximum))

        if self.x_tick_width == None:
            self.x_tick_width = self.x_lim[1]/10
        self.fig, ax = plt.subplots(1,1,figsize=(self.width,self.height))
        ax = self._create_hbar(ax, keys, values)    
        self.fig.subplots_adjust(bottom=0.2, left=0.35, top=0.95, right=0.95)



    def plot_hist(self, data):
        _, values, minimum, maximum = self._series_to_arrays(data)
        self.fig, ax = plt.subplots(1,1,figsize=(self.width,self.height))
        ax = self._create_histogram(ax,values)
        self.fig.subplots_adjust(bottom=0.2, left=0.2, top=0.95, right=0.95)


    
    def plot_data(self,x = np.empty([1]),y = None, data_label = None, fig_type = None,fig_kind="single", 
                  regression = False):
        """
            Input parameters:
            width (type: float): width of overall figure (default: 6)
            x (type: numpy-array): x-values
            y (type: numpy-array): y-values
            data_label (type: numpy-arra): contains the labels for the data in plots with multiple data which will either be 
                                           displayed in the legend(fig_kind="single") or the title (fig_kind="multiple")
            fig_type (type: string): Describes the figure to be plotted (possible: 'hist', 'scatter')
            fig_kind (type: string): Describes whether data should be plotted on one or on multiple graphs
                                     (possible: 'single', 'multiple'; default: 'single')
            regression (type: bool): Describes whether a regression line should be plotted in the graph (default: False)
            
            Return variable:
            fig (type: matplotlib figure): The figure containing the plot
        """
        
        
        
        try:
            if not x.shape[1]:
                pass
        except:
            x = np.expand_dims(x, axis=0)            
            
        if fig_kind=="single":
            self.fig, ax = plt.subplots(1,1,figsize=(self.width,self.height))
            title = False
            n_plots = 1
            
        if fig_kind == "multiple":            
            #ax = self.fig.add_subplot(int(i/4)+1,int(i%4)+1,i+1)
            self.fig, ax = plt.subplots(int(x.shape[0]/4)+1,4, figsize=(self.width,self.height))
            title = True
            #ax = self.fig.add_subplot(int(x.shape[0]/4)+1,int(x.shape[0]%4)+1)
            n_plots = (int(x.shape[0]/4)+1) * 4         
         
        
        
        if fig_type == "hist":

            ax = self._create_histogram(ax,x)
        
        
        """
            Start: Implementation Scatter Plot
        """
        
        for i in range(x.shape[0]):
            
            y_axis = True
            if int(i%4)!=0:
                y_axis = False


            if fig_type == "scatter":
                if fig_kind == "multiple":
                    ax[int(i/4)][int(i%4)] = self._create_scatter(ax[int(i/4)][int(i%4)] ,x = x[i],y = y[i] ,
                                                                  data_label = data_label[i], title = title ,
                                                                  y_axis = y_axis, regression = regression)
                if fig_kind == "single":
                    ax = self._create_scatter(ax ,x = x[i],y = y[i], data_label = data_label[i],title = title ,
                                              y_axis = y_axis, regression = regression)

         
        # Set subplot axis without content to invisible
        gap = n_plots - x.shape[0]
        if gap:
            for i in range(gap):
                ax[int(x.shape[0]/4)][4-i-1].axis('off')
                
        try:        
            if data_label[0] and fig_kind == "single":    
                ax.legend(fontsize="xx-small")
        except:
            pass
        

        
		
		
        # Fit plot into figure
        if fig_kind == "multiple":
            self.fig.tight_layout()
        else:
            self.fig.subplots_adjust(bottom=0.2, left=0.35, top=0.95, right=0.95)


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
