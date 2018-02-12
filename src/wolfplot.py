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




class Plot:    
    def __init__(self, 
                 width = 6, 
                 height = 4.5, 
                 xlable = "", 
                 x_lim=None, 
                 x_tick_width = None, 
                 x_minor_tick_width = None,              
                 ylable="", 
                 y_lim=None, 
                 y_tick_width = None, 
                 y_minor_tick_width = None,

                 
                 
                 x_number_format = "{:,}",
                 xrotation=45, 
                 xhorizontalalignment="right", 
                 y_number_format = "{:,}",
                 yrotation=0, 
                 yhorizontalalignment="right",
                 
                 grid_ticktype="major",
                 griddirection="y"):
        
        """
            Input parameters:
            width (type: float): width of overall figure (default: 6)
            height (type: float): height of overall figure (default: 4.5)
            xlable(type: string): label displayed on the x-axis (default: empty string)
            x_lim (type: tuple): limits for x-axis in the format (lower limit, upper limit)
            x_tick_width (type: float): width of the major x-tick labels (default: None)
            x_minor_tick_width (type: float): width of the minor x-tick labels, which should be a possible divider 
                                              of the major tick width (default: None)

            ylable(type: string): label displayed on the y-axis (default: empty string)
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
            griddirection (type: string): describes whether grid lines are displayed on x, y or both ticks 
                                          (possible: 'x','y','xy'; default: 'y')
    
        """
        
        
        # Initialize variables
        self.width = width
        self.height = height
        self.xlable = xlable
        self.x_number_format = x_number_format
        self.x_lim = x_lim
        self.x_tick_width = x_tick_width
        self.x_minor_tick_width = x_minor_tick_width
        self.xrotation = xrotation
        self.xhorizontalalignment = xhorizontalalignment
        self.ylable = ylable
        self.y_number_format = y_number_format
        self.y_lim = y_lim
        self.y_tick_width = y_tick_width
        self.y_minor_tick_width =y_minor_tick_width
        self.yrotation = yrotation
        self.yhorizontalalignment = yhorizontalalignment
        self.griddirection = griddirection
        self.grid_ticktype = grid_ticktype       
        
        
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


    def _set_x(self, ax, x_pos=None, x_ticklables=None):
        ax.set_xlabel(self.xlable)
        if x_pos:
            ax.set_xticks(x_pos)
        if x_ticklables:
            ax.set_xticklabels(x_ticklables)
        if self.x_lim:
            ax.set_xlim(self.x_lim[0], self.x_lim[1], auto=False)

        # Set 1000-seperator
        if self.x_number_format:
            ax.xaxis.set_major_formatter(plt.FuncFormatter((lambda x, loc: self.x_number_format.format(int(x)).replace(",","x").replace(".",",").replace("x","."))))

        if self.x_tick_width:
            if not self.x_lim:
                print("x_lim is missing and required to set x_tick_width")
            # Set major x-ticks
            ax.set_xticks(np.arange(self.x_lim[0], self.x_lim[1]+1, self.x_tick_width))

        if self.x_minor_tick_width:
            minorLocator = MultipleLocator(self.x_minor_tick_width)
            ax.xaxis.set_minor_locator(minorLocator)


        # Set the x-ticklabels to 45 degrees and align them right
        plt.setp(ax.get_xticklabels(), rotation=self.xrotation, horizontalalignment=self.xhorizontalalignment)

        return ax
    

    def _set_y(self, ax, y_pos=None,y_ticklables=None, y_axis = True):
        if y_axis:
            ax.set_ylabel(self.ylable)
        if y_pos:
            ax.set_yticks(y_pos)

        if y_ticklables and y_axis:
            ax.set_yticklabels(y_ticklables)

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
            plt.setp(ax.get_yticklabels(), rotation=self.yrotation, horizontalalignment=self.yhorizontalalignment)
        
        if not y_axis:
            ax.get_yaxis().set_ticklabels([])

        return ax

    def _create_bar(self, ax, x_data,y_data):

         # Set x-number format to None as there are no numbers on the axis
        self.x_number_format = None
        
        # Determine number of y-values and create vector representing y-Axis
        x_pos = np.arange(len(x_data)).tolist()

        ax = self._set_x(ax, x_pos=x_pos, x_ticklables = x_data.tolist())  

        ax = self._set_y(ax) 
        
        # Set grid lines
        ax.grid(axis=self.griddirection,which=self.grid_ticktype)
           
        # Remove y-axis spine
        ax.spines['bottom'].set_visible(False)
        # set lengths of y-tick parameters to 0
        ax.tick_params(axis='x', length=0)

        # Create data part of the chart
        ax.bar(x_pos, y_data,1)
        
        return ax




    def _create_hbar(self, ax, x_data,y_data):

        # Set y-number format to None as there are no numbers on the axis
        self.y_number_format = None

        # Determine number of y-values and create vector representing y-Axis
        y_pos = np.arange(len(y_data)).tolist()


        # Format x-axis    
        ax = self._set_x(ax)


        ax = self._set_y(ax, y_pos=y_pos, y_ticklables = y_data.tolist())
        
        # Set grid lines
        ax.grid(axis=self.griddirection,which=self.grid_ticktype)

        # Remove y-axis spine
        ax.spines['left'].set_visible(False)
        # set lengths of y-tick parameters to 0
        ax.tick_params(axis='y', length=0)

        # Create data part of the chart
        ax.barh(y_pos, x_data,1)

        return ax


    def _create_histogram(self,ax, x,bins,buckets):

        ax = self._set_x(ax, x_pos=buckets)

        ax = self._set_y(ax)
        
        # Set grid lines
        ax.grid(axis=self.griddirection,which=self.grid_ticktype) 
        
        # Create data part of the chart
        ax.hist(x, bins)
        
        return ax
    
    def _create_scatter(self, ax, x, y, data_label = None, title = False, y_axis = True, regression = False):
        
        if title:
            ax = self._set_title(ax, data_label)
            
        
       
        ax = self._set_x(ax)
        
        
        ax = self._set_y(ax, y_axis = y_axis)
        
        
        # Set grid lines
        ax.grid(axis=self.griddirection,which=self.grid_ticktype)        
        
        # Create graph
        ax.scatter(x ,y, label=data_label, s = 3, linewidth = 0.0)
        
        if regression:
            x_reg, y_reg = self._linear_regression(x,y)
            ax.plot(x_reg,y_reg)
            
        
        return ax
    
    def plot_data(self,x = np.empty([1]),y = None, data_label = None, bins = None, fig_type = None,fig_kind="single", 
                  regression = False):
        """
            Input parameters:
            width (type: float): width of overall figure (default: 6)
            x (type: numpy-array): x-values
            y (type: numpy-array): y-values
            data_label (type: numpy-arra): contains the labels for the data in plots with multiple data which will either be 
                                           displayed in the legend(fig_kind="single") or the title (fig_kind="multiple")
            fig_type (type: string): Describes the figure to be plotted (possible: 'bar', 'hbar', 'hist', 'scatter')
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
         
        
        
        if fig_type == "bar":
            ax = self._create_bar(ax,x[0],y)    

        if fig_type == "hbar":
            ax = self._create_hbar(ax,x[0],y)

        if fig_type == "hist":
            self.x_lim = (bins[0],bins[1])
            self.x_minor_tick_width = bins[2]

            bins_ = np.arange(bins[0],bins[1]+1,bins[2]).tolist()

            # Determine number of y-values and create vector representing y-Axis
            buckets = np.arange(bins[0],bins[1],bins[2]).tolist()     

            ax = self._create_histogram(ax,x,bins_,buckets)
        
        
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

        return self.fig

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