import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class Map_Diff:
    '''
    This class can be used to plot certain values as colors onto given coordinates either in a 3d or a 2d plot, by using matplotlib.

    Arguments: 
        mapped_vals (numpy.ndarray): The values to be plotted as colors.
        coords (numpy.ndarray): The points to be plotted. 
        title (str): The title of the plot.
        size (int): The number of coordinates.
        unit (str): The unit used for the axes
    '''
    def __init__(self, mapped_vals, coords, title='', unit='m'):
        '''
        The constructor of the Map_Diff class.

        Parameters:
            self (Map_Diff): The object itself:
            mapped_vals (numpy.ndarray): The values to be plotted as colors
            coords (numpy.ndarray): The points to be plotted. 
            title (str): The title of the plot.
            unit (str): The unit used for the axes, default is meters.
        '''
        self.mapped_vals = mapped_vals
        self.title = title
        self.coords = coords
        self.size = int(np.size(coords/3))
        self.unit = unit
        self.point_size = 5

    def subplot(self, vals, crds, ttl, ax, min, max, proj='2d'):
        '''
        Draw given coordinates in a Subplot.

        Parameters:
            self (Compare): The object itself.
            vals (numpy.ndarray): The values to be plotted as colors
            crds (numpy.ndarray): The points to be plotted. 
            ttl (str): The title of the subplot.
            ax (matplotlib.axes._subplots.AxesSubplot): The Axes.
            max (float): The maximum of the mapped values.
            min (float): The minimum of the mapped values. 
            proj (str): Specifies if the plot is 2d or 3d.
        '''
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')

        x,y,z = crds[0:,0], crds[0:,1], crds[0:,2] #returns X,Y,Z coordinates

        pointColors = abs(vals)/max

        ax.title.set_text(ttl)    

        if proj=='2d':
            # set x and y scale equal
            ax.set_aspect(aspect=1)
            
            #draw points
            pts = ax.scatter(x,y,s=self.point_size, c=pointColors, cmap='YlGnBu')
        elif proj=='3d':
            #set viewport 
            ax.view_init(22, 112) 

            # set x, y and z scale equal
            ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))

            #draw points
            pts = ax.scatter3D(x,y,z,s=self.point_size, c=pointColors, cmap='YlGnBu')
            ax.zaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f ' + self.unit))

        # set units for the x and y axes
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f ' + self.unit))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f ' + self.unit))
        #ax.yaxis.set_ticklabels([]) #remove ticks for the y-axis

        return pts

    def plot(self, ax, min, max, proj='2d'):
        '''
        Draw the coordinates in the plot

        Parameters:
            self (Compare): The object itself.
            ax (matplotlib.axes._subplots.AxesSubplot): The Axes.
            max (float): The maximum of the mapped values.
            min (float): The minimum of the mapped values. 
            proj (str): Specifies if the plot is 2d or 3d.
        '''
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        x,y,z = self.coords[0:,0], self.coords[0:,1], self.coords[0:,2] #returns X,Y,Z coordinates

        pointColors = abs(self.mapped_vals)/max

        ax.title.set_text(self.title)

        if proj=='2d':        
            # set x and y scale equal
            ax.set_aspect(aspect=1)
            
            #draw points
            pts = ax.scatter(x,y,s=self.point_size, c=pointColors, cmap='YlGnBu')
        elif proj=='3d':
            #set viewport 
            ax.view_init(22, 112) 

            # set x, y and z scale equal
            ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))

            #draw points
            pts = ax.scatter3D(x,y,z,s=self.point_size, c=pointColors, cmap='YlGnBu')
            ax.zaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f ' + self.unit))

        # set units for the x and y axes
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f ' + self.unit))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f ' + self.unit))

        return pts

    def mapDiff(self, output=False, show=False, proj='2d'):
        '''
        Create a 2d/3d plot that shows a point cloud and maps given values as colors onto them.

        Parameters:
            self (Compare): The object itself.
            output (str): The path for the output file. If not set the plot wont be saved.
            show (bool): Specifies if the plot is shown or not.
            proj (str): Specifies if the plot is 2d or 3d.
        '''
        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection = None if proj=='2d' else proj)

        #get max distances
        max = np.nanmax(self.mapped_vals)
        min = np.nanmin(self.mapped_vals)

        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        if proj=='2d':
            plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right') #rotate x-ticklabels to prevent overlapping
            pts = self.plot(ax, min, max, '2d')
        elif proj=='3d':
            ax.set_zlabel('Z Axis')
            pts = self.plot(ax, min, max, '3d')
        else: 
            print('Not a valid projection')
            return
        #set color bar with individual ticks and labels
        cbar = plt.colorbar(pts)
        ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([str(s) + ' ' + self.unit for s in np.round([max*i for i in ticks], 3)])

        if show: plt.show()
        if output: fig.savefig(output)
        plt.close()

    def compare(self, vals, crds, ttls, output=False, proj='2d', show=False):
        '''
        Show a plot, that makes comparison easier by giving the oportunity to also plot different points and data next to the the initial plot. Subplots are used.

        Parameters:
            self (Compare): The object itself.
            vals (list): A list of arrays, whereby each array contains one set of values that gets plotted as colors. More than one set of values is possibly, but then also more coords must be provided.
            crds (list): A list of coordinate arrays. Each coordinate array will be plotted in a different subplot.
            ttls (list): A list of titles for each subplot.
            output (str): The path for the output file. If not set the plot wont be saved.
            show (bool): Specifies if the plot is shown or not.
            proj (str): Specifies if the plot is 2d or 3d.
        '''
        dim = len(vals)
        fig, ax = plt.subplots(1, dim+1, subplot_kw={'projection' : None if proj=='2d' else proj}, figsize=(15,7), constrained_layout=True) # create as many subplots as big the dimension is

        if proj=='2d': 
            for a in ax: plt.setp(a.get_xticklabels(), rotation=30, horizontalalignment='right') #rotate x-ticklabels to prevent overlapping
            
        #get max value of all mapped values to determine maximum of colorbar
        highest = 0
        max = np.nanmax(self.mapped_vals)
        min = np.nanmin(self.mapped_vals)
        for i in range(0, dim):
            new_max = np.nanmax(vals[i])
            new_min = np.nanmin(vals[i])
            if new_max > max: 
                max = new_max
                highest = i+1
            if new_min < min:
                min = new_min
        if min < 0: min = 0
        # create subplots
        pts = np.empty(dim+1, dtype=object)
        pts[0] = self.plot(ax[0], min, max, proj)
        for pt in range(0,dim):
            pts[pt+1] = self.subplot(vals[pt], crds[pt], ttls[pt], ax[pt+1], min, max, proj)

        # set minimum of colorbar
        pts[highest].set_clim(min)
        
        #set color bar with individual ticks and labels
        cbar = plt.colorbar(pts[highest], ax=ax, orientation='horizontal')
        ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([str(s) + ' ' + self.unit for s in np.round([max*i for i in ticks], 3)])

        if show: plt.show()
        if output: fig.savefig(output)
        plt.close()