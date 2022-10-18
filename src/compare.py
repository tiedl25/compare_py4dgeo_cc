import numpy as np
import csv
import statistics
import matplotlib.pyplot as plt  
import matplotlib.ticker as ticker

import file_handle as fhandle
import vec_calc as vc

from map_diff import Map_Diff
from pathlib import Path


class Compare:
    '''
    A class for comparing the outputs of the m3c2-algorithm between CloudCompare and Py4dGeo

    Attributes: 
        re (dict): Stores all the information of the reference point cloud (Cloud compare) -> coordinates, distances, lodetection, spread1, spread2 and normal coordinates
        cl (dict): Stores all the information of the point cloud (Py4dGeo) -> coordinates, distances, lodetection, spread1, spread2 and normal coordinates
        size (int): The number of points in each cloud
        diffs (dict): The differences between between each related argument of both clouds -> gets calculated in calc_differences()
        aspects (numpy.array): The aspects between related normal vectors in both clouds.
        slopes (numpy.array): The slopes between related normal vectors in both clouds.
    '''
    def __init__(self, re_pts, re_normals, re_dist, re_lod, cl_pts, cl_normals, cl_dist, cl_lod):
        '''
        The class constructor, that initializes the attributes of the Compare class

        Parameters:
            self (Compare): the object itself
            re_pts (numpy.ndarray): the xyz-coordinates of the reference point cloud
            re_dist (numpy.ndarray): the m3c2-distances of the reference point cloud
            re_lod (numpy.ndarray): the level-of-detection of the reference point cloud
            re_spread (list): the standard deviation of distances of the reference point cloud
            re_samples (list): the total number of points taken into consideration in either cloud 
            re_normals (numpy.ndarray): the normal-coordinates of the reference point cloud
            cl_pts (numpy.ndarray): the xyz-coordinates of the py4dgeo point cloud
            cl_dist (numpy.ndarray): the m3c2-distances of the py4dgeo point cloud
            cl_lod (numpy.ndarray): the level-of-detection of the py4dgeo point cloud
            cl_spread (list): the standard deviation of distances of the py4dgeo point cloudy
            cl_samples (list): the total number of points taken into consideration in either cloud 
            cl_normals (numpy.ndarray): the normal-coordinates of the py4dgeo point cloudy)
        '''
        self.re = {'pts': re_pts, 'dist' : re_dist, 'lod' : re_lod, 'normals' : re_normals}
        self.cl = {'pts': cl_pts, 'dist' : cl_dist, 'lod' : cl_lod, 'normals' : cl_normals}

        self.diffs = {}     
        self.nan_mode = []

        self.size = int(np.size(re_dist))
        self.calc = False

        self.slopes = np.empty(self.size, dtype=object)
        self.aspects = np.empty(self.size, dtype=object)

    def calc_differences(self):
        '''
        Calculate all the differences between related arguments of this class

        Parameters:
            self (Compare): the object itself
        
        Returns:
            dict: The differences between each related argument of both point clouds
        '''
        self.calc = True

        x1,y1,z1 = np.transpose(self.cl['pts'])
        x2,y2,z2 = np.transpose(self.re['pts'])

        nx1,ny1,nz1 = np.transpose(self.cl['normals'])
        nx2,ny2,nz2 = np.transpose(self.re['normals'])

        self.diffs['X'] = abs(x1-x2)
        self.diffs['Y'] = abs(y1-y2)
        self.diffs['Z'] = abs(z1-z2)
        self.diffs['NX'] = abs(nx1-nx2)
        self.diffs['NY'] = abs(ny1-ny2)
        self.diffs['NZ'] = abs(nz1-nz2)
        self.diffs['Distance'] = abs(self.cl['dist']-self.re['dist'])
        self.diffs['LODetection'] = abs(self.cl['lod']-self.re['lod'])

        # (cloud is nan, reference is nan)
        switch={'(False, False)': 0,
                '(False, True)': 1,
                '(True, False)': 2,
                '(True, True)': 3}

        for pair in zip(np.isnan(self.cl['dist']), np.isnan(self.re['dist'])):
            self.nan_mode.append(switch.get(str(pair)))

        for i in range(0, self.size):      
            normalized_vector = vc.transform((nx2[i], ny2[i], nz2[i]), (nx1[i], ny1[i], nz1[i]))
            self.slopes[i] = vc.getSlope(normalized_vector)
            self.aspects[i] = vc.getAspect(normalized_vector)

        return self.diffs

    def mapDiff(self, path1, path2, proj='2d', advanced=False, ps=10):
        '''
        Handle the plots of distance and lodetection. Decide if the plots are single-plotted or the distance/lodetection plots from CC and Py4dGeo are also shown.

        Parameters:
            self (Compare): The object itself.
            path1 (str): The output path for the distance plot.
            path2 (str): The output path for the lodetection plot.
            proj (str): Specifies whether the plot is plotted in 2d or 3d.
            advanced (bool): Specifies whether the differences gets plotted in comparison to the CC and Py4dGeo Distances/LODetection or just alone.
            ps (int): Sets the point size.
        '''
        if not self.calc: 
            print('There is nothing to plot. You have to call calc_differences first.')
            return

        map1 = Map_Diff(self.diffs['Distance'], self.re['pts'], "Difference in distance between CC and Py4dGeo", point_size=ps, cmap='jet')
        map2 = Map_Diff(self.diffs['LODetection'], self.re['pts'], "Difference in lodetection between CC and Py4dGeo", point_size=ps, cmap='jet')

        show = True if proj=='3d' else False

        print('Plot differences on a {} map'.format(proj)) 
        if advanced:
            map1.compare([self.re['dist'], self.cl['dist']], [self.re['pts'], self.cl['pts']], ['CC Distances', 'Py4dGeo Distances'], output=path1, show=show, proj=proj)
            map2.compare([self.re['lod'], self.cl['lod']], [self.re['pts'], self.cl['pts']], ['CC LODetection', 'Py4dGeo LODetection'], output=path2, show=show, proj=proj)
        else:
            map1.mapDiff(path1, show, proj)    
            map2.mapDiff(path2, show, proj)

    def plotNormDiff(self, path):
        '''
        Draw the normal differences between both point clouds on a polar plot by using matplotlib.

        Parameters:
            self (Compare): The object itself.
            path (str): The output path for the normal plot.
        '''
        if not self.calc: 
            print('There is nothing to plot. You have to call calc_differences first.')
            return

        print('Plot normal differences of the normal vectors')

        #setup the plot
        ax = plt.axes(projection='polar')
        plt.title('Polar plot of normal differences')
        plt.xlabel('Aspect')
        plt.ylabel('Slope', rotation=0)
        
        ax.yaxis.set_label_coords(1, 0.75)
        ax.yaxis.set_offset_position('right')
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f' + '°'))

        plt.scatter(self.aspects, self.slopes, s=2, c='b')
        plt.savefig(path)
        plt.close()

    def plotNormHist(self, path):
        '''
        Plot the normal differences between both point clouds in a histogram.

        Parameters:
            self (Compare): The object itself.
            path (str): The output path for the normal plot.
        '''
        if not self.calc: 
            print('There is nothing to plot. You have to call calc_differences first.')
            return

        fig, ax = plt.subplots(1,2)
        plt.subplot(1,2,1)
        plt.title('Aspect')
        ax[0].hist(self.aspects)
        ax[0].xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f' + '°'))

        plt.subplot(1,2,2)
        plt.title('Slope')
        ax[1].hist(self.slopes)

        ax[1].xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f' + '°'))

        plt.savefig(path)
        plt.close()

    def spreadDiff(self, re_spread, cl_spread, path, plot=False):
        '''
        Plot standard deviation differences between both point clouds.

        Parameters:
            self (Compare): The object itself.
            path (str): The output path for the spread plot.
        '''
        self.re.update({'spread' : re_spread})
        self.cl.update({'spread' : cl_spread})

        self.diffs['Spread1'] = abs(self.cl['spread'][0]-self.re['spread'][0])
        self.diffs['Spread2'] = abs(self.cl['spread'][1]-self.re['spread'][1])

        if plot:
            print('Plot spread differences')

            titles = ('Spread1', 'Spread2')

            plt.xlabel('Difference')
            plt.ylabel('Number of points')

            plt.subplot(1,2,1)
            li = [x for x in self.diffs['Spread1'] if np.isnan(x) == False]
            plt.hist(li, bins=50)
            plt.title(titles[0])
            # set units for the x and y axes
            plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f m'))

            plt.subplot(1,2,2)
            li = [x for x in self.diffs['Spread2'] if np.isnan(x) == False]
            plt.hist(li, bins=50)
            plt.title(titles[1])
            # set units for the x and y axes
            plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f m'))

            plt.savefig(path)
            plt.close()

    def sampleDiff(self, re_samples, cl_samples, path, plot=False):
        '''
        Plot point density differences between both point clouds.

        Parameters:
            self (Compare): The object itself.
            path (str): The output path for the spread plot.
        '''
        self.re.update({'num_samples' : re_samples})
        self.cl.update({'num_samples' : cl_samples})

        self.diffs['NumSamples1'] = abs(self.cl['num_samples'][0]-self.re['num_samples'][0])
        self.diffs['NumSamples2'] = abs(self.cl['num_samples'][1]-self.re['num_samples'][1])

        if plot:
            print('Plot num_sample differences')

            titles = ('num_samples1', 'num_samples2')

            plt.xlabel('Difference')
            plt.ylabel('Number of points')

            plt.subplot(1,2,1)
            li = [x for x in self.diffs['NumSamples1'] if np.isnan(x) == False]
            plt.hist(li, bins=50)
            plt.title(titles[0])

            plt.subplot(1,2,2)
            li = [x for x in self.diffs['NumSamples2'] if np.isnan(x) == False]
            plt.hist(li, bins=50)
            plt.title(titles[1])

            plt.savefig(path)
            plt.close()      

    def writeStatistics(self, path):
        '''
        Write statistics (mean, median, standard-deviation) of the differences between both point clouds to a csv file:

        Parameters: 
            self (Compare): The object itself.
            path (str): The output path for the csv file.
        '''
        if not self.calc: 
            print('There is nothing to write. You have to call calc_differences first.')
            return

        s = len(self.diffs)+2
        mean = np.empty(s, dtype=np.float64)
        median = np.empty(s, dtype=np.float64)
        std_dev = np.empty(s, dtype=np.float64)

        keys = ['Statistics']
        keys.extend(list(self.diffs.keys()))
        keys.extend(['Aspect', 'Slope'])
        
        for i in range(1,s):
            if i<s-2: li = [x for x in self.diffs[keys[i]] if np.isnan(x) == False]
            elif i==s-2: li = [x for x in self.aspects if np.isnan(x) == False]
            elif i==s-1: li = [x for x in self.slopes if np.isnan(x) == False]

            if np.size(li) > 1:
                mean[i] = statistics.mean(li)
                median[i] = statistics.median(li)
                std_dev[i] = statistics.stdev(li)
            else:
                mean[i] = None
                median[i] = None
                std_dev[i] = None

        bo_nan = len([i for i in self.nan_mode if i==3])
        cl_nan = len([i for i in self.nan_mode if i==2])
        re_nan = len([i for i in self.nan_mode if i==1])

        with open(path, mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
            
            writer.writerow(keys)

            mean = np.append(['Mean'], mean)
            median = np.append(['Median'], median)
            std_dev = np.append(['Standard Deviation'], std_dev)

            writer.writerow(mean)
            writer.writerow(median)
            writer.writerow(std_dev)

            writer.writerow(['Nan-Values in CloudCompare', f'{(bo_nan + re_nan)/self.size*100:.2f} %'])
            writer.writerow(['Nan-Values in Py4dGeo', f'{(bo_nan + cl_nan)/self.size*100:.2f} %'])
            writer.writerow(['Nan-Values in Both', f'{bo_nan/self.size*100:.2f} %'])

    def writeDiff(self, path):
        '''
        Write all the differences between both point clouds to a csv file.

        Parameters:
            self (Compare): The object itself.
            path (str): The output path for the csv file.
        '''
        if not self.calc: 
            print('There is nothing to write. You have to call calc_differences first.')
            return

        with open(path, mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
            header = ['X Coord', 'Y Coord', 'Z Coord']
            header.extend(list(self.diffs.keys()))
            header.extend(['Aspect', 'Slope', 'Nan-Mode'])
            writer.writerow(header)

            x,y,z = np.transpose(self.re['pts'])

            spread_set=True if 'Spread1' in self.diffs else False
            sample_set=True if 'NumSamples1' in self.diffs else False

            for i in range(0, self.size):
                row = [x[i], y[i], z[i],
                        self.diffs['X'][i],
                        self.diffs['Y'][i],
                        self.diffs['Z'][i],
                        self.diffs['NX'][i],
                        self.diffs['NY'][i],
                        self.diffs['NZ'][i],
                        self.diffs['Distance'][i],
                        self.diffs['LODetection'][i]]
                if spread_set: row.extend([self.diffs['Spread1'][i], self.diffs['Spread2'][i]])
                if sample_set: row.extend([self.diffs['NumSamples1'][i], self.diffs['NumSamples2'][i]])
                row.extend([self.aspects[i], self.slopes[i], self.nan_mode[i]])
                
                writer.writerow(row)

    def writeCloud(self, path, cc_mode=True):
        '''
        Handle writing to different filetypes(ascii and las/laz), so theres no need to change the function when using a different file extension.

        Parameters:
            self (Py4d_M3C2): The object itself.
            cc_mode (bool): Specifies if the header is written with CC vocabulary or py4dgeo vocabulary.
        '''
        extension = Path(path).suffix

        if cc_mode: keys = ['M3C2__distance', 'distance__uncertainty', 'STD_cloud1', 'STD_cloud2', 'Npoints_cloud1', 'Npoints_cloud2', 'NormalX', 'NormalY', 'NormalZ']
        else: keys = ['distance', 'lodetection', 'spread1', 'spread2', 'num_samples1', 'num_samples2', 'nx', 'ny', 'nz']

        attributes={keys[0] : self.diffs['Distance'], 
                    keys[1] : self.diffs["LODetection"]}

        if 'Spread1' in self.diffs:
            attributes.update({keys[2] : self.diffs["Spread1"], keys[3] : self.diffs["Spread2"],})
        if 'NumSamples1' in self.diffs:
            attributes.update({keys[4] : self.diffs["NumSamples1"], keys[5] : self.diffs["NumSamples2"]})

        attributes.update({keys[6] : self.diffs['NX'], keys[7] : self.diffs['NY'], keys[8] : self.diffs['NZ']})

        if extension in [".las", ".laz"]:
            fhandle.write_las(self.re['pts'], path, attributes)
        elif extension in [".xyz", ".txt", ".asc"]:
            fhandle.write_xyz(self.re['pts'], path, attributes)
        else:
            print("File extension has to be las, laz, xyz or txt")
            return