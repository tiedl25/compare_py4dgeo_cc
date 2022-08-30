import sys
import numpy as np
import os
import platform
import argparse
import csv
import statistics
import matplotlib.pyplot as plt  
import matplotlib.ticker as ticker
import helper as hlp

from py4d_m3c2 import Py4d_M3C2
from vec_calc import Vec_Calc
from map_diff import Map_Diff

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
    def __init__(self, re_pts, re_dist, re_lod, re_spread1, re_spread2, re_normals, cl_pts, cl_dist, cl_lod, cl_spread1, cl_spread2, cl_normals):
        '''
        The class constructor, that initializes the attributes of the Compare class

        Parameters:
            self (Compare): the object itself
            re_pts (numpy.ndarray): the xyz-coordinates of the reference point cloud
            re_dist (numpy.ndarray): the m3c2-distances of the reference point cloud
            re_lod (numpy.ndarray): the level-of-detection of the reference point cloud
            re_spread1 (numpy.ndarray): the standard deviation of distances of the reference point cloud
            re_spread2 (numpy.ndarray): the standard deviation of distances of the reference point cloud
            re_normals (numpy.ndarray): the normal-coordinates of the reference point cloud
            cl_pts (numpy.ndarray): the xyz-coordinates of the py4dgeo point cloud
            cl_dist (numpy.ndarray): the m3c2-distances of the py4dgeo point cloud
            cl_lod (numpy.ndarray): the level-of-detection of the py4dgeo point cloud
            cl_spread1 (numpy.ndarray): the standard deviation of distances of the py4dgeo point cloudy
            cl_spread2 (numpy.ndarray): the standard deviation of distances of the py4dgeopoint cloudy
            cl_normals (numpy.ndarray): the normal-coordinates of the py4dgeo point cloudy)
        '''
        self.re = {'pts': re_pts, 'dist' : re_dist, 'lod' : re_lod, 'spread1' : re_spread1, 'spread2' : re_spread2, 'normals' : re_normals}
        self.cl = {'pts': cl_pts, 'dist' : cl_dist, 'lod' : cl_lod, 'spread1' : cl_spread1, 'spread2' : cl_spread2, 'normals' : cl_normals}
        self.size = int(np.size(re_pts)/3)

        self.slopes = np.empty(self.size, dtype=object)
        self.aspects = np.empty(self.size, dtype=object)

        self.diffs = {}     
        keys = ['X', 'Y', 'Z', 'NX', 'NY', 'NZ', 'Distance', 'LODetection', 'Spread1', 'Spread2']
        for i in range(0, 10):
                self.diffs[keys[i]] = np.empty(self.size, dtype=object)

    def calc_differences(self):
        '''
        Calculate all the differences between related arguments of this class

        Parameters:
            self (Compare): the object itself
        
        Returns:
            dict: The differences between each related argument of both point clouds
        '''
        x1,y1,z1 = hlp.reorder_list(self.cl['pts'])
        x2,y2,z2 = hlp.reorder_list(self.re['pts'])
        nx1,ny1,nz1 = hlp.reorder_list(self.cl['normals'])
        nx2,ny2,nz2 = hlp.reorder_list(self.re['normals'])

        vec_calc = Vec_Calc()

        for i in range(0, self.size):
            self.diffs['X'][i] = hlp.sigDiff(x1, x2, i)
            self.diffs['Y'][i] = hlp.sigDiff(y1, y2, i)
            self.diffs['Z'][i] = hlp.sigDiff(z1, z2, i)
            self.diffs['NX'][i] = hlp.sigDiff(nx1, nx2, i)
            self.diffs['NY'][i] = hlp.sigDiff(ny1, ny2, i)
            self.diffs['NZ'][i] = hlp.sigDiff(nz1, nz2, i)
            self.diffs['Distance'][i] = hlp.sigDiff(self.cl['dist'], self.re['dist'], i)
            self.diffs['LODetection'][i] = hlp.sigDiff(self.cl['lod'], self.re['lod'], i)
            self.diffs['Spread1'][i] = hlp.sigDiff(self.cl['spread1'], self.re['spread1'], i)
            self.diffs['Spread2'][i] = hlp.sigDiff(self.cl['spread2'], self.re['spread2'], i)

            cl_normal = nx1[i], ny1[i], nz1[i]
            re_normal = nx2[i], ny2[i], nz2[i]

            normalized_vector = vec_calc.transform(re_normal, cl_normal)
            self.slopes[i] = vec_calc.getSlope(normalized_vector)
            self.aspects[i] = vec_calc.getAspect(normalized_vector)

        return self.diffs

    def mapDiff(self, path1, path2, proj='2d', advanced=False):
        '''
        Handle the plots of distance and lodetection. Decide if the plots are single-plotted or the distance/lodetection plots from CC and Py4dGeo are also shown.

        Parameters:
            self (Compare): The object itself.
            path1 (str): The output path for the distance plot.
            path2 (str): The output path for the lodetection plot.
            proj (str): Specifies whether the plot is plotted in 2d or 3d.
            advanced (bool): Specifies whether the differences gets plotted in comparison to the CC and Py4dGeo Distances/LODetection or just alone.
        '''
        map1 = Map_Diff(self.diffs['Distance'], self.re['pts'], "Difference in distance between CC and Py4dGeo")
        map2 = Map_Diff(self.diffs['LODetection'], self.re['pts'], "Difference in lodetection between CC and Py4dGeo")

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
        Draw the normal differences between both point clouds on a polar plot by using matplotlib:

        Parameters:
            self (Compare): The object itself.
            path (str): The output path for the normal plot.
        '''
        print('Plot normal differences of the normal vectors')

        #setup the plot
        ax = plt.axes(projection='polar')
        plt.title('Polar plot of normal differences')
        plt.xlabel('Aspect')
        plt.ylabel('Slope', rotation=0)
        ax.yaxis.set_label_coords(1, 0.75)
        ax.yaxis.set_offset_position('right')
        # set units for the x and y axes
        #plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f m'))
        #plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f m'))
        plt.scatter(self.aspects, self.slopes, s=2, c='b')
        plt.savefig(path)
        plt.close()

    def plotSpreadDiff(self, path):
        '''
        Plot differences between standard deviation spread1/spread2 between both point clouds.

        Parameters:
            self (Compare): The object itself.
            path (str): The output path for the spread plot.
        '''
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
        plt.close

    def writeStatistics(self, path):
        '''
        Write statistics (mean, median, standard-deviation) of the differences between both point clouds to a csv file:

        Parameters: 
            self (Compare): The object itself.
            path (str): The output path for the csv file.
        '''
        s = 13
        mean = np.empty(s, dtype=object)
        median = np.empty(s, dtype=object)
        std_dev = np.empty(s, dtype=object)
        mean[0] = 'Mean'
        median[0] = 'Median'
        std_dev[0] = 'Standard Deviation'

        keys = ['Statistics']
        keys.extend(list(self.diffs.keys()))
        keys.extend(['Aspect', 'Slope'])
        
        for i in range(1,s):
            if i<11: li = [x for x in self.diffs[keys[i]] if np.isnan(x) == False]
            elif i==11: li = [x for x in self.aspects if np.isnan(x) == False]
            elif i==12: li = [x for x in self.slopes if np.isnan(x) == False]

            if np.size(li) > 1:
                mean[i] = statistics.mean(li)
                median[i] = statistics.median(li)
                std_dev[i] = statistics.stdev(li)
            else:
                mean[i] = None
                median[i] = None
                std_dev[i] = None

        with open(path, mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(keys)
            writer.writerow(mean)
            writer.writerow(median)
            writer.writerow(std_dev)

    def writeDiff(self, path):
        '''
        Write all the differences between both point clouds to a csv file.

        Parameters:
            self (Compare): The object itself.
            path (str): The output path for the csv file.
        '''
        with open(path, mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
            header = ['X Coord', 'Y Coord', 'Z Coord']
            header.extend(list(self.diffs.keys()))
            header.extend(['Aspect', 'Slope'])
            writer.writerow(header)

            x,y,z = hlp.reorder_list(self.re['pts'])

            for i in range(0, self.size):
                writer.writerow([x[i],
                                y[i],
                                z[i],
                                self.diffs['X'][i],
                                self.diffs['Y'][i],
                                self.diffs['Z'][i],
                                self.diffs['NX'][i],
                                self.diffs['NY'][i],
                                self.diffs['NZ'][i],
                                self.diffs['Distance'][i],
                                self.diffs['LODetection'][i],
                                self.diffs['Spread1'][i],
                                self.diffs['Spread2'][i],
                                self.aspects[i],
                                self.slopes[i]])

def checkParams(skip):
    '''
    Check if console arguments are provided. The arguments are handled by the ArgumentParser class.

    Parameters:
        skip (bool): Instead of using the console arguments, default parameters are set, for development purposes

    Returns:
        tuple:
            - str -- Path to the first point cloud
            - str -- Path to the second point cloud
            - str -- Path to the corepoint file
            - str -- Path to the output directory
            - str -- Path to the CC parameter file
            - str -- Defines if the lodetection/distances plot gets mapped in 3d or 2d
            - bool -- Defines if the lodetection/distances plot gets mapped in advanced mode or not
    '''
    if not skip:
        parser = argparse.ArgumentParser()
        parser.add_argument(dest='cloud1', type=argparse.FileType('r'), help='Path to the first pointcloud')
        parser.add_argument(dest='cloud2', type=argparse.FileType('r'), help='Path to the second pointcloud')
        parser.add_argument(dest='param_file', type=argparse.FileType('r'), help='path to the CC parameter file')
        parser.add_argument('-o', '--output_dir', dest='output_dir', type=str, help='path to the output directory', nargs=1)
        parser.add_argument('-c', '--corepoints', dest='corepoints', type=argparse.FileType('r'), help='path to the corepoints file', nargs=1)
        parser.add_argument('-2', '--plot_2d', dest='plot_2d', action='store_true', help='Plot differences in lodetection/distances in 2d instead of 3d')
        parser.add_argument('-a', '--advanced_dist_plot', dest='advanced_dist_plot', action='store_true', help='Plot differences in lodetection/distances in advanced mode for better comparability')
        
        args = parser.parse_args()
        crpts = args.corepoints[0].name if args.corepoints else False
        out_dir = args.output_dir[0] if args.output_dir else 'output' # default output directory if not specified
        return args.cloud1.name, args.cloud2.name, crpts, out_dir, args.param_file.name, '2d' if args.plot_2d else '3d', args.advanced_dist_plot

    else: return 'data/test1.xyz', 'data/test2.xyz', False, 'output', 'm3c2_params.txt', '2d', False

def main():
    '''
    Main part of the programm. Will only be executed when the file is executed directly from the command line, not by implementing it in another package.
    '''
    test = True # for debugging and developing use default parameters
    if len(sys.argv) > 1: test = False
    skip = True # for skipping the calculations in py4dgeo and cloudcompare, e.g. if calculated clouds already exist
    
    PATH_CLOUD1, PATH_CLOUD2, PATH_COREPTS, OUTPUT_DIR, PARAMS, PROJECTION, ADVANCED = checkParams(test)  
    if platform.system() == 'Windows': CC_BIN = 'C:/Programme/CloudCompare/CloudCompare'
    elif platform.system() == 'Linux': CC_BIN = 'org.cloudcompare.CloudCompare'

    OUTPUT = {'cc' : '/CC_Output.laz', 
            'py4d' : '/Py4dGeo_Output.laz', 
            'diffs' : '/m3c2_eval_diffs.csv', 
            'stats' : '/m3c2_eval_stats.csv', 
            'normal_diff' : '/plot_normals_dev', 
            'distance_diff' : '/map_diff_distance', 
            'lod_diff' : '/map_diff_lodetection',
            'spread_diff' : '/hist_diff_spread'}

    for out in OUTPUT.items(): OUTPUT[out[0]] = OUTPUT_DIR + out[1]

    if not os.path.exists(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)

    if not skip: 
        print('Calculate Py4dGeo output')
        py4d = Py4d_M3C2(path1=PATH_CLOUD1, path2=PATH_CLOUD2, corepoint_path=PATH_COREPTS, output_path=OUTPUT['py4d'], params=PARAMS)
        py4d.run()

    if not skip: 
        print('Calculate CloudCompare output')
        if PATH_COREPTS: 
            spaces = '   ' # 2 resp. 3 spaces before filename are needed to save only the calculated cloud and not the input clouds in CloudCompare
            os.system('{} -silent -NO_TIMESTAMP -auto_save on -c_export_fmt las -o {} -o {} -o {} -M3C2 {} -NORMALS_TO_SFS -SAVE_CLOUDS FILE "{}{}"'.format(CC_BIN, PATH_CLOUD1, PATH_CLOUD2, PATH_COREPTS, PARAMS, spaces, OUTPUT['cc']))
        else:
            spaces = '  '
            os.system('{} -silent -NO_TIMESTAMP -auto_save on -c_export_fmt las -o {} -o {} -M3C2 {} -NORMALS_TO_SFS -SAVE_CLOUDS FILE "{}{}"'.format(CC_BIN, PATH_CLOUD1, PATH_CLOUD2, PARAMS, spaces, OUTPUT['cc']))

    print('Read CloudCompare file')
    reference = hlp.read_las(OUTPUT['cc'], get_attributes=True)
    print('Read Py4dGeo file')
    cloud = hlp.read_las(OUTPUT['py4d'], get_attributes=True)

    re_normals = np.array([reference[1]['NormalX'], reference[1]['NormalY'], reference[1]['NormalZ']])
    cl_normals = np.array([cloud[1]['NormalX'], cloud[1]['NormalY'], cloud[1]['NormalZ']])
    re_normals = np.transpose(re_normals)
    cl_normals = np.transpose(cl_normals)

    comp = Compare(reference[0], 
            (reference[1])['M3C2__distance'], 
            (reference[1])['distance__uncertainty'], 
            (reference[1])['STD_cloud1'], 
            (reference[1])['STD_cloud2'], 
            re_normals, 
            cloud[0], 
            (cloud[1])['M3C2__distance'], 
            (cloud[1])['distance__uncertainty'], 
            (cloud[1])['STD_cloud1'], 
            (cloud[1])['STD_cloud2'], 
            cl_normals)

    comp.calc_differences()
    comp.plotNormDiff(OUTPUT['normal_diff'])
    comp.mapDiff(OUTPUT['distance_diff'], OUTPUT['lod_diff'], PROJECTION, ADVANCED)
    comp.plotSpreadDiff(OUTPUT['spread_diff'])
    comp.writeStatistics(OUTPUT['stats'])
    comp.writeDiff(OUTPUT['diffs'])

    # remove temporary files saved by CC
    if os.path.exists(PATH_CLOUD1[:-4] + '_M3C2__NORM_TO_SF.las'): os.remove(PATH_CLOUD1[:-4] + '_M3C2__NORM_TO_SF.las')
    if os.path.exists(PATH_CLOUD1[:-4] + '_M3C2.las'): os.remove(PATH_CLOUD1[:-4] + '_M3C2.las')

if __name__ == '__main__':
    main()
