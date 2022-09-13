import sys
import os
import platform
import argparse
import numpy as np

import file_handle as fhandle
from py4d_m3c2 import Py4d_M3C2
from compare import Compare

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

def reorder(cloud):
    '''
    Rearrange certain parameters.

    Parameters:
        cloud (numpy.ndarray): All the parameters of a the point cloud.

    Returns:
        tuple:
            - numpy.ndarray -- The normals of the cloud.
            - numpy.ndarray -- The spread information. 
            - numpy.ndarray -- The num_sample information.
    '''

    cl_normals = np.array([cloud[1]['NormalX'], cloud[1]['NormalY'], cloud[1]['NormalZ']])
    cl_normals = np.transpose(cl_normals)

    if 'STD_cloud1' in cloud[1]:
        cl_spread = (cloud[1]['STD_cloud1'], cloud[1]['STD_cloud2'])
    else: 
        cl_spread = None

    if 'Npoints_cloud1' in cloud[1]:
        cl_num_samples = (cloud[1]['Npoints_cloud1'], cloud[1]['Npoints_cloud2'])
    else:
        cl_num_samples = None

    return cl_normals, cl_spread, cl_num_samples

if __name__ == '__main__':
    test = True # for debugging and developing use default parameters
    if len(sys.argv) > 1: test = False
    skip = True # for skipping the calculations in py4dgeo and cloudcompare, e.g. if calculated clouds already exist

    PATH_CLOUD1, PATH_CLOUD2, PATH_COREPTS, OUTPUT_DIR, PARAMS, PROJECTION, ADVANCED = checkParams(test)  
    if platform.system() == 'Windows': CC_BIN = 'C:/Programme/CloudCompare/CloudCompare' #default installation directory
    elif platform.system() == 'Linux': CC_BIN = os.popen('which cloudcompare').read()

    if CC_BIN == '': 
        print('CloudCompare Binary not found')
        quit()

    OUTPUT = {'cc' : '/CC_Output.laz', 
            'py4d' : '/Py4dGeo_Output.laz', 
            'diffs' : '/m3c2_eval_diffs.csv', 
            'stats' : '/m3c2_eval_stats.csv', 
            'normal_diff' : '/plot_normals_dev', 
            'distance_diff' : '/map_diff_distance', 
            'lod_diff' : '/map_diff_lodetection',
            'spread_diff' : '/hist_diff_spread',
            'sample_diff' : '/hist_diff_sample'}

    for out in OUTPUT.items(): OUTPUT[out[0]] = OUTPUT_DIR + out[1] # add output directory to string

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
    reference = fhandle.read_las(OUTPUT['cc'], get_attributes=True)
    print('Read Py4dGeo file')
    cloud = fhandle.read_las(OUTPUT['py4d'], get_attributes=True)

    re_normals, re_spread, re_num_samples = reorder(reference)
    cl_normals, cl_spread, cl_num_samples = reorder(cloud)

    comp = Compare(reference[0], re_normals,
                    reference[1]['M3C2__distance'], 
                    reference[1]['distance__uncertainty'], 
                    cloud[0], cl_normals, 
                    cloud[1]['M3C2__distance'], 
                    cloud[1]['distance__uncertainty'])
    
    comp.calc_differences()
    comp.plotNormDiff(OUTPUT['normal_diff'])
    comp.mapDiff(OUTPUT['distance_diff'], OUTPUT['lod_diff'], PROJECTION, ADVANCED)
    #comp.spreadDiff(re_spread, cl_spread, OUTPUT['spread_diff'], plot=True)
    comp.sampleDiff(re_num_samples, cl_num_samples, OUTPUT['sample_diff'], plot=True)
    comp.writeStatistics(OUTPUT['stats'])
    comp.writeDiff(OUTPUT['diffs'])

    # remove temporary files saved by CC
    if os.path.exists(PATH_CLOUD1[:-4] + '_M3C2__NORM_TO_SF.las'): os.remove(PATH_CLOUD1[:-4] + '_M3C2__NORM_TO_SF.las')
    if os.path.exists(PATH_CLOUD1[:-4] + '_M3C2.las'): os.remove(PATH_CLOUD1[:-4] + '_M3C2.las')