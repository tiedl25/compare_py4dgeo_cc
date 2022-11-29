import os
import platform
import argparse
import numpy as np

import file_handle as fhandle
from py4d_m3c2 import Py4d_M3C2
from compare import Compare

def checkParams():
    '''
    Check if console arguments are provided. The arguments are handled by the ArgumentParser class.

    Returns:
        tuple:
            - str -- Path to the first point cloud
            - str -- Path to the second point cloud
            - str -- Path to the corepoint file
            - str -- Path to the output directory
            - str -- Path to the CC parameter file
            - str -- Defines if the lodetection/distances plot gets mapped in 3d or 2d
            - bool -- Defines if the lodetection/distances plot gets mapped in advanced mode or not
            - str -- The point size for the lodetection/distance plots
            - str -- The file format for the pointcloud files
            - str -- The path to the CC binary
            - bool -- If the py4dgeo calculations should use CC normals
            - bool -- Skip the calculations in py4dgeo/CC and instead provide pointcloud files
            - bool -- Repeat the difference calculations but skip the py4dgeo/CC calculations
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='cloud1', type=argparse.FileType('r'), help='Path to the first pointcloud or path to the calculated cc pointcloud if skip argument is set')
    parser.add_argument(dest='cloud2', type=argparse.FileType('r'), help='Path to the second pointcloud or path to the calculated py4dgeo pointcloud if skip argument is set')
    parser.add_argument(dest='param_file', type=argparse.FileType('r'), help='Path to the CC parameter file')
    parser.add_argument('-o', '--output_dir', dest='output_dir', type=str, help='Path to the output directory', nargs=1)
    parser.add_argument('-c', '--corepoints', dest='corepoints', type=argparse.FileType('r'), help='Path to the corepoints file', nargs=1)
    parser.add_argument('-2', '--plot_2d', dest='plot_2d', action='store_true', help='Plot differences in lodetection/distances in 2d instead of 3d')
    parser.add_argument('-a', '--advanced_dist_plot', dest='advanced_dist_plot', action='store_true', help='Plot differences in lodetection/distances in advanced mode for better comparability')
    parser.add_argument('-p', '--point_size', dest='point_size', type=int, help='Set the point size for the plots.', nargs=1)
    parser.add_argument('-s', '--skip', dest='skip', action='store_true', help='Skip the calculations in Py4dGeo and CloudCompare if the data already exists. Therefore you have to provide the calculated clouds as cloud1/2 parameter.')
    parser.add_argument('-r', '--repeat', dest='repeat', action='store_true', help='Repeat the comparing and plotting if the data already exists')
    parser.add_argument('-f', '--file_format', dest='file_format', type=str, help='Specify the output format for CloudCompare and Py4dgeo (las/laz, xyz/txt/asc)', nargs=1)
    parser.add_argument('-n', '--cc_normals', dest='cc_normals', action='store_true', help='Specify if py4dgeo should use the calculated normals from CloudCompare')
    parser.add_argument('-b', '--cc_binary', dest='cc_binary', type=str, help='Path to the CloudCompare executable/binary', nargs=1)

    args = parser.parse_args()

    crpts = args.corepoints[0].name if args.corepoints else False
    out_dir = args.output_dir[0] if args.output_dir else 'output' # default output directory if not specified
    file_format = args.file_format[0] if args.file_format else 'laz' # default file format if not specified
    point_size = args.point_size[0] if args.point_size else 10 # default point size if not specified
    cc_binary = args.cc_binary[0] if args.cc_binary else False

    return (args.cloud1.name, 
            args.cloud2.name, 
            crpts, 
            out_dir, 
            args.param_file.name, 
            '2d' if args.plot_2d else '3d', 
            args.advanced_dist_plot, 
            point_size, 
            file_format, 
            cc_binary, 
            args.cc_normals, 
            args.skip, 
            args.repeat)

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
    #check if normal information is available and if not set normal array to 'nan'
    try:
        cl_normals = np.array([cloud[1]['NormalX'], cloud[1]['NormalY'], cloud[1]['NormalZ']])
        cl_normals = np.transpose(cl_normals)
    except:
        print('No normal information available')
        li = []
        for i in range(0, int(np.size(cloud[0])/3)):
            li.append(np.nan)
        cl_normals = np.array([li,li,li])
        cl_normals = np.transpose(cl_normals)   

    #check if spread information is available
    if 'STD_cloud1' in cloud[1]:
        cl_spread = (cloud[1]['STD_cloud1'], cloud[1]['STD_cloud2'])
    else: 
        cl_spread = None

    #check if num_sample information is available
    if 'Npoints_cloud1' in cloud[1]:
        cl_num_samples = (cloud[1]['Npoints_cloud1'], cloud[1]['Npoints_cloud2'])
    else:
        cl_num_samples = None

    return cl_normals, cl_spread, cl_num_samples

if __name__ == '__main__':
    PATH_CLOUD1, PATH_CLOUD2, PATH_COREPTS, OUTPUT_DIR, PARAMS, PROJECTION, ADVANCED, POINT_SIZE, FILE_FORMAT, CC_BIN, CC_NORMALS, skip, repeat = checkParams()  
    
    #use the default cc binary if there is none specified
    if CC_BIN == False:
        if platform.system() == 'Windows': CC_BIN = 'C:/Programme/CloudCompare/CloudCompare' #default installation directory under Windows
        elif platform.system() == 'Linux': 
            CC_BIN = os.popen('which cloudcompare').read() #get installation directory under Linux
            if CC_BIN == '': CC_BIN = 'org.cloudcompare.CloudCompare' #default flatpak installation directory under Linux

    OUTPUT = {'cc' : '/CC_Output.' + FILE_FORMAT, 
            'py4d' : '/Py4dGeo_Output.' + FILE_FORMAT, 
            'diffs' : '/m3c2_eval_diffs.csv', 
            'stats' : '/m3c2_eval_stats.csv', 
            'normal_diff' : '/plot_normals_dev', 
            'normal_hist' : '/plot_normals_hist',
            'distance_diff' : '/map_diff_distance', 
            'lod_diff' : '/map_diff_lodetection',
            'spread_diff' : '/hist_diff_spread',
            'sample_diff' : '/hist_diff_sample',
            'cloud' : '/diff_cloud.' + FILE_FORMAT}

    for out in OUTPUT.items(): OUTPUT[out[0]] = OUTPUT_DIR + out[1] # add output directory to string

    if skip:
        OUTPUT['cc'] = PATH_CLOUD1
        OUTPUT['py4d'] = PATH_CLOUD2

    if not os.path.exists(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)

    #calculate clouds in py4dgeo and CC
    if not repeat and not skip: 
        if FILE_FORMAT in ['las', 'laz']: 
            export = 'las'
            normals = '-NORMALS_TO_SFS '
        elif FILE_FORMAT in ['xyz', 'txt', 'asc']: 
            export = 'asc -add_header'
            normals = ''
        else: 
            print('File format not supported')
            os._exit(0)

        print('Calculate CloudCompare output')       
        if PATH_COREPTS: 
            spaces = '   ' # 2 resp. 3 spaces before filename are needed to save only the calculated cloud and not the input clouds in CloudCompare
            os.system('{} -silent -NO_TIMESTAMP -c_export_fmt {} -o {} -o {} -o {} -M3C2 {} -SAVE_CLOUDS FILE {}"{}{}"'.format(CC_BIN, export, PATH_CLOUD1, PATH_CLOUD2, PATH_COREPTS, PARAMS, normals, spaces, OUTPUT['cc']))
        else:
            spaces = '  '
            os.system('{} -silent -NO_TIMESTAMP -c_export_fmt {} -o {} -o {} -M3C2 {} -SAVE_CLOUDS FILE {}"{}{}"'.format(CC_BIN, export, PATH_CLOUD1, PATH_CLOUD2, PARAMS, normals, spaces, OUTPUT['cc']))

        print('Calculate Py4dGeo output')
        py4d = Py4d_M3C2(path1=PATH_CLOUD1, path2=PATH_CLOUD2, corepoint_path=PATH_COREPTS, output_path=OUTPUT['py4d'], params=PARAMS, cc_normals=OUTPUT['cc'] if CC_NORMALS != False else None)
        py4d.run()

    #read calculated clouds
    if FILE_FORMAT in ['las', 'laz']:
        print('Read CloudCompare file')
        reference = fhandle.read_las(OUTPUT['cc'], get_attributes=True)
        print('Read Py4dGeo file')
        cloud = fhandle.read_las(OUTPUT['py4d'], get_attributes=True)
    elif FILE_FORMAT in ['xyz', 'txt', 'asc']:
        print('Read CloudCompare file')
        reference = fhandle.read_xyz(OUTPUT['cc'], get_attributes=True)
        print('Read Py4dGeo file')
        cloud = fhandle.read_xyz(OUTPUT['py4d'], get_attributes=True)

    #check if some parameters are available and reorder them
    re_normals, re_spread, re_num_samples = reorder(reference)
    cl_normals, cl_spread, cl_num_samples = reorder(cloud)

    #CC has different output naming schemes for the distance and uncertainties
    if 'M3C2__distance' in reference[1]: sep = '__'
    elif 'M3C2_distance' in reference[1]: sep = '_'
    else: sep = ' '

    compare = Compare(reference[0], re_normals,
                    reference[1][f'M3C2{sep}distance'], 
                    reference[1][f'distance{sep}uncertainty'], 
                    cloud[0], cl_normals, 
                    cloud[1]['M3C2__distance'], 
                    cloud[1]['distance__uncertainty'])
    
    #calculate differences
    compare.calculate_differences()

    #different plots
    compare.plotNormDiff(OUTPUT['normal_diff'])
    compare.plotNormHist(OUTPUT['normal_hist'])
    compare.mapDiff(OUTPUT['distance_diff'], OUTPUT['lod_diff'], PROJECTION, ADVANCED, POINT_SIZE)
    compare.spreadDiff(re_spread, cl_spread, OUTPUT['spread_diff'], plot=True)
    compare.sampleDiff(re_num_samples, cl_num_samples, OUTPUT['sample_diff'], plot=True)

    #statistics
    compare.writeStatistics(OUTPUT['stats'])

    #output -> first method writes output to csv and second method to specified file format
    #compare.writeDiff(OUTPUT['diffs'])
    compare.writeCloud(OUTPUT['cloud'])