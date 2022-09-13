import py4dgeo
import file_handle as fhandle
import numpy as np

class Py4d_M3C2:
    '''
    A class for calculating distances between point clouds by implementing the m3c2-algorithm from the py4dgeo library. 
    
    Py4d_M3C2 also provides some extra functionality, such as general read/write functions for ascii(xyz/txt) and las/laz files. It is also able to read parameters from a CloudCompare param file. And for a better impression it also can plot the distances and lodetection by using the matplotlib package.

    Attributes: 
        epoch1 (py4dgeo.epoch): An epoch object of the first point cloud.
        epoch2 (py4dgeo.epoch): An epoch object of theh second cloud.
        output_path (str): The name and path for the output file.
        params (str/dict): Either the path to a CC params file or a dictionary of params
        corepoints: Either the pointcloud from a separate corepoint file or a subsampled version of the first point cloud.
    '''
    def __init__(self, path1, path2, corepoint_path=False, output_path=False, params=False, compr=1):
        '''
        The class constructor.

        Parameters:
            self (Py4d_M3C2): The object itself.
            path1 (str): The path to the first point cloud file.
            path2 (str): The path to the second point cloud file.
            corepoint_path (str/bool): The path to a corepoint file. If not provided, the first point cloud is used instead.
            output_path (str/bool): The name and path for the output file. If not provided, the output won't be saved.
            params (dict/str/bool): Either the path to a CC params file or a dictionary featuring the keys: 'cyl_radii', normal_radii', 'max_distance' and 'registration_error'. If not provided default parameters are set.
            compr (int): A compression rate for the corepoints if no corepoint file is given. That means only every n-th point is used. Default value is 1, so no compression at all.
        '''
        self.output_path = output_path
        self.params = params

        self.epoch1, self.epoch2 = self.read(path1, path2)

        if corepoint_path: self.corepoints = self.read(corepoint_path).cloud
        else: self.corepoints = self.epoch1.cloud[::compr]

        self.exp_samples = False
        self.exp_spread = False

        if not self.params:
            self.params = {"cyl_radii":(0.5,), "normal_radii":(4.0, 0.5, 7.5),"max_distance":(15),"registration_error":(0.0024)}
        elif type(self.params) == str:
            self.read_cc_params()

    def write(self, cc_mode=True):
        '''
        Handle writing to different filetypes(ascii and las/laz), so theres no need to change the function when using a different file extension.

        Parameters:
            self (Py4d_M3C2): The object itself.
            cc_mode (bool): Specifies if the header is written with CC vocabulary or py4dgeo vocabulary.
        '''
        if cc_mode: keys = ['M3C2__distance', 'distance__uncertainty', 'STD_cloud1', 'STD_cloud2', 'Npoints_cloud1', 'Npoints_cloud2', 'NormalX', 'NormalY', 'NormalZ']
        else: keys = ['distance', 'lodetection', 'spread1', 'spread2', 'num_samples1', 'num_samples2', 'nx', 'ny', 'nz']

        attributes={keys[0] : self.distances, 
                    keys[1] : self.uncertainties["lodetection"], 
                    keys[2] : self.uncertainties["spread1"], 
                    keys[3] : self.uncertainties["spread2"],
                    keys[4] : self.uncertainties["num_samples1"],
                    keys[5] : self.uncertainties["num_samples2"],
                    keys[6] : self.normals[0:,0], 
                    keys[7] : self.normals[0:,1], 
                    keys[8] : self.normals[0:,2]}

        if not self.exp_spread: 
            attributes.pop(keys[2])
            attributes.pop(keys[3])
        if not self.exp_samples:
            attributes.pop(keys[4])
            attributes.pop(keys[5])

        if self.output_path[-3:] == "las" or self.output_path[-3:] == "laz":
            fhandle.write_las(self.corepoints, self.output_path, attributes)
        elif self.output_path[-3:] == "xyz" or self.output_path[-3:] == "txt":
            fhandle.write_xyz(self.corepoints, self.output_path, attributes)
        else:
            print("File extension has to be las, laz, xyz or txt")
            quit()

    def read_cc_params(self):
        '''
        Read the required parameters from a given file out and store them in a dictionary.
        
        Parameters:
            self (Py4d_M3C2): The object itself.
        
        Returns:
            dict: A dictionary containing the required parameters for the m3c2-algorithm.
        '''
        dc = {}
        with open(self.params, mode='r') as file:
            for line in file.readlines():
                line = line.split('\n')[0] #remove line break   
                if line != '[General]':
                    line_li = line.split('=')
                    dc.update({line_li[0]:line_li[1]})

        # Orientation
        # X, -X, Y, -Y, Z, -Z, Barycenter, -Barycenter, Origin, -Origin
        orientation_mapping = np.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1], [0,0,1], [0,0,1], [0,0,0], [0,0,0]])
        prefered_orientation = int(dc['NormalPreferedOri'])
        if prefered_orientation >5 and prefered_orientation <8: 
            print(f"Orientation vector is set to Z due to a CC prefered orientation of '{prefered_orientation}', which isn't implemented yet")

        self.params = {'cyl_radii' : (float(dc['SearchScale'])/2,), 
                    'normal_radii' : (float(dc['NormalScale'])/2,), 
                    'max_distance' : float(dc['SearchDepth']), 
                    'robust_aggr': dc['UseMedian'],
                    'orientation_vector': orientation_mapping[prefered_orientation]}
        
        if dc['RegistrationErrorEnabled'] == 'true': self.params.update({'registration_error': float(dc['RegistrationError'])})

        if dc['ExportDensityAtProjScale'] == 'true': self.exp_samples = True
        if dc['ExportStdDevInfo'] == 'true': self.exp_spread = True

        # Multi-Scale Mode
        if dc['NormalMode'] == '2': self.params['normal_radii'] = (float(dc['NormalMinScale']), float(dc['NormalStep']), float(dc['NormalMaxScale']))

    def read(self, *path, other_epoch=None, **parse_opts):
        '''
        Handle reading epochs from different file types(ascii and las/laz), so theres no need to change the function when using a different file extension.
        
        Parameters:
            self (Py4d_M3C2): The object itself.
            *path (str): The path to a point cloud file. Can also handle multiple files.
        '''
        from pathlib import Path
        extension = Path(path[0]).suffix
        if extension == (".las" or ".laz"):
            return py4dgeo.read_from_las(*path, other_epoch=other_epoch)
        elif extension == (".xyz" or ".txt"):
            return py4dgeo.read_from_xyz(*path, other_epoch=other_epoch, comments="//", **parse_opts)
        else:
            print("File extension has to be las, laz, xyz or txt")
            quit()

    def run(self):
        '''
        Main function for calculating the distances. Implements the m3c2-algorithm from the py4dgeo library.

        Parameters: 
            self (Py4d_M3C2): The object itself.

        Returns: 
            dict: The calculated m3c2 distances.
        '''
        m3c2 = py4dgeo.M3C2(
            epochs=(self.epoch1, self.epoch2),
            corepoints=self.corepoints,
            **self.params
        )
        self.distances, self.uncertainties = m3c2.run() 
        self.normals = m3c2.corepoint_normals
        if self.output_path: self.write()
        return self.distances