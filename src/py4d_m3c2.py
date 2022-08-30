import py4dgeo
import helper as hlp
import numpy as np

from map_diff import Map_Diff

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
            corepoint_path (str): (optional) The path to a corepoint file. If not provided, the first point cloud is used instead.
            output_path (str): (optional) The name and path for the output file. If not provided, the output won't be saved.
            params (dict/str): (optional) Either the path to a CC params file or a dictionary featuring the keys: 'cyl_radii', normal_radii', 'max_distance' and 'registration_error'. If not provided default parameters are set.
            compr: (optional) A compression rate for the corepoints if no corepoint file is given. That means only every n-th point is used. Default value is 1, so no compression at all.
        '''
        self.output_path = output_path
        self.params = params

        self.epoch1, self.epoch2 = self.read(path1, path2)

        if corepoint_path: self.corepoints = self.read(corepoint_path).cloud
        else: self.corepoints = self.epoch1.cloud[::compr]

        if not self.params:
            self.params = {"cyl_radii":(0.5,), "normal_radii":(4.0, 0.5, 7.5),"max_distance":(15),"registration_error":(0.0024)}
        elif type(self.params) == str:
            self.read_cc_params()

    def mapDiff(self, path1, path2, proj='2d'):
        '''
        Handle the plots of distance and lodetection. define if the plot should be in 3d or 2d.

        Parameters:
            self (Compare): The object itself.
            path1 (str): The output path for the distance plot.
            path2 (str): The output path for the lodetection plot.
            dimension (bool): Specifies whether the plot is plotted in 2d(True) or 3d(False).
        '''
        map1 = Map_Diff(self.distances, self.corepoints, "M3C2 distances")
        map2 = Map_Diff(self.uncertainties['lodetection'], self.corepoints, "M3C2 lodetection")
        print('Plot differences on a {} map'.format(proj))
        map1.mapDiff(path1, True, proj)    
        map2.mapDiff(path2, True, proj)

    def write_to_xyz(self):
        '''
        Write a point cloud (coordinates) with additional parameters to an ascii file.

        Parameters:
            self (Py4d_M3C2): The object itself.
        '''
        print(self.corepoints)
        f = open(self.output_path, "w")
        f.write("//X Y Z M3C2__distance distance__uncertainty STD_cloud1 STD_cloud2 NormalX NormalY NormalZ\n")
        for i in range(0, int(np.size(self.corepoints)/3)):
            x,y,z = self.corepoints[i]
            nx,ny,nz = self.normals[i]
            f.write("{} {} {} {} {} {} {} {} {} {} {} {}\n".format(str(x), str(y), str(z),
                                                                str(self.distances[i]), str(self.uncertainties['lodetection'][i]),
                                                                str((self.uncertainties["spread1"])[i]), str((self.uncertainties["spread2"])[i]),
                                                                str((self.uncertainties["num_samples1"])[i]), str((self.uncertainties["num_samples2"])[i]),
                                                                str(nx), str(ny), str(nz)))
        f.close()

    def write_to_las(self):
        '''
        Write a point cloud (coordinates) with additional parameters to a las/laz file.

        Parameters:
            self (Py4d_M3C2): The object itself.
        '''
        attributes = {"M3C2__distance" : self.distances, 
                        "distance__uncertainty" : self.uncertainties["lodetection"], 
                        "NormalX" : self.normals[0:,0], 
                        "NormalY" : self.normals[0:,1], 
                        "NormalZ" : self.normals[0:,2], 
                        "STD_cloud1" : self.uncertainties["spread1"], 
                        "STD_cloud2" : self.uncertainties["spread2"],
                        "Npoints_cloud1" : self.uncertainties["num_samples1"],
                        "Npoints_cloud2" : self.uncertainties["num_samples2"]}

        hlp.write_las(self.corepoints, self.output_path, attribute_dict=attributes)

    def write(self):
        '''
        Handle writing to different filetypes(ascii and las/laz), so theres no need to change the function when using a different file extension.

        Parameters:
            self (Py4d_M3C2): The object itself.
        '''
        if self.output_path[-3:] == "las" or self.output_path[-3:] == "laz":
            self.write_to_las()
        elif self.output_path[-3:] == "xyz" or self.output_path[-3:] == "txt":
            self.write_to_xyz()
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
                    'registration_error': float(dc['RegistrationError']),
                    'robust_aggr': dc['UseMedian'],
                    'orientation_vector': orientation_mapping[prefered_orientation]}
        
        # Multi-Scale Mode
        if dc['NormalMode'] == '2': self.params['normal_radii'] = (float(dc['NormalMinScale']), float(dc['NormalStep']), float(dc['NormalMaxScale']))

    def read_params(self):
        '''
        Read the required parameters from a given file out and store them in a dictionary.
        
        Parameters:
            self (Py4d_M3C2): The object itself.
        
        Returns:
            dict: A dictionary containing the required parameters for the m3c2-algorithm.
        '''
        dc = {}
        f = open(self.params)
        for i in f.readlines():
            i = i.split("\n")[0]        
            if i != "[General]":
                i = i.split("=")
                dc.update({i[0]:i[1]})
        params = {"cyl_radii" : (float(dc["SearchScale"])/2,), 
                "normal_radii" : (float(dc["NormalScale"])/2,), 
                "max_distance" : float(dc["SearchDepth"]), 
                "registration_error": float(dc["RegistrationError"])}
        
        if dc["NormalMode"] == "2": params["normal_radii"] = (float(dc["NormalMinScale"]), float(dc["NormalStep"]), float(dc["NormalMaxScale"]))
        return params

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

    def read_with_magic(self, *path, other_epoch=None, **parse_opts):
        '''
        Handle reading epochs from different file types(ascii and las/laz), so theres no need to change the function when using a different file extension.
        
        Parameters:
            self (Py4d_M3C2): The object itself.
            *path (str): The path to a point cloud file. Can also handle multiple files.
        '''
        import magic #pip install python-magic-bin

        filetype = magic.from_file(path[0], mime=True)
        if filetype == 'application/octet-stream':
            return py4dgeo.read_from_las(*path, other_epoch=other_epoch)
        elif filetype == 'text/plain':
            return py4dgeo.read_from_xyz(*path, other_epoch=other_epoch, comments="//", **parse_opts)
        else:
            print("File type has to be ascii or las")
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

def main():
    '''
    Main part for instatianting a Py4d_M3C2 object. Will only be executed when the file is executed directly from the command line, not by implementing it in another package.
    '''
    # specify the parameters in a dictionary
    #params = {"cyl_radii":(0.025,), "normal_radii":(0.025,),"max_distance":(0.5),"registration_error":(0.0)}

    py4d = Py4d_M3C2(path1='data/test1.xyz', 
                    path2='data/test2.xyz', 
                    params='m3c2_params.txt', 
                    output_path='run.xyz')
    py4d.run()
    py4d.mapDiff('output/py4d_distance', 'output/py4d_lodetection', '2d')

if __name__ == "__main__":
    main()