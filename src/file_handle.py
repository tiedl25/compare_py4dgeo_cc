import numpy as np
import laspy

def read_las(pointcloudfile, get_attributes=False, useevery=1):
    '''
    Read a pointcloud from a las/laz file.

    Parameters:
        pointcloudfile (str): specification of input file
        get_attributes (bool): if True, will return all attributes in file, otherwise will only return XYZ (default is False)
        useevery (int): value specifies every n-th point to use from input, i.e. simple subsampling (default is 1, i.e. returning every point)
    
    Returns:
        numpy.ndarray: 3D array of points (x,y,z) of length number of points in input file (or subsampled by 'useevery')
    '''

    # read the file
    inFile = laspy.read(pointcloudfile)

    # get the coordinates (XYZ)
    coords = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()
    coords = coords[::useevery, :]

    if get_attributes == False:
        return (coords)

    else:
        las_fields= [info.name for info in inFile.points.point_format.dimensions]
        attributes = {}

        for las_field in las_fields[3:]: # skip the X,Y,Z fields
            attributes[las_field] = inFile.points[las_field][::useevery]

            try:
                normals = getattr(inFile, 'normals', None)
            except:
                normals = None
            if normals is None:
                try:
                    n0 = inFile.points["point"]["NormalX"]
                    n1 = inFile.points["point"]["NormalY"]
                    n2 = inFile.points["point"]["NormalZ"]
                    normals = np.stack((n0,n1,n2)).T
                except:
                    normals = None

        return (coords, attributes, normals)

def write_las(points, path, attributes={}):
    '''
    Write a point cloud to a las/laz file

    Parameters:
        points (numpy.array): 3D array of points to be written to output file
        path (str): specification of output file
        attributes (dict): dictionary of attributes (key: name of attribute; value: 1D array of attribute values in order of points in 'outpoints'); if not specified, dictionary is empty and nothing is added
    '''
    hdr = laspy.LasHeader(version="1.4", point_format=6)

    las = laspy.LasData(hdr)

    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    for key,vals in attributes.items():
        try:
            las[key] = vals
        except:
            las.add_extra_dim(laspy.ExtraBytesParams(
                name=key,
                type=type(vals[0])
                ))
            las[key] = vals

    las.write(path)

def read_xyz(pointcloudfile, get_attributes=False):
    '''Read a pointcloud from an ascii file.'''
    dc = {}
    with open(pointcloudfile, mode='r') as file:
        lines = file.readlines()
        for line in lines: 
            line = line.split()
            if line[0][:2] == '//':
                line[0] = line[0][2:]
                for s in line:
                    dc.update({s : []})
            else:
                for s in zip(dc, line):
                    dc[s[0]].append(float(s[1]))

        coords = np.array([dc.pop('X'), dc.pop('Y'), dc.pop('Z')]).transpose()

        if 'Nx' in dc: dc.update({'NormalX' : dc.pop('Nx')})
        if 'Ny' in dc: dc.update({'NormalY' : dc.pop('Ny')})
        if 'Nz' in dc: dc.update({'NormalZ' : dc.pop('Nz')})

        for i in dc:
            dc[i] = np.array(dc[i])
            
    if not get_attributes: return(coords)
    return(coords, dc)

def write_xyz(points, path, attributes={}):
    '''
    Write a point cloud to a las/laz file

    Parameters:
        points (numpy.array): 3D array of points to be written to output file
        path (str): specification of output file
        attributes (dict): dictionary of attributes (key: name of attribute; value: 1D array of attribute values in order of points in 'outpoints'); if not specified, dictionary is empty and nothing is added
    '''
    with open(path, mode='w') as file:
        file.write('//X Y Z ')
        for s in attributes.keys(): file.write(s + ' ')
        file.write('\n')

        for i in range(0, int(np.size(points)/3)):
            x,y,z = points[i]
            file.write("{} {} {} ".format(
                        str(x), str(y), str(z)))
            for item in attributes.keys():
                file.write(str(attributes[item][i]) + ' ')
            file.write('\n')