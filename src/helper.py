import numpy as np
import laspy

def read_las(pointcloudfile,get_attributes=False,useevery=1):
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

def write_las(outpoints,outfilepath,attribute_dict={}):
    '''
    Write a point cloud to a las/laz file

    Parameters:
        outpoints (numpy.array): 3D array of points to be written to output file
        outfilepath (str): specification of output file
        attribute_dict (dict): dictionary of attributes (key: name of attribute; value: 1D array of attribute values in order of points in 'outpoints'); if not specified, dictionary is empty and nothing is added
    '''

    hdr = laspy.LasHeader(version="1.4", point_format=6)

    las = laspy.LasData(hdr)

    las.x = outpoints[:, 0]
    las.y = outpoints[:, 1]
    las.z = outpoints[:, 2]
    for key,vals in attribute_dict.items():
        try:
            las[key] = vals
        except:
            las.add_extra_dim(laspy.ExtraBytesParams(
                name=key,
                type=type(vals[0])
                ))
            las[key] = vals

    las.write(outfilepath)

    return