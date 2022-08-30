import os
import numpy as np

def sigDiff(val1, val2, i="false", r=20):
    '''
    Determine if the difference between two values is significant or not.

    Parameters:
        val1 (float): The first value.
        val2 (float): The second value.
        i (int): (optional) Sets the index in an array/list.
        r (int): (optional) Sets the number of decimal places which are included. The default value is 20

    Returns:
        float: The significant difference.
    '''
    if i != "false": return 0 if round(abs(val1[i]-val2[i]),r) == 0 or None else abs(val1[i]-val2[i])
    return 0 if round(abs(val1-val2),r) == 0 or None else abs(val1-val2)

def reorder_list(li):
    '''
    Reorder a list by columns instead of rows.

    Parameters:
        li (list): The input list.
    Returns: 
        list: The reordered list
    '''
    return (li[0:,0], li[0:,1], li[0:,2])

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

    import laspy
    import numpy as np

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

    import laspy

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
    
#doenst work properly
def xyz_to_las(path):
    '''
    Get the Numsamples from CC. CC has no option available to directly save Numsamples/Npoints to a las file, but to an ascii file. The CC output needs to be saved in an xyz file, which then gets translated to las.
    
    Parameters:
        path (str): Path to the xyz file.
    '''
    attributes = {}

    f = open(path,"r")
    lines = f.readlines()
    header = lines[0][2:]
    header = header.strip('\n')
    li = header.split(' ')

    ptrs = []

    attr_keys = ['normal_scale', 'Npoints_cloud1', 'Npoints_cloud2', 'STD_cloud1', 'STD_cloud2', 'significant_change', 'distance__uncertainty', 'M3C2__distance', 'NormalX', 'NormalY', 'NormalZ']
    for i in attr_keys: attributes[i] = np.empty(np.size(lines))
    
    for i in range(1, np.size(lines)):
        line = lines[i].split(' ')
        for j in range(0, np.size(attr_keys)): attributes[attr_keys[j]][i] = line[j+3]
        ptrs.append([float(line[0]), float(line[1]), float(line[2])])
        
    ptrs = np.asarray(ptrs)

    f.close()

    os.remove(path)
    path = path[:-3]
    path = path + 'laz'
    
    write_las(ptrs, path, attribute_dict=attributes)
