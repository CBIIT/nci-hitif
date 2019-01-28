import json
import numpy as np


def create_rectangle(indexes):
    """
    Returns the rectange dictionary in supervisely format
	Arguments:
		indexes: tuple(4)
            Contains the bounding box for this ROI in the oder
            left, top, right, bottom 
    """

    template = '''
    {   
        "bitmap": null,
        "classTitle": "bounding-box",
        "description": "",
        "points": {
            "interior": []
        },
        "tags": []
    }
    '''
    exterior = [[indexes[0], indexes[1]],[indexes[2], indexes[3]]]
    obj_dict = json.loads(template)
    obj_dict["points"]["exterior"] = exterior 

    return obj_dict


def create_polygon(indexes, title):
    """
    Returns the object dictionary in supervisely format
    Arguments:
        indexes: ndarray of ints (nvertices, 2)
            2D indexes of points of the polygon
        title: string
            The type of the polygon as defined in the project meta

    returns: dict
    """

    #        "classTitle" : "Nucleus",
    template = '''
    {
            "bitmap": null,
            "description": "",
            "points": {
                "interior": []
            },
            "tags": []
    }
    '''
    obj_dict = json.loads(template)

    #Create the list of indexes 
    poly_list = []
    for index in indexes:
        try:
            poly_list.append([int(index[0]), int(index[1])]) 
        except Exception as e:
            print indexes
            print index

    obj_dict["points"]["exterior"] = poly_list
    obj_dict["classTitle"] = title
    return obj_dict

def create_ann(size, objects, bb=None, already_processed=None):
    """
	Returns the json dictionary for an image

    Arguments:
        size: tuple (height, width)
            The dimension of the image 
        objects: list(ndarray)
            The list of polygons of the ojbects. Every polygon indexes are defined 
            as an ndarray
        bb: tuple(4) 
            Contains the bounding box for this ROI in the oder
            left, upper, right, lower
        already_processed: list(ndarray)
            The list of polygons of the ojbects that are processed in a differenr ROI. Every polygon indexes are defined 
            as an ndarray
            
    Returns: dict        
        The superviserly dictionary for this image
    """
    
    template = '''
    {
        "description": "",
        "objects": [],
        "size": {
            "height": null,
            "width": null
        },
        "tags": []
    }
    '''
    img_dict = json.loads(template)
    img_dict["size"]["height"] = size[0]
    img_dict["size"]["width"] = size[1]

    object_list = []

    #This order will make bb always be in the bottom in supervisely
    if bb is not None:
        bb_object = create_rectangle(bb)
        object_list.append(bb_object)

    if already_processed is not None:
        for nucleus in already_processed:
            object_dict = create_polygon(nucleus, "Nucleus-processed")
            object_list.append(object_dict)

    for nucleus in objects:
        object_dict = create_polygon(nucleus, "Nucleus")
        object_list.append(object_dict)

    img_dict["objects"] = object_list 

    return img_dict


def generate_masks(annotations):
    """
    Returns the mask array with every object has a unique id

    Arguments
        annotations: foldername
            The folder name where the supervisely annotations exists
    """

    #Find all annotations in the annotation folder 
    import cv2
    import glob
    import os
    json_regex = "*.json"
    file_regex = os.path.join(annotations, json_regex)
    json_files = glob.glob(file_regex)
    
    #Get the size of the image
    first_file = json_files[0]
    with open(first_file, 'r') as json_file:
        json_str = json_file.read()    
    json_data = json.loads(json_str)
    height = int(json_data['size']['height'])
    width = int(json_data['size']['width'])

    #Create the numpy array
    masks = np.zeros((height, width), dtype=np.uint16)

    #For every file:
    object_id = 1
    for filename in json_files:

        with open(filename, 'r') as json_file:
            json_str = json_file.read()    
        json_data = json.loads(json_str)
        objects = json_data["objects"]
        
        #For every object
        for obj_dict in objects:
            if obj_dict["classTitle"] != "Nucleus":
                pass
            else:
                poly_list = obj_dict["points"]["exterior"] 
                contour_list = [[int(point[0]), int(point[1])] for point in poly_list ]
                nd_contour = np.array(contour_list).astype("int64")
                cv2.fillPoly(masks,pts=[nd_contour], color=object_id)
                object_id += 1 



    return masks
