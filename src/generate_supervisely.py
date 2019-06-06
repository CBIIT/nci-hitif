import json
import numpy as np
from PIL import  Image
import io
import base64
import zlib
import cv2
from string import Template


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
            print(indexes)
            print(index)
            raise

    obj_dict["points"]["exterior"] = poly_list
    obj_dict["classTitle"] = title
    return obj_dict

def create_bitmap(indexes, title):
    """
    Returns the bitmap dictionary in supervisely format
	Arguments:
		indexes: tuple(2)
            Contains the origin (e.g., tuple(2)) of the bitmap and the 2d numy bitmap  

    Returns: dictionary
    """
    template = '''
    {
        "description": "",
        "tags": [],
        "bitmap": {
            "origin": [
              1177,
              931
            ],
            "data": "eJwBvwJA/YlQTkcNC ... AEDm2GYAAAJn"
        },
        "points": {
            "exterior": [],
            "interior": []
        }
    }
    '''

    origin = indexes[0]
    bitmap_str = mask_2_base64(indexes[1])
    obj_dict = json.loads(template)

    obj_dict["bitmap"]["origin"] = [int(origin[1]), int(origin[0])]
    obj_dict["bitmap"]["data"] =  bitmap_str
    obj_dict["classTitle"] = title
    return obj_dict

def create_ann(size, objects, dic_generator,  bb=None, already_processed=None):
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
        dic_generator: functor
            a functor to the function that generates the object dictionaries 
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
            object_dict = dic_generator(nucleus, "Nucleus-processed")
            object_list.append(object_dict)

    for nucleus in objects:
        object_dict = dic_generator(nucleus, "Nucleus")
        object_list.append(object_dict)

    img_dict["objects"] = object_list 

    return img_dict


def generate_masks(annotations, shape="bitmap"):
    """
    Returns the mask array with every object has a unique id

    Arguments
        annotations: foldername
            The folder name where the supervisely annotations exists
        shape: string
            The shape of the objects in supervisely. They can be either "bitmap" or "polygon"

    Returns
        masks: np (uint16)
            a mask array where the pixels that belong to a given object have a unique uint id. 
    """

    assert ((shape == "bitmap") or (shape == "polygon")), "shape should be either bitmap or polygon. Received:{}".format(shape)


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
    print(height, width)

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
                if shape == "polygon":
                    poly_list = obj_dict["points"]["exterior"] 
                    contour_list = [[int(point[0]), int(point[1])] for point in poly_list ]
                    nd_contour = np.array(contour_list).astype("int64")
                    cv2.fillPoly(masks,pts=[nd_contour], color=object_id)
                elif shape == "bitmap":
                    supervisely_bitmap = obj_dict["bitmap"]["data"]
                    bitmap_bool = base64_2_mask(supervisely_bitmap)
                    origin_x = obj_dict["bitmap"]["origin"][1]
                    origin_y = obj_dict["bitmap"]["origin"][0]
                    x_length = bitmap_bool.shape[0] 
                    y_length = bitmap_bool.shape[1]
                    print(bitmap_bool.shape)
                    bitmap_uint16 = np.zeros_like(bitmap_bool, dtype=np.uint16)
                    bitmap_uint16[bitmap_bool] = object_id
                    output_loc = masks[origin_x:origin_x + x_length, origin_y: origin_y + y_length]
                    print(origin_x)
                    print(output_loc.shape)
                    np.copyto(output_loc, bitmap_uint16, where = bitmap_bool)

                object_id += 1 
                print(object_id)
    return masks

def base64_2_mask(s):
    """
    Return numpy bitmask from the supervisely base64 string 
    https://docs.supervise.ly/ann_format/ 
    Argument:
        s: base64 
            The input supervisely base 64 string

    Returns:
        np array of the mask
    """
    z = zlib.decompress(base64.b64decode(s))
    n = np.fromstring(z, np.uint8)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
    return mask

def mask_2_base64(mask):
    """
    Return the supervisely base64 string for a  numpy bitmask. 
    https://docs.supervise.ly/ann_format/ 
    Argument:
        mask: np(2D)
            The numpy bitmask

    Returns:
        string:  
            The supervisely base64 string
    """
    img_pil = Image.fromarray(np.array(mask, dtype=np.uint8))
    img_pil.putpalette([0,0,0,255,255,255])
    bytes_io = io.BytesIO()
    img_pil.save(bytes_io, format='PNG', transparency=0, optimize=0)
    bytes = bytes_io.getvalue()
    return base64.b64encode(zlib.compress(bytes)).decode('utf-8')

def generate_project_template(shape="bitmap"):
    """
    Returns the meta json file for the supervisely project.
    Arguments:
        shape: string ["bitmap", "polygon"]
            The shape of the objects. 
    """

    meta_template = '''
    {
        "classes": [
            {
                "title": "Nucleus", 
                "color": "#6CC751", 
                "shape": "$SHAPE"
            }, 
            {
                "title": "Nucleus-processed", 
                "color": "#6FF6D8", 
                "shape": "$SHAPE"
            }, 
            {
                "title": "bounding-box", 
                "color": "#F6AFD8", 
                "shape": "rectangle"
            }
        ], 
        "tags_images": ["corrected"], 
        "tags_objects": []
    }   
    '''
   
    values = dict(SHAPE=shape)
    meta_string = Template(meta_template).substitute(values)
    return json.loads(meta_string)
