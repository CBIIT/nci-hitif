from skimage.morphology import  binary_dilation
from scipy.ndimage.morphology import binary_erosion, binary_dilation, binary_fill_holes
from scipy import ndimage
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from packaging import version
def generate_features(mask_image, binary_dt=None):
    """
        Generate the edge and distance transform for the mask image:
        
        Arguments:
            mask: 2D np array 
                    Every object in the array should have a unique positive interger id.
            binary_dt: float or None
                    None: Return the float values for the distance transform (0-1)
                    float: Convert all all normalized distance transfor values greater than binary_dt to 1, otherwise zero.        
        Return:
            A tuple of (edges, distance_transform)
    """

    assert (len(mask_image.shape) == 2)   
    
    #place holder for output
    center_blobs = np.zeros_like(mask_image, dtype = 'float')
    edges = np.zeros_like(mask_image, 'uint16')
    areas = []
    borders = np.zeros_like(mask_image, 'uint16')
    contours = []

    #Get unique cell ids:
    regions = regionprops(mask_image)
    for region in regions:
        #To generate distance trasform, add a boundry of false around the region
        areas.append(region.area)
        region_mask = np.zeros((region.image.shape[0] + 2, region.image.shape[1] + 2),dtype=np.bool)
        region_indexes = (region_mask == True)
                               
        min_row = region.bbox[0]
        max_row = region.bbox[2]
        min_col = region.bbox[1]
        max_col = region.bbox[3]
                               
        region_mask[1:-1, 1:-1] = region.image        
        dt = ndimage.distance_transform_edt(region_mask)
        max_value = np.amax(dt)    
        dt = dt / max_value
        
        if binary_dt != None:
            #only set the pixels that has their distance transform > 0.5 the max of the cell
            dt[dt >= binary_dt] = 1
            dt[dt < binary_dt] = 0
       
        region_center_blobs = center_blobs[min_row:max_row, min_col:max_col] 
        np.copyto(region_center_blobs, dt[1:-1, 1:-1], where = region.image)
        
        #To generate edges, erode than substract from binary mask,
        region_int_image = region.image.astype(np.uint16)
        eroded = binary_erosion(region_int_image)
        region_edge = region_int_image - eroded
        output_loc = edges[min_row:max_row, min_col:max_col]
        #The "where" argument in copyto makes sure only the ROI is copied, not the whole bounding box.
        np.copyto(output_loc, region_edge , where = region.image)
        
        #Generate the contour for significantly large labels
        if region.area > 10:
            region_contour = get_contour(region_int_image.astype(np.uint8), (min_row, min_col))
            #region_contour = get_contour(region_int_image.astype(np.uint8), (min_row, min_col))
            if region_contour is not  None:                    
                contours.append((region_contour, region.label, region.centroid,region.bbox, region.image))
                  
        min_row_b = min_row if min_row == 0 else min_row -1
        max_row_b = max_row - 1 if max_row == (mask_image.shape[0]) else max_row;
        min_col_b = min_col if min_col == 0 else min_col -1
        max_col_b = max_col-1 if max_col == (mask_image.shape[1]) else max_col;

    
        borders[min_row_b:max_row_b, min_col_b] = 2
        borders[min_row_b:max_row_b, max_col_b] = 2
        borders[min_row_b, min_col_b:max_col_b] = 2
        borders[max_row_b, min_col_b:max_col_b + 1] = 2

    return [edges, center_blobs, areas, borders, contours]


def get_hull(image, origin = None):
    """
    Returns the simplicies from image
    Arguments: 
        image: 2D np array or [0,1]
        origin: tuple
            Shift the simplicies by origin if defined
    Returns
        ndarray of simplices
    """
    from scipy.spatial import ConvexHull
    points = np.where(image == 1)
    nd_points = np.zeros((points[0].shape[0], 2))
    nd_points[:,0] = points[0]
    nd_points[:,1] = points[1]
    hull = ConvexHull(nd_points)
    vertices = hull.vertices
    n_points = vertices.shape[0]
    indexes = nd_points[vertices]
    if origin != None:
        origin_x = np.ones(n_points, dtype = indexes.dtype) * origin[0]
        origin_y = np.ones(n_points, dtype = indexes.dtype) * origin[1]
        indexes = indexes + np.stack((origin_x, origin_y), axis = -1)
        
    return indexes

def get_max_region(array):
    """
        Calculate the maximum area and the id of the region that has the larget area.
        Arguments: 
            array: a binary 2D arrary
        Return
            label_image, max_area, region_id
    """
    label_image = label(array)
    region_pro = regionprops(label_image)
    max_area = 0
    max_label = -1 
    for region in region_pro:
        if region.area > max_area:
            max_area = region.area
            max_label = region.label

    return label_image, max_area, max_label


    
def get_contour(image, origin = None):
    """
    Returns the contour polygon from image
    Arguments: 
        image: 2D np array or [0,1]
        origin: tuple
            Shift the simplicies by origin if defined
    Returns
        ndarray of simplices
    """
    try:

        #Get the conected component with the maximum label
        label_image, max_area, max_label = get_max_region(image)
        label_image[label_image != max_label] = 0 
        label_image[label_image == max_label] = 1
        max_label_image = label_image.astype(image.dtype)

        #Fill any holes in the max_label_image
        filled_holes = ndimage.binary_fill_holes(max_label_image).astype(image.dtype) 
        #The contour function deletes an edge from the image in openCV         
        #I will add a border then substract 1 from all indexes
        border = cv2.copyMakeBorder(filled_holes, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0 )
        if version.parse(cv2.__version__) < version.parse("3.2"):
            contour_indexes, hierarchy = cv2.findContours(border, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            correction = 1
        else:
            contour_indexes, hierarchy = cv2.findContours(filled_holes, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            correction = 0
    except Exception as e:
            print("Can not calculate contours for region")
            fig, ax = plt.subplots(2,1)
            ax[0].imshow(image)
            #ax[1].imshow(border)
            plt.show()
            raise
    n_contours = len(contour_indexes)
    if n_contours == 0:
        return None
    elif n_contours > 1: 
        print("Found more thant one contour")
        fig, ax = plt.subplots(2,1)
        ax[0].imshow(image)
        ax[1].imshow(max_label_image)
        plt.show()
    else:
        first_contour = contour_indexes[0]
        #Simplify the contour
        epsilon = 0.00*cv2.arcLength(first_contour,True)
        approx_contour = cv2.approxPolyDP(first_contour,epsilon,True)
        if approx_contour.ndim > 2 :
            approx_contour = np.squeeze(approx_contour)
        if approx_contour.ndim != 2:
            print("Ignoring contour that consists of one point")
            print(approx_contour)
            return None
        else:        
            if origin != None:
                n_points = approx_contour.shape[0]
                #Note the returned contour is transposed, 
                #I will add the origin in reverse, and substract the correction I added earlier
                origin_x = np.ones(n_points, dtype = approx_contour.dtype) * origin[1] - correction 
                origin_y = np.ones(n_points, dtype = approx_contour.dtype) * origin[0] - correction 
                approx_contour = approx_contour + np.stack((origin_x, origin_y), axis = -1)
                #print approx_contour.shape
            return approx_contour
