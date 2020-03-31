
#Author: George Zaki 
#Date: 04-02-2019

def generate_mask_features(mask, **kwargs):
    """
    Generate features from the unit16 masks
    Parameters:
    -----------

        mask: 2D np array unit16
            The mask array with a unique id for every instance
        kwargs: dict {"bitmap", 
                      "distance_transform":cutoff, 
                      "edge", 
                      "blured_contour":sigma, 
                      "blured_outer_contour":sigma, 
                      "eroded"
                     }
            dictionary of features to be generated with any specific parameter for that feature

    Returns:
        features: dictionary
            dictionary with features as keys and values as the 2D numpy array of that feature 
    """

    import numpy as np

    assert (len(mask.shape) == 2), "mask shape should be 2D numpy array. Received:{}".format(maks.shape)
    
    #place holder for output
    features = {}
    
    bit_mask_str = "bitmask"
    dist_trans_str = "distance_transform"
    eroded_str = "erosion"
    edge_str = "edge"
    blured_ctr_str = "blured_contour"
    blured_outer_ctr_str = "blured_outer_contour"

    
    if bit_mask_str in kwargs:
        features[bit_mask_str] = np.zeros_like(mask, dtype = 'uint8')
        features[bit_mask_str][mask > 0] = 1


    eroded = blured = edge = dist_trans = blured_outer = False

    if dist_trans_str in kwargs:
        dist_trans = True
        features[dist_trans_str] = np.zeros_like(mask, dtype = 'float')
        dt_cutoff = kwargs[dist_trans_str]
        if dt_cutoff != None:
            assert (dt_cutoff > 0 and dt_cutoff <= 1), \
                "Distance transform cutoff should be between 0 and 1. Received:{} ".format(dt_cutoff)


    from scipy.ndimage.morphology import binary_erosion, binary_dilation
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import gaussian


    if eroded_str in kwargs:
        features[eroded_str] = np.zeros_like(mask, dtype = 'uint8')
        eroded = True 

    if edge_str in kwargs:
        features[edge_str] = np.zeros_like(mask, dtype = 'uint8')
        edge = True

    if blured_ctr_str in kwargs:
        blured = True
        blur_sigma = kwargs[blured_ctr_str]
        if blur_sigma != None:
            assert (blur_sigma > 0), \
                "The sigma value for the guassian blur should be greater than zero. Received:{} ".format(blur_sigma)
        else:
            #The the default value of 1
            blur_sigma = 1


    regions = regionprops(mask)
    for region in regions:

        min_row = region.bbox[0]
        max_row = region.bbox[2]
        min_col = region.bbox[1]
        max_col = region.bbox[3]

        if dist_trans:
        
            #To generate distance trasform or the outer edge, add a boundry of false around the region
            region_mask_with_boundaries = np.zeros((region.image.shape[0] + 2, region.image.shape[1] + 2),dtype=np.bool)
            region_mask_with_boundaries[1:-1, 1:-1] = region.image        

        if dist_trans_str in kwargs:
            dt = ndimage.distance_transform_edt(region_mask_with_boundaries)
            max_value = np.amax(dt)    
            dt = dt / max_value
            if dt_cutoff != None:
                #only set the pixels that has their distance transform > cutoff the max of the cell
                dt[dt >= dt_cutoff] = 1
                dt[dt < dt_cutoff] = 0
            region_center_blobs = features[dist_trans_str][min_row:max_row, min_col:max_col] 

            #The "where" argument in copyto makes sure only the ROI is copied, not the whole bounding box.
            np.copyto(region_center_blobs, dt[1:-1, 1:-1], where = region.image)

        #To generate edges, erode than substract from binary mask,
        if edge or eroded or blured: 
            region_int_image = region.image.astype(np.uint16)
            eroded_region = binary_erosion(region_int_image)

            if eroded:
                output_loc = features[eroded_str][min_row:max_row, min_col:max_col]
                np.copyto(output_loc, eroded_region, where = region.image)

            if edge or blured:
                region_edge = region_int_image - eroded_region

            if edge:
                output_loc = features[edge_str][min_row:max_row, min_col:max_col]
                np.copyto(output_loc, region_edge , where = region.image)

    if blured:
        features[blured_ctr_str] = gaussian(features[edge_str].astype('float32'), sigma=1, mode='constant')        

    return features
