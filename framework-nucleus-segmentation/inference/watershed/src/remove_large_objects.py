# This code has been generated based on remove_small_object (SkImage) module.

import numpy as np

def remove_large_objects(ar, max_size=64, connectivity=1, in_place=False):
    """Remove objects larger than the specified size.
    Expects ar to be an array with labeled objects, and removes objects
    smaller than max_size. If `ar` is bool, the image is first labeled.
    This leads to potentially different behavior for bool and 0-and-1
    arrays.
    Parameters
    ----------
    ar : ndarray (arbitrary shape, int or bool type)
        The array containing the objects of interest. If the array type is
        int, the ints must be non-negative.
    max_size : int, optional (default: 64)
        The largest allowable object size.
    connectivity : int, {1, 2, ..., ar.ndim}, optional (default: 1)
        The connectivity defining the neighborhood of a pixel. Used during
        labelling if `ar` is bool.
    in_place : bool, optional (default: False)
        If ``True``, remove the objects in the input array itself.
        Otherwise, make a copy.
    Raises
    ------
    TypeError
        If the input array is of an invalid type, such as float or string.
    ValueError
        If the input array contains negative values.
    Returns
    -------
    out : ndarray, same shape and type as input `ar`
        The input array with small connected components removed.
    # Examples
    # --------
    # >>> from skimage import morphology
    # >>> a = np.array([[0, 0, 0, 1, 0],
    # ...               [1, 1, 1, 0, 0],
    # ...               [1, 1, 1, 0, 1]], bool)
    # >>> b = morphology.remove_small_objects(a, 6)
    # >>> b
    # array([[False, False, False, False, False],
    #        [ True,  True,  True, False, False],
    #        [ True,  True,  True, False, False]])
    # >>> c = morphology.remove_small_objects(a, 7, connectivity=2)
    # >>> c
    # array([[False, False, False,  True, False],
    #        [ True,  True,  True, False, False],
    #        [ True,  True,  True, False, False]])
    # >>> d = morphology.remove_small_objects(a, 6, in_place=True)
    # >>> d is a
    # True
    # """
    # # Raising type error if not int or bool
    # _check_dtype_supported(ar)

    if in_place:
        out = ar
    else:
        out = ar.copy()

    if max_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = ndi.generate_binary_structure(ar.ndim, connectivity)
        ccs = np.zeros_like(ar, dtype=np.int32)
        ndi.label(ar, selem, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError("Negative value labels are not supported. Try "
                         "relabeling the input with `scipy.ndimage.label` or "
                         "`skimage.morphology.label`.")

    if len(component_sizes) == 2 and out.dtype != bool:
        warn("Only one label was provided to `remove_small_objects`. "
             "Did you mean to use a boolean array?")

    too_large = component_sizes > max_size
    too_large_mask = too_large[ccs]
    out[too_large_mask] = 0

    return out