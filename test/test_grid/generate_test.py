import numpy as np

masks = np.zeros((10,10), dtype = np.uint8)

masks[0:4, 2:5] = 128
masks[1:5, 3:9] = 129
print(masks)
print(masks.shape)

#Save masks as image
from skimage.io import imsave
imsave("input.tif", masks)
imsave("masks.tif", masks)

