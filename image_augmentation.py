"""
.. module:: image_augmentation
   :synopsis: This is the module used for augmenting images, the main function of which is augment_images().
        This module implements the imgaug package with reasonable, working-out-of-the-box settings as a template for further usage.
        Note: This module the imgaug package, which can be acquired on GitHub at https://github.com/aleju/imgaug.
.. moduleauthor:: Andrew Weisman <andrew.weisman@nih.gov>
"""


class AugmentationSettings:
    """Class specifying the augmentation parameters and sequences to apply and from which a new class can be inherited if desired;
    default values/sequences are already-working sets.
    """
    def __init__(self):
        # Define a nice default set of augmentation parameters
        self.flip_factor = 0.5
        self.add_vals = (-30,30)
        self.multiply_factors = (0.75,1.25)
        self.gaussian_blur_sigma = (0,0.75)
        self.average_blur_pixels = (1,2)
        self.median_blur_pixels = (1,3)
        self.gaussian_noise_vals = (0,0.035*255)
        self.contrast_normalization_factors = (0.75,1.25)
        self.rotation_degrees = (-90,90)
        self.scale_factors = (0.8,1.2)

    # Define a nice default composite sequence of augmentations
    def composite_sequence(self):
        """Return the composite sequence to run, i.e., a set of transformations to all be applied to a set of images and/or masks.

        :returns: Sequential object from the augmenters module of the imgaug package
        """
        from imgaug import augmenters as iaa
        return(iaa.Sequential([
            iaa.Flipud(self.flip_factor),
            iaa.Fliplr(self.flip_factor),
            iaa.Sometimes(0.5,
                iaa.OneOf([
                    iaa.GaussianBlur(sigma=self.gaussian_blur_sigma),
                    #iaa.AverageBlur(k=self.average_blur_pixels), # There at least used to be a weird shift for this augmentation so I removed it
                    iaa.MedianBlur(k=self.median_blur_pixels)
                ])),
            iaa.ContrastNormalization(self.contrast_normalization_factors),
            iaa.AdditiveGaussianNoise(loc=0,scale=(self.gaussian_noise_vals)),
            iaa.OneOf([
                iaa.Add(self.add_vals),
                iaa.Multiply(self.multiply_factors)
            ]),
            iaa.Affine(
                rotate=self.rotation_degrees,
                scale=self.scale_factors
            )
        ]))

    # Define a nice default set of individual augmentations in order to see what they do individually
    def individual_seqs_and_outnames(self):
        """Return a list of individual sequences to run, i.e., a set of transformations to be applied one-by-one to a set of images and/or masks in order to see what the augmentations do individually.

        :returns: List of Sequential objects from the augmenters module of the imgaug package
        """
        from imgaug import augmenters as iaa
        return([
            [iaa.Sequential(iaa.Flipud(1)), 'flipud'],
            [iaa.Sequential(iaa.Fliplr(1)), 'fliplr'],
            [iaa.Sequential(iaa.Add((self.add_vals[0],self.add_vals[0]))), 'add_low'],
            [iaa.Sequential(iaa.Add((self.add_vals[1],self.add_vals[1]))), 'add_high'],
            [iaa.Sequential(iaa.Multiply((self.multiply_factors[0],self.multiply_factors[0]))), 'multiply_low'],
            [iaa.Sequential(iaa.Multiply((self.multiply_factors[1],self.multiply_factors[1]))), 'multiply_high'],
            [iaa.Sequential(iaa.GaussianBlur(sigma=(self.gaussian_blur_sigma[1],self.gaussian_blur_sigma[1]))), 'gaussian_blur'],
            [iaa.Sequential(iaa.AverageBlur(k=(self.average_blur_pixels[1],self.average_blur_pixels[1]))), 'average_blur'],
            [iaa.Sequential(iaa.MedianBlur(k=(self.median_blur_pixels[1],self.median_blur_pixels[1]))), 'median_blur'],
            [iaa.Sequential(iaa.AdditiveGaussianNoise(loc=0,scale=(self.gaussian_noise_vals[1],self.gaussian_noise_vals[1]))), 'additive_gaussian_noise'],
            [iaa.Sequential(iaa.ContrastNormalization((self.contrast_normalization_factors[0],self.contrast_normalization_factors[0]))), 'contrast_norm_low'],
            [iaa.Sequential(iaa.ContrastNormalization((self.contrast_normalization_factors[1],self.contrast_normalization_factors[1]))), 'contrast_norm_high'],
            [iaa.Sequential(iaa.Affine(rotate=self.rotation_degrees,scale=self.scale_factors)), 'affine']])

#def augment_images(images, masks=None, num_aug=1, do_composite=True, output_dir=None, composite_sequence=None, individual_seqs_and_outnames=None, aug_params=None):
def augment_images(images, masks=None, num_aug=1, do_composite=True, output_dir=None, AugSettings=None):
    """Augment images and/or masks.

    :param images: Images to augment;
        NumPy array of shape (H,W), (H,W,3), (N,H,W), or (N,H,W,3);
        values can be in range [0,1], [0,2^8-1], or [0,2^16-1]
    :param masks:
        (Optional) Masks to correspondingly augment;
        NumPy array of shape (H,W) or (N,H,W);
        values are 0 and positive integers
    :param num_aug:
        (Optional) Number of augmentations to perform;
        number of output images will be N * num_aug
    :type num_aug: int.
    :param do_composite:
        (Optional) Whether to do composite augmentations (multiple augmentations at once; True) or individual augmentations (for observing the effect of each augmentation; False)
    :type do_composite: bool.
    :param output_dir:
        (Optional) If not set to None, location where .tif images should be saved for observation purposes;
        if set to None, no saving will be done
    :type output_dir: str.
    :param AugSettings:
        (Optional)  Object of the class that inherists from AugmentationSettings specifying the augmentation parameters and sequences to apply; whether composite or individual augmentation is determined by the do_composite parameter

        if set to None, the default base class AugmentationSettings, defined in this module, is used;
        to customize the augmentation parameters only, instantiate from AugmentationSettings and modify the instance variables;
        to customize the sequences as well, inherit a custom derived class from AugmentationSettings and override the composite_sequence and/or individual_seqs_and_outnames methods
    :type AugSettingsClass: cls.
    :returns:
        * If do_composite=True: augmented images ((N,H,W,C)), and, if masks were input, augmented masks ((N,H,W)); these are both NumPy arrays of dtype='uint8'\n
        * If do_composite=False: list of augmented images ((N,H,W,C)), one for each individual augmentation; these are all NumPy arrays of dtype='uint8'\n
    """
    # Tested in its own function in the testing module

    # Temporarily hardcoded until modules is implemented
    #imgaug_repo = '/data/BIDS-HPC/public/software/checkouts/imgaug'
    #imgaug_repo = '/Users/weismanal/checkouts/imgaug'

    # Since imgaug is a GitHub repo, we need to point to where the clone is so we can import the library
    import sys
    #sys.path.append(imgaug_repo)

    # Import relevant modules
    import imgaug as ia
    from imgaug import augmenters as iaa
    import numpy as np
    from skimage import io
    import utils
    #from . import utils

    # Instantiate from the input augmentation settings class, whether a base class or derived
    if AugSettings == None:
        aug_settings = AugmentationSettings()
    else:
        aug_settings = AugSettings

    import os
    if output_dir is not None:
       if not os.path.isdir(output_dir):
         os.mkdir(output_dir)

    # Initialize the randomizer - note that by setting this you will get the same augmentations every run of the script
    # Causes issues when we want to run the same images through the pipeline
    #ia.seed(1)

    # Preprocess the images and masks (note that masks in imgaug are assumed to be integer-based, bool, or float-based; we're settling on int16)
    images = utils.normalize_images(images,1) # ensure the images are uint8; inputs can therefore be [0,1], [0,2^8-1], or [0,2^16-1]
    images, masks = utils.stack_and_color_images(images, masks=masks) # allow for flexible dimensions of input images and masks

    # If multiple augmentations are desired, set up imgaug to do the multiple augmentations
    # images is (N,H,W,C) and masks is (N,H,W) due to the stack_and_color_images() function called above
    images = np.tile(images,(num_aug,1,1,1))
    if masks is not None:
        masks = np.tile(masks,(num_aug,1,1))

    # Get the shapes of the images and masks
    N, H, W, C = images.shape

    # Optionally save .tif files to put the input in the same format as the output in order to do visual sanity checks
    if output_dir is not None:
        io.imsave(output_dir+'/'+'input.tif',images)
        if masks is not None:
            utils.quick_overlay_output(images, masks, output_dir+'/'+'input-overlay.tif')

    # If we want to do each augmentation individually in order to manually inspect exactly what each augmentation does...
    if not do_composite:

        # For each augmentation...
        image_aug_list = []
        for aug_num, seq_and_outname in enumerate(aug_settings.individual_seqs_and_outnames(), start=1):
            seq = seq_and_outname[0]
            outname = '{:02d}'.format(aug_num) + '-' + seq_and_outname[1]

            # For each image in the stack...
            image_aug = []
            for i in range(N):
                seq_det = seq.to_deterministic()
                img_channels = []
                for ic in range(C):
                    img_channels.append(seq_det.augment_image(images[i,:,:,ic].squeeze()))
                image_aug.append(img_channels)

            # Convert the lists to numpy arrays
            image_aug = np.array(image_aug,dtype=np.uint8)
            # Transpose the indices since they come out like (N,C,H,W)
            image_aug = np.transpose(image_aug,[0,2,3,1]) # comes out as (N,H,W,C)

            # Save the augmented images to a list
            image_aug_list.append(image_aug)

            # Optionally save the outputs as .tif files so you can check the individual augmentations
            if output_dir is not None:
                io.imsave(output_dir+'/'+outname+'.tif',image_aug)

        # Return the list of numpy arrays
        return(image_aug_list)
        
    # If we want to do a composite augmentation...
    else:

        # Determine if there are masks present
        if masks is not None:
            do_masks = True
        else:
            do_masks = False
    
        # Define the arrays to contain the stack for the current single augmentation
        image_aug = []
        if do_masks:
            mask_aug = []

        # For each image in the stack...
        for i in range(N):
            seq_det = aug_settings.composite_sequence().to_deterministic()
            img_channels = []
            for ic in range(C):
                img_channels.append(seq_det.augment_image(images[i,:,:,ic].squeeze()))
            image_aug.append(img_channels)
            if do_masks:
                #mask_segmap = ia.SegmentationMapOnImage(masks[i,:,:].squeeze(),shape=(H,W),nb_classes=2)
                mask_segmap = ia.SegmentationMapOnImage(masks[i,:,:].squeeze(),shape=(H,W),nb_classes=np.max(masks[i,:,:])+1)
                mask_aug.append(seq_det.augment_segmentation_maps([mask_segmap])[0].get_arr_int())

        # Convert the lists to numpy arrays
        image_aug = np.array(image_aug,dtype=np.uint8)
        if do_masks:
            mask_aug = np.array(mask_aug,dtype=np.int16) # comes out as (N,H,W)

        # Transpose the indices since they come out like (N,C,H,W)
        image_aug = np.transpose(image_aug,[0,2,3,1]) # comes out as (N,H,W,C)

        # Optionally save the output as a .tif file and overlay the masks on the images and save as .tif
        if output_dir is not None:
            io.imsave(output_dir+'/'+'output.tif',image_aug)
            if do_masks:
                utils.quick_overlay_output(image_aug, mask_aug, output_dir+'/'+'output-overlay.tif')

        # Return the augmented images (and masks if they were input)
        if not do_masks:
            return(image_aug)
        else:
            return(image_aug, mask_aug)
