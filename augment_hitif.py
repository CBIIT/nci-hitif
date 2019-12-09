
from image_augmentation import AugmentationSettings, augment_images

class hitif_aug(AugmentationSettings):


    def __init__(self, configuration):
        """
        Initialized the configuration prameters 
    
        Arguments:
            configuration: file pointer
                The hitif configuration file 
        
        """
        import configparser   
        
        config = configparser.ConfigParser()
        config.read(configuration)
    
        #Parse the augmentation parameters
        aug_prms = config['augmentation']
        self.CLAHE= eval(aug_prms['AllChannelsCLAHE'])
        self.saturation= eval(aug_prms['Saturation'])													 
        self.impulse_noise = eval(aug_prms['ImpulseNoise'])
        self.gaussian_blur = eval(aug_prms['GaussianBlur'])
        self.poisson = eval(aug_prms['AdditivePoissonNoise'])
        self.median = eval(aug_prms['MedianBlur'])
        self.flip = float(aug_prms["flip"])
        self.rotate = eval(aug_prms["rotate"])
        self.gamma = eval(aug_prms["GammaContrast"])
        self.gaussian_noise = eval(aug_prms["AdditiveGaussianNoise"])
        self.dropout= eval(aug_prms["Dropout"])
        self.salt_pepper = eval(aug_prms["SaltAndPepper"])
        self.multiply = eval(aug_prms['Multiply'])


        from imgaug import augmenters as iaa
        import imgaug as ia
        self.augmenters = {} 
        augmenters = self.augmenters

        import numpy as np
        seed = np.random.randint(0, 2**31-1)
        ia.seed(seed)        

        #Affine augmentation
        augmenters["fliplr"] = iaa.Fliplr(self.flip)
        augmenters["flipud"] = iaa.Flipud(self.flip)
        augmenters["rotate"] = iaa.Affine(rotate=[self.rotate[0],\
                                                  self.rotate[1],\
                                                  self.rotate[2]])
        
        #Contrast augmentation
        #augmenters["CLAHE"] = iaa.AllChannelsCLAHE(self.CLAHE)
        augmenters["CLAHE"] = iaa.CLAHE(self.CLAHE)
        #augmenters["CLAHE"] = iaa.AllChannelsCLAHE(self.CLAHE[0], self.CLAHE[1], self.CLAHE[2],self.CLAHE[3])
        augmenters["gamma"] = iaa.GammaContrast(self.gamma, True)
        augmenters["saturation"] = iaa.Saturation(self.saturation)

        #Blur augmenters
        augmenters["median_blur"] = iaa.MedianBlur(self.median)
        augmenters["gaussian_blur"] = iaa.GaussianBlur(self.gaussian_blur)

        #Noise augmenters
        augmenters["impulse_noise"] = iaa.ImpulseNoise(self.impulse_noise)
        augmenters["poisson_noise"] = iaa.AdditivePoissonNoise(lam=self.poisson)
        augmenters["gaussian_noise"] = iaa.AdditiveGaussianNoise(scale = self.gaussian_noise)
        augmenters["dropout"] = iaa.Dropout(self.dropout)
        augmenters["multiply"] = iaa.MultiplyElementwise(self.multiply)
        augmenters["salt_pepper"] = iaa.SaltAndPepper(p=self.salt_pepper)

    def composite_sequence(self):
        """Return the composite sequence to run, i.e., a set of transformations to all be applied to a set of images and/or masks.

        :returns: Sequential object from the augmenters module of the imgaug package
        """

        augmenters = self.augmenters
    
        from imgaug import augmenters as iaa
        self.seq = iaa.OneOf([
            #pick up one of CLAHE or gamma
            iaa.OneOf([
                augmenters["CLAHE"],
                augmenters["gamma"],
                augmenters["saturation"]
            ]),

            iaa.OneOf([
                iaa.OneOf([
                    augmenters["impulse_noise"], 
                    augmenters["poisson_noise"], 
                    augmenters["gaussian_noise"], 
                    augmenters["multiply"],
                    augmenters["salt_pepper"]
                ])
            ]),

            iaa.OneOf([
                    augmenters["gaussian_blur"],
                    augmenters["median_blur"]
            ])
            
        ])
    
        return self.seq

    def individual_seqs_and_outnames(self):
        """Return a list of individual sequences to run, i.e., a set of transformations to be applied one-by-one to a set of images and/or masks in order to see what the augmentations do individually.

        :returns: List of Sequential objects from the augmenters module of the imgaug package
        """

        from imgaug import augmenters as iaa

        
        augmentation_tasks = []
        augmenters = self.augmenters
        for name, augmentation in self.augmenters.items():
            augmentation_tasks.append([augmentation, name])

        return augmentation_tasks