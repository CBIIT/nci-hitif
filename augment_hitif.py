
from image_augmentation import AugmentationSettings, augment_images

from imgaug import augmenters as iaa
import imgaug as ia
import numpy as np

class hitif_aug(AugmentationSettings):


    def __init__(self, configuration):
        """
        Initialized the configuration prameters 
    
        Arguments:
            configuration: file pointer
                The hitif configuration file 
        
        """
        import configparser   
        
        self.config = configparser.ConfigParser()
        self.config.read(configuration)
    
        #Parse the augmentation parameters
        self.CLAHE= self.__eval__('AllChannelsCLAHE')
        self.Saturation= self.__eval__('Saturation')
        self.impulse_noise = self.__eval__('ImpulseNoise')
        self.gaussian_blur = self.__eval__('GaussianBlur')
        self.poisson = self.__eval__('AdditivePoissonNoise')
        self.median = self.__eval__('MedianBlur')
        self.flip = self.__eval__("flip")
        self.rotate = self.__eval__("rotate")
        self.gamma = self.__eval__("GammaContrast")
        self.gaussian_noise = self.__eval__("AdditiveGaussianNoise")
        self.dropout= self.__eval__("Dropout")
        self.salt_peper = self.__eval__("SaltAndPepper")


        seed = np.random.randint(0, 2**31-1)
        ia.seed(seed)

        self.augmenters = {} 
        augmenters = self.augmenters

        #Affine augmentation
        augmenters["fliplr"] = self._get_augmenter(iaa.Fliplr,self.flip, self.flip)
        augmenters["flipud"] = self._get_augmenter(iaa.Flipud,self.flip, self.flip)

        rotate_dict = {}
        if self.rotate != None:
            rotate_dict = {"rotate": [self.rotate[0], self.rotate[1], self.rotate[2]]}
        augmenters["rotate"] = self._get_augmenter(iaa.Affine, self.rotate, **rotate_dict)

        #Contrast augmentation

        #augmenters["CLAHE"] = iaa.AllChannelsCLAHE(self.CLAHE)
        augmenters["CLAHE"] = self._get_augmenter(iaa.CLAHE,self.CLAHE, self.CLAHE)
        #augmenters["CLAHE"] = iaa.AllChannelsCLAHE(self.CLAHE[0], self.CLAHE[1], self.CLAHE[2],self.CLAHE[3])
        gama_dict = {"per_channel": True} 
        augmenters["gamma"] = self._get_augmenter(iaa.GammaContrast, self.gamma, self.gamma, **gama_dict)
        #augmenters['saturation'] = iaa.Lambda(func_images=self.saturate_images, func_heatmaps=self.func_heatmaps, func_keypoints=self.func_keypoints)
        augmenters['Saturation'] = self._get_augmenter(iaa.Saturation, self.Saturation, self.Saturation)

        #Blur augmenters
        augmenters["median_blur"] = self._get_augmenter(iaa.MedianBlur, self.median, self.median)
        augmenters["gaussian_blur"] = self._get_augmenter(iaa.GaussianBlur,self.gaussian_blur, self.gaussian_blur)

        #Noise augmenters
        augmenters["impulse_noise"] = self._get_augmenter(iaa.ImpulseNoise,self.impulse_noise,self.impulse_noise)
        augmenters["poisson_noise"] = self._get_augmenter(iaa.AdditivePoissonNoise,self.poisson,self.poisson)
        gn_dict = {"scale":self.gaussian_noise}
        augmenters["gaussian_noise"] = self._get_augmenter(iaa.AdditiveGaussianNoise,self.gaussian_noise, **gn_dict)
        augmenters["dropout"] = self._get_augmenter(iaa.Dropout,self.dropout, self.dropout)



    def __eval__(self, attribute):
        """
        Either return the expression if the attribute exists in the config file, or None
        Arguments: 
            attribute: string
                The attribute name in the configuration file 
        Returns:
            The expression to be evaluated or None if the attribute does not exist.
        """
        aug_section = "augmentation"
        if self.config.has_option(aug_section, attribute):
            return eval(self.config.get(aug_section, attribute))
        else:
            print("WARNING: {0} attribute does not exist in configuration file, seeting augmentation to identity".format(attribute))
            return None
 
    def _get_augmenter(self, augmenter, attribute, *params, **kwargs):
        """
        Returns an augmenters initialized with the params if params is None, or Identity augmenter
        Arguments:
            augmenter: function pointer
                A pointer of the iaa augmenter 
            params: function attributes
                Attributes of the augmenter
                
        Returns: iaa augmenter
           Either augmenter if params are valid, or an Identity augmenter 
        """
        print(params)
        print(kwargs)
        if attribute != None:
                return augmenter(*params, **kwargs)
        else:
            #Flipup with prob zero is equivalent to Identity augmenter
            return iaa.Flipud(0)
            

    def composite_sequence(self):
        """Return the composite sequence to run, i.e., a set of transformations to all be applied to a set of images and/or masks.

        :returns: Sequential object from the augmenters module of the imgaug package
        """

        augmenters = self.augmenters
    
        from imgaug import augmenters as iaa
        self.seq = iaa.Sequential([
            #pick up one affine transformation
            iaa.OneOf([
                augmenters["fliplr"],
                augmenters["flipud"],
                augmenters["rotate"] 
            ]),

            #pick up one or tow CLAHE 
            iaa.OneOf([
                augmenters["CLAHE"],
                iaa.Sequential([
                    augmenters["CLAHE"],
                    augmenters["CLAHE"]
                ])
            ]),

            iaa.OneOf([
                iaa.OneOf([
                    augmenters["impulse_noise"], 
                    augmenters["poisson_noise"], 
                    augmenters["gaussian_noise"], 
                    augmenters["dropout"]
                ]),
                iaa.OneOf([
                    augmenters["gaussian_blur"],
                    augmenters["median_blur"]
                ])
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
