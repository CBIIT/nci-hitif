import numpy as np

class augmented_predict:

    def __init__(self, images):

        self.stacks= {}
        self.array = np.copy(images)
        self.stacks["array"] = self.array
        self.nslides = np.shape(self.array)[0]
        self.all_aug_input =  None

    def augment_inputs(self):
        self.stacks["rot90"] = np.rot90(self.array, k = 1, axes = (1,2))
        self.stacks["rot180"] = np.rot90(self.array, k = 2, axes = (1,2))
        self.stacks["rot270"] = np.rot90(self.array, k = 3, axes = (1,2))
        self.stacks["lr"] = self.array[:,...,::-1]   
        self.stacks["ud"] = self.array[:,::-1,...] 
        self.all_aug_input =  self.__pack()

    def get_augmented_patch(self):
        if self.all_aug_input is not  None:
            return self.all_aug_input 
        else:
            raise Exception("Augmented patch not generated")
        #for image in self.all_aug_input:
        #    yield image 

    def get_n_aug_stacks(self):
        """
        Returns the number of augmented images
        """
        if self.all_aug_input is not  None:
            return np.shape(self.all_aug_input)[0]
        else:
            raise Exception("Augmented patch not generated")

    def __pack(self):
        self.packing_order = ["array", "rot90", "rot180", "rot270", "lr", "ud"]
        all_fov =  []
        for augmentation in self.packing_order:
            all_fov.append(self.stacks[augmentation])
        return np.concatenate(all_fov)

    def __unpack(self, all_predictions ):
        
        index = 0
        n_stacks = self.nslides
        self.predictions={}
        for augmentation in self.packing_order:
            self.predictions[augmentation] = all_predictions[index:index + n_stacks,:]
            index = index + n_stacks

    def reduce_predictions(self, predictions, reduce_func = np.mean):
        self.__unpack(predictions)
        self.predictions["rot90"] = np.rot90(self.predictions["rot90"], k = -1, axes = (1,2))
        self.predictions["rot180"] = np.rot90(self.predictions["rot180"], k = -2, axes = (1,2))
        self.predictions["rot270"] = np.rot90(self.predictions["rot270"], k = -3, axes = (1,2))
        self.predictions["lr"] = self.predictions["lr"][:,...,::-1]
        self.predictions["ud"] = self.predictions["ud"][:,::-1,...]

        #Creare a stack of numpy predictions
        all_predictions = [ self.predictions[augmentation] for augmentation in self.predictions]
        stacked_pred =  np.stack(all_predictions)
       
        #Reduce all predictions
        return reduce_func(stacked_pred, axis = 0)


if __name__ == "__main__":
    u = np.arange(8).reshape((2,2,2))
    f = augmented_predict(u)
    f.augment_inputs()
    n_stacks = f.get_n_aug_stacks()
    #print n_stacks
    x = f.get_augmented_patch()
    print x
    print f.reduce_predictions(x)
