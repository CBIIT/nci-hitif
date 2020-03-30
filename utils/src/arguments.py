import argparse

def get_unet_parser():

    parser = argparse.ArgumentParser(description="spot learner")
    parser.add_argument('images',  help="The 2d numpy array image stack or 128 * 128")
    parser.add_argument('masks',  help="The 2d numpy array mask (16bits) stack or 128 * 128")
    parser.add_argument('--nlayers', default=4, type = int, dest='nlayers', help="The number of layer in the forward path ")
    parser.add_argument('--num_filters', default=32, type = int, dest='num_filters', help="The number of convolution filters in the first layer")
    parser.add_argument('--conv_size', default='3', type = int, dest='conv_size', help="The convolution filter size.")
    parser.add_argument('--dropout', default=None, type = float, dest='dropout', help="Include a droupout layer with a specific dropout value.")
    parser.add_argument('--activation', default='relu',dest='activation', help="Activation function.")
    parser.add_argument('--augmentation', default=1,  type = float, dest='augmentation', help="Augmentation factor for the training set.")
    parser.add_argument('--initialize', default=None, dest='initialize', help="Numpy array for weights initialization.")
    parser.add_argument('--normalize_mask', action='store_true', dest='normalize_mask', help="Normalize the mask in case of uint8 to 0-1 by dividing by 255.")
    parser.add_argument('--predict', action='store_true', dest='predict', help="Use the model passed in initialize to perform segmentation")
    parser.add_argument('--loss_func', default='dice', dest='loss_func', help="Keras supported loss function, or 'dice'. ")
    parser.add_argument('--last_act', default='sigmoid', dest='last_act', help="The activation function for the last layer.")
    parser.add_argument('--batch_norm', default=False, action = "store_true", dest='batch_norm', help="Enable batch normalization")
    parser.add_argument('--lr', default='1e-5', type = float, dest='lr', help="Initial learning rate for the optimizer")
    parser.add_argument('--rotate', default=False, action = "store_true", dest='rotate', help="")

    return parser 
