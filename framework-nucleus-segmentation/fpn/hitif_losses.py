from __future__ import print_function
from keras import backend as K
from keras.backend import binary_crossentropy
import keras
import tensorflow as tf


def focal_loss(y_true, y_pred, gamma=2., alpha=.25):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    focal_loss_fixed = -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed
 

def jaccard_coef(y_true, y_pred, smooth=1e-3):
    intersection = K.sum(y_true * y_pred, axis=[-0, -1, 2])
    sum_ = K.sum(y_true + y_pred, axis=[-0, -1, 2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred, smooth=1e-3):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    intersection = K.sum(y_true * y_pred_pos, axis=[-0, -1, 2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[-0, -1, 2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def jaccard_coef_loss(y_true, y_pred):
    return -K.log(jaccard_coef(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)

def dice_coef_batch(y_true, y_pred, smooth=1e-3):
    intersection = K.sum(y_true * y_pred, axis=[-0, -1, 2])
    sum_ = K.sum(y_true + y_pred, axis=[-0, -1, 2])
    dice = ((2.0*intersection) + smooth) / (sum_ + smooth)
    return K.mean(dice)


def dice_coef(y_true, y_pred, smooth=1e-3):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice_smooth = ((2. * intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return (dice_smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.0-dice_coef(y_true, y_pred)

def dice_coef_batch_loss(y_true, y_pred):
    return 1.0-dice_coef_batch(y_true, y_pred)

def binary_crossentropy(y, p):
    return K.mean(K.binary_crossentropy(y, p))

def dice_coef_loss_bce(y_true, y_pred, dice=0.5, bce=0.5):
    return binary_crossentropy(y_true, y_pred) * bce + dice_coef_batch_loss(y_true, y_pred) * dice

def double_head_loss(y_true, y_pred):
    mask_loss = dice_coef_loss_bce(y_true[..., 0], y_pred[..., 0])
    contour_loss = dice_coef_loss_bce(y_true[..., 1], y_pred[..., 1])
    return mask_loss + contour_loss
