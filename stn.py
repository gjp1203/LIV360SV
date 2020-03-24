from spatial_transformer import transformer
from utils import img2array, array2img
import tensorflow.contrib.slim as slim
#from IPython.display import Image  
from functools import reduce
from random import shuffle
import tensorflow as tf
from PIL import Image
#import scipy.misc
from pylab import *
import numpy as np
import pylab as pl
import imageio
#import random
import glob
import time
import cv2

# Dimensions of the output image
DIMS = (500, 500)

"""def loadImages(path, extension=".png"):
    '''
    Returns np array of all files with the specified extension.
    :param string: path to folder 
    :param string: file type
    :return numpy array: Samples * height * width * channels
    '''
    filenames = glob.glob(path+"*"+extension)
    imgs = []
    print(len(filenames))
    for filename in filenames: 
        imgs.append(img2array(filename, DIMS, expand=True))
    return np.concatenate(imgs, axis=0), filenames 
"""

def loadImages(path, batch_size, extension=".png"):
    """
    Returns np array of all files with the specified extension.
    :param string: path to folder 
    :param string: file type
    :return numpy array: Samples * height * width * channels
    """
    filenames = glob.glob(path+"*"+extension)
    imgs = []
    shuffle(filenames)
    for filename in filenames[:batch_size]: 
        imgs.append(img2array(filename, DIMS, expand=True))
    return np.concatenate(imgs, axis=0)

def batch_norm(x, name="batch_norm"):
    """
    Batch normalization layer
    """
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)

def instance_norm(input, name="instance_norm"):
    """
    Instance normalization layer
    """
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    """
    Linear layer
    """
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [input_.get_shape()[-1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
        
def conv2d(input, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d"):
    """
    2D conv layer layer
    """
    with tf.variable_scope(name):
        return slim.conv2d(input, output_dim, ks, s, padding=padding, activation_fn=None,
                            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                            biases_initializer=None)        

def lrelu(x, leak=0.2, name="lrelu"):
    """
    Leaky relu
    """
    return tf.maximum(x, leak*x)  


def convert(target_image, input_img, originals, batch_size=32, iterations=300):
    """
    Method for training a convnet to learn transformation parameters for
    a target transformation. The parameters are supplied to the STN.
    :params np.array: Target image, in this case we just have one (square) 
    :params np.array: Input masks, shapes to be converted into the target image
    :params list: Filename
    :params np.array: Files to be converted.
    """
    #S, H, W, C = input_img.shape
    H, W, C = (DIMS[0], DIMS[1], 3)
    x = tf.placeholder(tf.float32, [None, H, W, C])
    target_batch = [target_image[0] for i in range (batch_size)]
    food_x = tf.placeholder(tf.float32, [None, H, W, C])
    target = tf.placeholder(tf.float32, [None, H, W, C])
    with tf.variable_scope('spatial_transformer'):
        theta = np.array([1.1, .0, .0, .0, 1.1, .0]).astype('float32')
        
        # Conv Layers
        h0 = lrelu(conv2d(target, 64, name='t_h0_conv'))
        
        # h0 is (128 x 128 x self.df_dim)
        #h1 = lrelu(instance_norm(conv2d(h0, 64*2, name='t_h1_conv'), 't_bn1'))
        h1 = lrelu(instance_norm(conv2d(h0, 64, name='t_h1_conv'), 't_bn1'))

        # h1 is (64 x 64 x self.df_dim*2)
        #h2 = lrelu(instance_norm(conv2d(h1, 64*4, name='t_h2_conv'), 't_bn2'))
        h2 = lrelu(instance_norm(conv2d(h1, 64, name='t_h2_conv'), 't_bn2'))

        # Fully connected layer:
        shape = h2.get_shape().as_list()
        h2_flat = tf.reshape(h2, [-1, reduce(lambda x, y: x * y, shape[1:])])
        l1 = linear(h2_flat, 512, scope="l1")
        W_loc = tf.Variable(tf.zeros([l1.get_shape()[-1], 6]), name='W_fc1')
        b_loc = tf.Variable(initial_value=theta, name='b_loc')

        # tie everything together
        fc_loc = tf.matmul(l1, W_loc) + b_loc
        h_trans = transformer(x, fc_loc, [H, W])
        f_trans = transformer(food_x, fc_loc, [H, W])
        loss = tf.losses.huber_loss(target, h_trans)
        optim = tf.train.AdamOptimizer(1e-07).minimize(loss)
           
    
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        sess.run(tf.global_variables_initializer())
        losses = []
        d = {'losses':[]}
        #for i in range (0, S-batch_size, batch_size):
        for i in range(iterations):
            _, fc_o, l, w, b, y = sess.run([optim, fc_loc, loss, W_loc, b_loc, h_trans], feed_dict={x:loadImages("./mask/", batch_size), target:target_batch})
            if i%10 == 0:
                print(l)
            if l < 0.005:
                break

        # Transform and save the cropped images:
        filenames = glob.glob(originals+"*.png")
        for filename in filenames: 
            [y] = sess.run([f_trans], feed_dict={food_x:img2array(filename, DIMS, expand=True), target:[target_batch[0]]})
            imageio.imwrite("Results/"+filename.replace("./masked/", ""), (y[0]*255).astype(np.uint8))
        sess.close()
    tf.reset_default_graph()

print("Loading images")       
target = img2array("./target.jpg", DIMS, expand=True)
print("Building net")       
convert(target, "./mask/", "./masked/")
