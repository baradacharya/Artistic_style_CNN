# -*- coding: utf-8 -*-
"""
Created on Fri May 19 11:25:36 2017

@author: Barada
"""

from __future__ import print_function
from keras.preprocessing.image import load_img, img_to_array
from scipy.misc import imsave
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse
from PIL import Image as PImage
import keras

from scipy import misc
import tensorflow as tf

from keras.applications import vgg19
from keras import backend as K


parser = argparse.ArgumentParser(description = 'A Neural Algorithm of Artistic Style')
parser.add_argument('style_image', metavar='ref', type=str,
                    help='Path to the style reference image.')
args = parser.parse_args()
Content_image = "image.jpg"
style_image = "style_" + args.style_image + ".jpg" 
result_prefix = "result"



# Declare Hyper parameters
iter = 4
total_variation_weight = 4
style_weight = 25
content_weight = 0.5

#n_row  <-> height, n_col <-> width
#want to maintain same aspect ratio as original image n_row/n_col = height/width
width, height = load_img(Content_image).size
nrows = 400
ncols = int(width * nrows / height)

###################
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(nrows,ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_image(x):
    if K.image_dim_ordering() == 'th':
        x = x.reshape((3, nrows, ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((nrows, ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x
################

Content_reference_image = K.variable(preprocess_image(Content_image))
style_reference_image = K.variable(preprocess_image(style_image))
print('Images loaded....')
#New generated image
if K.image_dim_ordering() == 'th':
    new_image = K.placeholder((1, 3, nrows, ncols))
else:
    new_image = K.placeholder((1, nrows, ncols, 3))
    

                         
print('New Random image created....')
# Create a single Keras tensor by combining our images
input_tensor = K.concatenate([Content_reference_image,style_reference_image,new_image], axis=0)


#VGG16 network with pre-trained ImageNet weights
model = vgg19.VGG19(input_tensor=input_tensor,weights='imagenet', include_top=False)
#model.summary()
print('Model loaded.')

# Create a dictionary whose "key" will be layername and contain will be layer dim.
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

###########################
# compute the neural style loss
# the gram matrix of an image tensor (feature-wise outer product)

"""
we built a style representation that computes the correlations between the different filter responses, These feature correlations 
are given by the Gram matrix where Gl(ij) is the inner product between the vectorised feature map i and j in layer l:

We can think gram matrix as a covariance matrix of various filters. where we see relation between filers how they will 
fire together or which filters are same. so it helped us in determinning the style. 
"""

def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_dim_ordering() ==  'th':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram
"""
it consists in a sum of L2 distances between the Gram matrices of the representations of
the base image and the style reference image, extracted from different layers of a convnet.
The general idea is to capture color/texture information at different spatial scales
"""
def mean_style(x):
    #print(x.shape)
    mean_x =K.mean(x,axis=(0,1))
    #print(mean_x.shape)
    return mean_x

def style_loss(style_img, new_img):
    assert K.ndim(style_img) == 3
    assert K.ndim(new_img) == 3
    #mean_style_ = mean_style(style)
    #mean_combinaion_ = mean_style(combination)
    
    S = gram_matrix(style_img)
    C = gram_matrix(new_img)
    a,b,c = new_img.shape
    #print(int(c))
    channels = 3 #int(c) #(3) doubt change to style.shape as per paper it is the no of  distinct filters
    size = nrows * ncols
    
    #mean_loss = K.sum(K.square(mean_style_ - mean_combinaion_)) / (4. * (channels ** 2) * (size ** 2))
    gramm_matrix_loss = K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))
    style_loss_ = gramm_matrix_loss #+ mean_loss
    return style_loss_

# an auxiliary loss function designed to maintain the "content" of the base image in the generated image
"""
The content loss is a L2 distance between the features of the base image (extracted from a deep layer) and the features of 
the combination image, keeping the generated image close enough to the original one.

The concept behind content loss is if at any level for same picture neuron wll fire same output so by comparing output we can
come to know how far are they from each other.
"""
def content_loss(base, combination):
    return K.sum(K.square(combination - base))
"""
Total variation loss, designed to keep the generated image locally coherent
The total variation loss imposes local spatial continuity between the pixels of the new image, giving it visual coherence.
""" 

def total_variation_loss(x):
    assert K.ndim(x) == 4
    if K.image_dim_ordering() == 'channels_first':
        
        a = K.square(x[:, :, :nrows - 1, :ncols - 1] - x[:, :, 1:, :ncols - 1])
        b = K.square(x[:, :, :nrows - 1, :ncols - 1] - x[:, :, :nrows - 1, 1:])
        
    else:
        #calculate row wise continuity by taking squae of to consecutive pixel
        a = K.square(x[:, :nrows - 1, :ncols - 1, :] - x[:, 1:, :ncols - 1, :])
        #print(a.shape)
        #calculate col wise continuity by taking squae of to consecutive pixel
        b = K.square(x[:, :nrows - 1, :ncols - 1, :] - x[:, :nrows - 1, 1:, :])
        #print(b.shape)
    return K.sum(K.pow(a + b, 1.25))
# combine these loss functions into a single scalar

loss = K.variable(0.)

layer_features = outputs_dict['block5_conv2']

#content loss between base_image and newly created image combination_features
content_image_features = layer_features[0, :, :, :]#base image
new_features = layer_features[2, :, :, :] #new image
loss += content_weight * content_loss(content_image_features,new_features)

#calculate style Loss
#Declare Layers upon which we will learn styles
feature_layers = ['block1_conv1', 'block2_conv1','block3_conv1', 'block4_conv1', 'block5_conv1']

for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :] #feature 
    new_image_features = layer_features[2, :, :, :]#new feature
    styleloss = style_loss(style_reference_features, new_image_features)
    loss += (style_weight / len(feature_layers)) * styleloss
loss += total_variation_weight * total_variation_loss(new_image)


# get the gradients of the generated image wrt the loss
"""
see https://github.com/fchollet/keras/blob/master/keras/backend/tensorflow_backend.py
def gradients(loss, variables):
    Returns the gradients of `variables` w.r.t. `loss`.
    # Arguments
        loss: Scalar tensor to minimize.
        variables: List of variables.
    # Returns
        A gradients tensor.
        
        https://www.tensorflow.org/api_docs/python/tf/gradients
"""
#how loss changes wrt change in new_image pixel value
grads = K.gradients(loss, new_image)
#output will contain loss and grad
outputs = [loss]
if isinstance(grads, (list, tuple)):
    #code comes here
    outputs += grads
else:
    outputs.append(grads)
"""    
def function(inputs, outputs, updates=None, **kwargs):
#K.function takes the input and output tensors as list so that you can create a function from many input to many output. 
"""    
f_outputs = K.function([new_image], outputs)
#f_outputs return loss grads and take input image


def eval_loss_and_grads(x):
    if K.image_dim_ordering() == 'th':
        x = x.reshape((1, 3, nrows, ncols))
    else:
        x = x.reshape((1, nrows, ncols, 3))
    # function gives input image x an returns outs which contains loss and grad value
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()
# run scipy-based optimization (L-BFGS) over the pixels of the generated image
# so as to minimize the neural style loss
"""
if K.image_dim_ordering() == 'th':
    x = np.random.uniform(0, 255, (1, 3, nrows, ncols)) - 128.
else:
    x = np.random.uniform(0, 255, (1, nrows, ncols, 3)) - 128.
"""                 
x = preprocess_image(Content_image)
img = deprocess_image(x.copy())
for i in range(iter):
    print('Start of iteration', i)
    start_time = time.time()
    """
    #https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
    #min_l_bfgs_b(func, x0, fprime=None, maxfun=15000) 
    #func : Function to minimise.
    #x0 : ndarray Initial guess.
    #fprime : callable fprime(x,*args) The gradient of func.
    """
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    # save current generated image
    img = deprocess_image(x.copy())

    end_time = time.time()    
    print('Iteration %d completed in %ds' % (i, end_time - start_time))       
fname = result_prefix + '.png'
imsave(fname, img)  
print('Image saved as', fname)          