import tensorflow as tf
import numpy
import os

path = os.path.dirname(os.path.abspath(__file__)) + "/"
param_dict = {} #numpy.load( path + 'vgg16.npy' ).item()

def fc( bottom, num_out, name, trainable = True ):
  shape = bottom.get_shape().as_list()
  if len(shape) > 2:
    bottom = tf.contrib.layers.flatten( bottom )

  num_in = bottom.get_shape().as_list()[-1]

  W = tf.get_variable( name + "/weights", shape = [num_in, num_out], initializer = tf.contrib.layers.xavier_initializer(), trainable = trainable )
  b = tf.get_variable( name + "/biases", shape = [num_out], initializer = tf.zeros_initializer(), trainable = trainable )
  return tf.matmul( bottom, W ) + b

def lstm( bottom, T, dim, name ):

  batch_size = (tf.shape(bottom)[0])/T
  dim_in     = bottom.get_shape().as_list()[-1] # We need this value as a constant

  bottom = tf.reshape( bottom, [batch_size, T, dim_in] )

  with tf.variable_scope( "name" ) as scope:
    lstm = tf.contrib.rnn.LSTMCell( dim )
    state = lstm.zero_state( batch_size, "float" )    

    splits = tf.split( bottom, T, axis=1 )

    for t in range( T ):
      if t > 0:
        scope.reuse_variables()

      out, state = lstm( tf.reshape( splits[t], [batch_size, dim_in] ), state )
  return tf.reshape( out, [batch_size, dim] )


def conv2d( bottom, ksize, name, stride = [1,1,1,1] ):
  num_c = bottom.get_shape().as_list()[-1]
  kernel = [ksize[0], ksize[1], num_c, ksize[2]]

  if name in param_dict.keys():
    init = tf.constant_initializer(value=param_dict[name][0], dtype=tf.float32)
    b_init = tf.constant_initializer( value = param_dict[name][1], dtype=tf.float32)
  else:
    init = tf.contrib.layers.xavier_initializer_conv2d()
    b_init = tf.zeros_initializer()

  w = tf.get_variable( name + "/weights", shape = kernel, initializer = init )
  b = tf.get_variable( name + "/biases", shape = [ksize[2]], initializer = b_init )

  return tf.nn.conv2d( bottom, w, strides = stride, padding = 'SAME' ) + b

def conv2d_transpose( bottom, ksize, stride, name ):
  shape = bottom.get_shape().as_list()
  num_c = shape[-1]
  kernel = [ksize[0], ksize[1], ksize[2], num_c]
  output_shape = [shape[0], shape[1]*2, shape[2]*2, ksize[2]]

  w = tf.get_variable( name + "/weights", shape = kernel, initializer = tf.contrib.layers.xavier_initializer_conv2d() )
  b = tf.get_variable( name + "/biases", shape = [ksize[2]], initializer = tf.zeros_initializer() )
  return tf.nn.conv2d_transpose( bottom, w, output_shape, strides = stride ) + b

class BatchNorm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)
