import tensorflow as tf
from tensorflow import keras

"""
Build object tracking net work with Resnet backbone

set bias of last layer to -3.2 according to website below:
Focal Loss Trick: https://leimao.github.io/blog/Focal-Loss-Explained/
"""

class ResNet_BottleNeck(keras.layers.Layer):
  def __init__(self, filters, strides, decoder=False, **conv_kwargs):
    super(ResNet_BottleNeck, self).__init__()
    self.bn_1 = keras.layers.BatchNormalization()
    self.active_1 = keras.layers.Activation("relu")
    self.conv_1 = keras.layers.Conv2D(filters, kernel_size=(1, 1), strides=1, padding='same', data_format='channels_first', **conv_kwargs)

    self.bn_2 = keras.layers.BatchNormalization()
    self.active_2 = keras.layers.Activation("relu")
    self.conv_2 = keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=strides, padding='same', data_format='channels_first', **conv_kwargs)

    self.bn_3 = keras.layers.BatchNormalization()
    self.active_3 = keras.layers.Activation("relu")

    if not decoder:
      self.conv_3 = keras.layers.Conv2D(2*filters, (1, 1), strides=1, padding="same", data_format='channels_first', **conv_kwargs)
    else:
      self.conv_3 = keras.layers.Conv2D(filters, (1, 1), strides=1, padding="same", data_format='channels_first', **conv_kwargs)

    if strides==2:
      self.short_cut = keras.Sequential([
        keras.layers.AveragePooling2D((2, 2), strides=strides, padding='same', data_format='channels_first'),
        keras.layers.Conv2D(2*filters, (1, 1), strides=1, padding='same', data_format='channels_first'),
        keras.layers.BatchNormalization()
      ])

    elif not decoder:
      self.short_cut = lambda x: x
    else:
      self.short_cut = keras.layers.Conv2D(filters, (1,1), strides=1, padding='same', data_format='channels_first')

  def call(self, inputs):
    x = self.bn_1(inputs)
    x = self.active_1(x)
    x = self.conv_1(x)

    x = self.bn_2(x)
    x = self.active_2(x)
    x = self.conv_2(x)

    x = self.bn_3(x)
    x = self.active_3(x)
    x = self.conv_3(x)

    short_cut = self.short_cut(inputs)
    outputs = keras.layers.add([x, short_cut])
    return outputs

class ResNet_Transpose(keras.layers.Layer):
  def __init__(self, filters, strides, **conv_kwargs):
    super(ResNet_Transpose, self).__init__()
    self.bn_1 = keras.layers.BatchNormalization()
    self.active_1 = keras.layers.Activation("relu")
    self.conv_1 = keras.layers.Conv2D(filters, kernel_size=(1, 1), strides=1, padding='same', data_format='channels_first', **conv_kwargs)

    self.bn_2 = keras.layers.BatchNormalization()
    self.active_2 = keras.layers.Activation("relu")
    self.conv_t = keras.layers.Conv2DTranspose(filters, kernel_size=(3, 3), strides=strides, padding='same', data_format='channels_first', output_padding=1, **conv_kwargs)

    self.bn_3 = keras.layers.BatchNormalization()
    self.active_3 = keras.layers.Activation("relu")
    self.conv_3 = keras.layers.Conv2D(filters, (1, 1), strides=1, padding="same", data_format='channels_first', **conv_kwargs)

    self.short_cut = keras.Sequential([
        keras.layers.UpSampling2D((2, 2), interpolation='bilinear', data_format='channels_first'),
        keras.layers.Conv2D(filters, (1,1), strides=1, padding='same', data_format='channels_first'),
        keras.layers.BatchNormalization()
      ])

  def call(self, inputs):
    x = self.bn_1(inputs)
    x = self.active_1(x)
    x = self.conv_1(x)

    x = self.bn_2(x)
    x = self.active_2(x)
    x = self.conv_t(x)

    x = self.bn_3(x)
    x = self.active_3(x)
    x = self.conv_3(x)

    short_cut = self.short_cut(inputs)
    outputs = keras.layers.add([x, short_cut])
    return outputs

class ResNet_Track(keras.models.Model):
  def __init__(self, input_shape, structure=[3, 3, 4, 3], num_filters=[16, 32, 64, 128]):
    super(ResNet_Track, self).__init__()
    # Initial
    self.inital = keras.Sequential([
                  keras.layers.Conv2D(64, (3,3), padding='same', data_format='channels_first', input_shape=input_shape),
                  keras.layers.BatchNormalization(),
                  keras.layers.Activation("relu"),
                  keras.layers.Conv2D(64, (3,3), padding='same', data_format='channels_first'),
                  keras.layers.BatchNormalization(),
                  keras.layers.Activation("relu"),
    ])

    # Encoder
    self.block_1 = self.build_block(structure[0], num_filters[0], strides=2)
    self.block_2 = self.build_block(structure[1], num_filters[1], strides=2)
    self.block_3 = self.build_block(structure[2], num_filters[2], strides=2)
    self.block_4 = self.build_block(structure[3], num_filters[3], strides=2)

    # Decoder
    self.conv_t1 = ResNet_Transpose(num_filters[3], strides=2)
    self.conv_d1 = self.build_block((structure[2]-1), num_filters[3], strides=1, decoder=True)

    self.conv_t2 = ResNet_Transpose(num_filters[2], strides=2)
    self.conv_d2 = self.build_block((structure[1]-1), num_filters[2], strides=1, decoder=True)

    self.conv_t3 = ResNet_Transpose(num_filters[1], strides=2)
    self.conv_d3 = self.build_block((structure[0]-1), num_filters[1], strides=1, decoder=True)

    self.conv_t4 = ResNet_Transpose(num_filters[0], strides=2)
    
    # Last
    self.last = keras.Sequential([
                keras.layers.Conv2D(64, (3,3), padding='same', data_format='channels_first'),
                keras.layers.BatchNormalization(),
                keras.layers.Activation("relu"),
                keras.layers.Conv2D(64, (3,3), padding='same', data_format='channels_first'),
                keras.layers.BatchNormalization(),
                keras.layers.Activation("relu"),
                keras.layers.Conv2D(256, (3,3), padding='same', data_format='channels_first', bias_initializer=keras.initializers.constant(-3.2)),
                keras.layers.BatchNormalization(),
                keras.layers.Activation("relu"),
                keras.layers.Activation("softmax")
    ])
  
  def build_block(self, num_blocks, filters, strides, **conv_kwargs):
    block = keras.Sequential()
    block.add(ResNet_BottleNeck(filters, strides=strides, **conv_kwargs))
    for _ in range(num_blocks-1):
      block.add(ResNet_BottleNeck(filters, strides=1, **conv_kwargs))

    return block

  def call(self, inputs):
    x = self.inital(inputs)

    e1 = self.block_1(x)
    e2 = self.block_2(e1)
    e3 = self.block_3(e2)
    e4 = self.block_4(e3)

    d_u3 = self.conv_t1(e4)
    d_u3 = tf.concat([d_u3, e3], axis=1)
    d_c3 = self.conv_d1(d_u3)

    d_u2 = self.conv_t2(d_c3)
    d_u2 = tf.concat([d_u2, e2], axis=1)
    d_c2 = self.conv_d2(d_u2)

    d_u1 = self.conv_t3(d_c2)
    d_u1 = tf.concat([d_u1, e1], axis=1)
    d_c1 = self.conv_d3(d_u1)

    outputs = self.conv_t4(d_c1)
    outputs = self.last(outputs)
    outputs = tf.reduce_max(outputs, axis=1)
    outputs = tf.expand_dims(outputs, axis=1)
    return outputs

if __name__=='__main__':
  model = ResNet_Track(input_shape=(3, 288, 512))
  model.build(input_shape=(None, 3, 288, 512))
  model.summary()