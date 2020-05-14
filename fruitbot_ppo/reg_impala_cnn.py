import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()



def build_reg_impala_cnn(unscaled_images, depths=[16,32,32], **conv_kwargs):
    """
    Model used in the paper "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
    """

    layer_num = 0

    def get_layer_num_str():
        nonlocal layer_num
        num_str = str(layer_num)
        layer_num += 1
        return num_str

    def conv_layer(out, depth):
        return tf.layers.conv2d(out, depth, 3, padding='same', name='layer_' + get_layer_num_str())

    def residual_block(inputs):
        depth = inputs.get_shape()[-1].value

        out = tf.nn.leaky_relu(inputs)

        out = conv_layer(out, depth)
        # Potential dropout here
        out = tf.nn.leaky_relu(out)
        out = conv_layer(out, depth)
        return out + inputs

    def conv_sequence(inputs, depth):
        out = conv_layer(inputs, depth)
        out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
        out = residual_block(out)
        # Dropout or Batchnorm here?
        out = tf.layers.batch_normalization(out)
        out = tf.layers.dropout(out, rate = .5)
        out = residual_block(out)
        return out

    out = tf.cast(unscaled_images, tf.float32) / 255.

    for depth in depths:
        out = conv_sequence(out, depth)

    out = tf.layers.flatten(out)
    out = tf.nn.leaky_relu(out)
    out = tf.layers.dense(out, 256, activation=tf.nn.leaky_relu, name='layer_' + get_layer_num_str())

    return out



def build_reg_impala_cnn_verbose(unscaled_images, depths=[16,32,32], **conv_kwargs):
    """
    Model used in the paper "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
    """

    layer_num = 0

    def get_layer_num_str():
        nonlocal layer_num
        num_str = str(layer_num)
        layer_num += 1
        return num_str

    def conv_layer(out, depth):
        name='layer_' + get_layer_num_str()
        print(name)
        layer = tf.layers.conv2d(out, depth, 3, padding='same', name=name)
        print(tf.shape(layer))
        return layer

    def residual_block(inputs):
        depth = inputs.get_shape()[-1].value

        out = tf.nn.leaky_relu(inputs)

        out = conv_layer(out, depth)
        # Potential dropout here
        out = tf.nn.leaky_relu(out)
        out = conv_layer(out, depth)
        return out + inputs

    def conv_sequence(inputs, depth):
        out = conv_layer(inputs, depth)
        out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
        print('maxpool')
        print(tf.shape(out))
        out = residual_block(out)
        # Dropout or Batchnorm here?
        out = tf.layers.batch_normalization(out)
        out = tf.layers.dropout(out, rate = .5)
        out = residual_block(out)
        return out

    out = tf.cast(unscaled_images, tf.float32) / 255.

    for depth in depths:
        out = conv_sequence(out, depth)

    out = tf.layers.flatten(out)
    print('flattened')
    print(tf.shape(out))
    out = tf.nn.leaky_relu(out)
    out = tf.layers.dense(out, 256, activation=tf.nn.leaky_relu, name='layer_' + get_layer_num_str())

    return out