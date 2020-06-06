import tensorflow.keras as K
import inspect


def get_network(network_name):
    """Compile network by name"""

    networks = __import__(inspect.getmodulename(__file__))

    try:
        network = getattr(networks, network_name)
    except AttributeError:
        print('Error: {} is no valid network name.'.format(network_name))

    return network


def fnn(input_shape=(116, 80, 1)):

    inputs = K.layers.Input(shape=input_shape)
    batchNorm = K.layers.BatchNormalization(momentum=0.1)(inputs)
    flatten = K.layers.Flatten()(batchNorm)
    # drop1 = K.layers.Dropout(0.5)(flatten)
    dense1 = K.layers.Dense(128, activation='sigmoid')(flatten)
    # drop2 = K.layers.Dropout(0.5)(dense1)
    outputs = K.layers.Dense(1, activation='sigmoid')(dense1)

    model = K.models.Model(inputs=inputs, outputs=outputs)

    return model


def cnn_ismir2014(input_shape=(116, 80, 1)):
    """Creates the CNN of ismir2014.

    Parameters
    ----------
    input_shape : tuple
        Tuple with (n_t, n_f, channels)

    Returns
    -------
    model : keras.model
        Keras model of the CNN.
    """

    inputs = K.layers.Input(shape=input_shape)
    batchNorm = K.layers.BatchNormalization(momentum=0.1, axis=2)(inputs)  # batch norm over time
    conv1 = K.layers.Conv2D(16, (8, 6),
                            padding='valid',
                            strides=(1, 1),
                            data_format='channels_last',
                            activation='tanh')(batchNorm)
    pool1 = K.layers.MaxPooling2D(pool_size=(3, 6),
                                  strides=(3, 6),
                                  padding='valid',
                                  data_format='channels_last')(conv1)
    conv2 = K.layers.Conv2D(32, (6, 3),
                            padding='valid',
                            strides=(1, 1),
                            data_format='channels_last',
                            activation='tanh')(pool1)

    flatten = K.layers.Flatten()(conv2)
    drop1 = K.layers.Dropout(0.5)(flatten)
    dense1 = K.layers.Dense(128, activation='sigmoid')(drop1)
    drop2 = K.layers.Dropout(0.5)(dense1)
    outputs = K.layers.Dense(1, activation='sigmoid')(drop2)

    model = K.models.Model(inputs=inputs, outputs=outputs)

    return model


def cnn_mirex2014_tanh(input_shape=(116, 80, 1)):
    """Same as ismir2014, except double the number of filters in convolutional layers.

    Parameters
    ----------
    input_shape : tuple
        Tuple with (n_t, n_f, channels)

    Returns
    -------
    model : keras.model
        Keras model of the CNN.
    """

    inputs = K.layers.Input(shape=input_shape)
    batchNorm = K.layers.BatchNormalization(momentum=0.1, axis=2)(inputs)  # batch norm over time
    conv1 = K.layers.Conv2D(32, (8, 6),
                            padding='valid',
                            strides=(1, 1),
                            data_format='channels_last',
                            activation='tanh')(batchNorm)
    pool1 = K.layers.MaxPooling2D(pool_size=(3, 6),
                                  strides=(3, 6),
                                  padding='valid',
                                  data_format='channels_last')(conv1)
    conv2 = K.layers.Conv2D(64, (6, 3),
                            padding='valid',
                            strides=(1, 1),
                            data_format='channels_last',
                            activation='tanh')(pool1)

    flatten = K.layers.Flatten()(conv2)
    drop1 = K.layers.Dropout(0.5)(flatten)
    dense1 = K.layers.Dense(128, activation='sigmoid')(drop1)
    drop2 = K.layers.Dropout(0.5)(dense1)
    outputs = K.layers.Dense(1, activation='sigmoid')(drop2)

    model = K.models.Model(inputs=inputs, outputs=outputs)

    return model


def cnn_mirex2014_relu(input_shape=(116, 80, 1)):
    """Creates the CNN of ismir2014 with strides(3, 6) in the first pooling layer to reduce parameters.

    Parameters
    ----------
    input_shape : tuple
        Tuple with (n_t, n_f, channels)

    Returns
    -------
    model : keras.model
        Keras model of the CNN.
    """

    inputs = K.layers.Input(shape=input_shape)
    batchNorm = K.layers.BatchNormalization(momentum=0.1, axis=2)(inputs)  # batch norm over time
    conv1 = K.layers.Conv2D(32, (8, 6),
                            padding='valid',
                            strides=(1, 1),
                            data_format='channels_last',
                            activation='relu')(batchNorm)
    pool1 = K.layers.MaxPooling2D(pool_size=(3, 6),
                                  strides=(3, 6),
                                  padding='valid',
                                  data_format='channels_last')(conv1)
    conv2 = K.layers.Conv2D(64, (6, 3),
                            padding='valid',
                            strides=(1, 1),
                            data_format='channels_last',
                            activation='relu')(pool1)

    flatten = K.layers.Flatten()(conv2)
    drop1 = K.layers.Dropout(0.5)(flatten)
    dense1 = K.layers.Dense(128, activation='relu')(drop1)
    drop2 = K.layers.Dropout(0.5)(dense1)
    outputs = K.layers.Dense(1, activation='sigmoid')(drop2)

    model = K.models.Model(inputs=inputs, outputs=outputs)

    return model


def cnn_mirex2014_relu_bn(input_shape=(116, 80, 1)):
    """Creates the CNN of ismir2014 with strides(3, 6) in the first pooling layer to reduce parameters.

    Parameters
    ----------
    input_shape : tuple
        Tuple with (n_t, n_f, channels)

    Returns
    -------
    model : keras.model
        Keras model of the CNN.
    """

    inputs = K.layers.Input(shape=input_shape)
    bn_input = K.layers.BatchNormalization(momentum=0.1, axis=2)(inputs)  # batch norm over time
    conv1 = K.layers.Conv2D(32, (8, 6),
                            padding='valid',
                            strides=(1, 1),
                            data_format='channels_last',
                            activation='relu')(bn_input)
    pool1 = K.layers.MaxPooling2D(pool_size=(3, 6),
                                  strides=(3, 6),
                                  padding='valid',
                                  data_format='channels_last')(conv1)
    bn1 = K.layers.BatchNormalization(momentum=0.1)(pool1)  # batch norm over time

    conv2 = K.layers.Conv2D(64, (6, 3),
                            padding='valid',
                            strides=(1, 1),
                            data_format='channels_last',
                            activation='relu')(bn1)
    bn2 = K.layers.BatchNormalization(momentum=0.1)(conv2)  # batch norm over time

    flatten = K.layers.Flatten()(bn2)
    drop1 = K.layers.Dropout(0.5)(flatten)
    dense1 = K.layers.Dense(128, activation='relu')(drop1)
    drop2 = K.layers.Dropout(0.5)(dense1)
    outputs = K.layers.Dense(1, activation='sigmoid')(drop2)

    model = K.models.Model(inputs=inputs, outputs=outputs)

    return model


def cnn_mirex2014_lrelu_bn(input_shape=(116, 80, 1)):
    """Creates the CNN of ismir2014 with strides(3, 6) in the first pooling layer to reduce parameters.

    Parameters
    ----------
    input_shape : tuple
        Tuple with (n_t, n_f, channels)

    Returns
    -------
    model : keras.model
        Keras model of the CNN.
    """

    inputs = K.layers.Input(shape=input_shape)
    bn_input = K.layers.BatchNormalization(momentum=0.1, axis=2)(inputs)  # batch norm over time
    conv1 = K.layers.Conv2D(32, (8, 6),
                            padding='valid',
                            strides=(1, 1),
                            data_format='channels_last',
                            activation='linear')(bn_input)
    act1 = K.layers.LeakyReLU()(conv1)
    pool1 = K.layers.MaxPooling2D(pool_size=(3, 6),
                                  strides=(3, 6),
                                  padding='valid',
                                  data_format='channels_last')(act1)
    bn1 = K.layers.BatchNormalization(momentum=0.1)(pool1)  # batch norm over time

    conv2 = K.layers.Conv2D(64, (6, 3),
                            padding='valid',
                            strides=(1, 1),
                            data_format='channels_last',
                            activation='linear')(bn1)
    act2 = K.layers.LeakyReLU()(conv2)
    bn2 = K.layers.BatchNormalization(momentum=0.1)(act2)  # batch norm over time

    flatten = K.layers.Flatten()(bn2)
    drop1 = K.layers.Dropout(0.5)(flatten)
    dense1 = K.layers.Dense(128, activation='linear')(drop1)
    act3 = K.layers.LeakyReLU()(dense1)
    drop2 = K.layers.Dropout(0.5)(act3)
    outputs = K.layers.Dense(1, activation='sigmoid')(drop2)

    model = K.models.Model(inputs=inputs, outputs=outputs)

    return model


def cnn_vggish(input_shape=(116, 80, 1)):
    """Creates a VGGish CNN.

    Parameters
    ----------
    input_shape : tuple
        Tuple with (n_t, n_f, channels)

    Returns
    -------
    model : keras.model
        Keras model of the CNN.
    """

    inputs = K.layers.Input(shape=input_shape)
    batchNorm = K.layers.BatchNormalization(momentum=0.1, axis=2)(inputs)  # batch norm over time

    conv1 = K.layers.Conv2D(32, (5, 5),
                            padding='valid',
                            strides=(2, 2),
                            data_format='channels_last',
                            activation='relu')(batchNorm)
    bn1 = K.layers.BatchNormalization(momentum=0.1, )(conv1)
    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(bn1)
    drop1 = K.layers.Dropout(0.3)(pool1)

    conv2 = K.layers.Conv2D(64, (3, 3),
                            padding='valid',
                            strides=(2, 2),
                            data_format='channels_last',
                            activation='relu')(drop1)
    bn2 = K.layers.BatchNormalization(momentum=0.1, )(conv2)
    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(bn2)
    drop2 = K.layers.Dropout(0.3)(pool2)

    conv3 = K.layers.Conv2D(128, (3, 3),
                            padding='valid',
                            strides=(1, 1),
                            data_format='channels_last',
                            activation='relu')(drop2)
    bn3 = K.layers.BatchNormalization(momentum=0.1, )(conv3)
    pool3 = K.layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(bn3)
    drop3 = K.layers.Dropout(0.3)(pool3)

    conv4 = K.layers.Conv2D(1, (2, 1),
                            padding='valid',
                            strides=(1, 1),
                            data_format='channels_last',
                            activation='linear')(drop3)

    gap = K.layers.GlobalAveragePooling2D(data_format='channels_last')(conv4)

    model = K.models.Model(inputs=inputs, outputs=gap)

    return model
