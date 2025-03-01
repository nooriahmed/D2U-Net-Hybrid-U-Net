def single_path_unet_global(input_shape=(224, 224, 3), dropout_rate=0.5):
    inputs = Input(input_shape)

    # Encoder: Global Features
    conv1 = SeparableConv2D(64, (3, 3), padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = PReLU()(conv1)
    conv1 = SeparableConv2D(64, (3, 3), padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = PReLU()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = SeparableConv2D(128, (3, 3), padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = PReLU()(conv2)
    conv2 = SeparableConv2D(128, (3, 3), padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = PReLU()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # CSFM_module
    conv3 = SeparableConv2D(256, (3, 3), padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = PReLU()(conv3)
    conv3 = Dropout(dropout_rate)(conv3)
    conv3 = SeparableConv2D(256, (3, 3), padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = PReLU()(conv3)

    # Decoder
    up4 = UpSampling2D(size=(2, 2))(conv3)
    up4 = Concatenate()([up4, conv2])
    conv4 = SeparableConv2D(128, (3, 3), padding='same')(up4)
    conv4 = BatchNormalization()(conv4)
    conv4 = PReLU()(conv4)
    conv4 = SeparableConv2D(128, (3, 3), padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = PReLU()(conv4)

    up5 = UpSampling2D(size=(2, 2))(conv4)
    up5 = Concatenate()([up5, conv1])
    conv5 = SeparableConv2D(64, (3, 3), padding='same')(up5)
    conv5 = BatchNormalization()(conv5)
    conv5 = PReLU()(conv5)
    conv5 = SeparableConv2D(64, (3, 3), padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = PReLU()(conv5)

    # Output Layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv5)

    model = Model(inputs, outputs)
    return model
