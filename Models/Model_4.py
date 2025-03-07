def dual_path_unet(input_shape=(224, 224, 3), dropout_rate=0.5):
    inputs = Input(input_shape)

    # Path 1: Global Features (Contextual Information)
    conv1_1 = SeparableConv2D(64, (3, 3), padding='same')(inputs)
    conv1_1 = BatchNormalization()(conv1_1)
    conv1_1 = PReLU()(conv1_1)
    conv1_1 = SeparableConv2D(64, (3, 3), padding='same')(conv1_1)
    conv1_1 = BatchNormalization()(conv1_1)
    conv1_1 = PReLU()(conv1_1)
    pool1_1 = MaxPooling2D(pool_size=(2, 2))(conv1_1)

    conv2_1 = SeparableConv2D(128, (3, 3), padding='same')(pool1_1)
    conv2_1 = BatchNormalization()(conv2_1)
    conv2_1 = PReLU()(conv2_1)
    conv2_1 = SeparableConv2D(128, (3, 3), padding='same')(conv2_1)
    conv2_1 = BatchNormalization()(conv2_1)
    conv2_1 = PReLU()(conv2_1)
    pool2_1 = MaxPooling2D(pool_size=(2, 2))(conv2_1)

    # Path 2: Local Features (Edges)
    conv1_2 = SeparableConv2D(64, (3, 3), padding='same')(inputs)
    conv1_2 = BatchNormalization()(conv1_2)
    conv1_2 = PReLU()(conv1_2)
    conv1_2 = SeparableConv2D(64, (3, 3), padding='same')(conv1_2)
    conv1_2 = BatchNormalization()(conv1_2)
    conv1_2 = PReLU()(conv1_2)
    pool1_2 = MaxPooling2D(pool_size=(2, 2))(conv1_2)

    conv2_2 = SeparableConv2D(128, (3, 3), padding='same')(pool1_2)
    conv2_2 = BatchNormalization()(conv2_2)
    conv2_2 = PReLU()(conv2_2)
    conv2_2 = SeparableConv2D(128, (3, 3), padding='same')(conv2_2)
    conv2_2 = BatchNormalization()(conv2_2)
    conv2_2 = PReLU()(conv2_2)
    pool2_2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)

    # **U-Net Bottleneck**
    merged = Concatenate()([pool2_1, pool2_2])
    conv_b1 = SeparableConv2D(256, (3, 3), padding='same')(merged)
    conv_b1 = BatchNormalization()(conv_b1)
    conv_b1 = PReLU()(conv_b1)
    conv_b1 = Dropout(dropout_rate)(conv_b1)
    conv_b2 = SeparableConv2D(256, (3, 3), padding='same')(conv_b1)
    conv_b2 = BatchNormalization()(conv_b2)
    conv_b2 = PReLU()(conv_b2)

    up_b1 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv_b2) # Changed from UpSampling2D to Conv2DTranspose
    up_b1 = Concatenate()([up_b1, conv2_1, conv2_2])
    conv_b3 = SeparableConv2D(128, (3, 3), padding='same')(up_b1)
    conv_b3 = BatchNormalization()(conv_b3)
    conv_b3 = PReLU()(conv_b3)

    up_b2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv_b3) # Changed from UpSampling2D to Conv2DTranspose
    up_b2 = Concatenate()([up_b2, conv1_1, conv1_2])
    conv_b4 = SeparableConv2D(64, (3, 3), padding='same')(up_b2)
    conv_b4 = BatchNormalization()(conv_b4)
    conv_b4 = PReLU()(conv_b4)

    # Merge Bottleneck Output into Decoder Paths
    decoder_input = Concatenate()([conv_b4, conv1_1, conv1_2])

    # Decoder 1 (Global Features)
    up4_1 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(decoder_input) # Changed from UpSampling2D to Conv2DTranspose
    conv4_1 = SeparableConv2D(128, (3, 3), padding='same')(up4_1)
    conv4_1 = BatchNormalization()(conv4_1)
    conv4_1 = PReLU()(conv4_1)

    up5_1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv4_1) # Changed from UpSampling2D to Conv2DTranspose
    conv5_1 = SeparableConv2D(64, (3, 3), padding='same')(up5_1)
    conv5_1 = BatchNormalization()(conv5_1)
    conv5_1 = PReLU()(conv5_1)

    # Decoder 2 (Local Features)
    up4_2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(decoder_input) # Changed from UpSampling2D to Conv2DTranspose
    conv4_2 = SeparableConv2D(128, (3, 3), padding='same')(up4_2)
    conv4_2 = BatchNormalization()(conv4_2)
    conv4_2 = PReLU()(conv4_2)

    up5_2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv4_2) # Changed from UpSampling2D to Conv2DTranspose
    conv5_2 = SeparableConv2D(64, (3, 3), padding='same')(up5_2)
    conv5_2 = BatchNormalization()(conv5_2)
    conv5_2 = PReLU()(conv5_2)

    # Merge Outputs from Both Decoders
    merged_output = Concatenate()([conv5_1, conv5_2])
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(merged_output)

    model = Model(inputs, outputs)
    return model


# Create the model
model = dual_path_unet((224, 224, 3))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

# Training with augmented data
history = model.fit(augmented_data, epochs=1, validation_data=(X_val, Y_val), verbose=1)
