# Define Dual Path U-Net with Attention, Residual Connections, PReLU, and Dropout
def standard_unet_csfm(input_shape=(224, 224, 3), dropout_rate=0.5):
    inputs = Input(input_shape)

    # Standard U-Net Encoder
    conv1 = Conv2D(64, (3, 3), padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = PReLU()(conv1)
    conv1 = Conv2D(64, (3, 3), padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = PReLU()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = PReLU()(conv2)
    conv2 = Conv2D(128, (3, 3), padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = PReLU()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = PReLU()(conv3)
    conv3 = Conv2D(256, (3, 3), padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = PReLU()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = PReLU()(conv4)
    conv4 = Conv2D(512, (3, 3), padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = PReLU()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # CSFM Bottleneck
    conv_b = Conv2D(1024, (3, 3), padding='same')(pool4)
    conv_b = BatchNormalization()(conv_b)
    conv_b = PReLU()(conv_b)
    conv_b = Dropout(dropout_rate)(conv_b)
    conv_b = Conv2D(1024, (3, 3), padding='same')(conv_b)
    conv_b = BatchNormalization()(conv_b)
    conv_b = PReLU()(conv_b)

    # Standard U-Net Decoder
    up4 = UpSampling2D(size=(2, 2))(conv_b)
    up4 = Concatenate()([up4, conv4])
    conv_up4 = Conv2D(512, (3, 3), padding='same')(up4)
    conv_up4 = BatchNormalization()(conv_up4)
    conv_up4 = PReLU()(conv_up4)

    up3 = UpSampling2D(size=(2, 2))(conv_up4)
    up3 = Concatenate()([up3, conv3])
    conv_up3 = Conv2D(256, (3, 3), padding='same')(up3)
    conv_up3 = BatchNormalization()(conv_up3)
    conv_up3 = PReLU()(conv_up3)

    up2 = UpSampling2D(size=(2, 2))(conv_up3)
    up2 = Concatenate()([up2, conv2])
    conv_up2 = Conv2D(128, (3, 3), padding='same')(up2)
    conv_up2 = BatchNormalization()(conv_up2)
    conv_up2 = PReLU()(conv_up2)

    up1 = UpSampling2D(size=(2, 2))(conv_up2)
    up1 = Concatenate()([up1, conv1])
    conv_up1 = Conv2D(64, (3, 3), padding='same')(up1)
    conv_up1 = BatchNormalization()(conv_up1)
    conv_up1 = PReLU()(conv_up1)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv_up1)

    model = Model(inputs, outputs)
    return model


# Create the model
model = standard_unet_csfm((224, 224, 3))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

# Training with augmented data
history = model.fit(augmented_data, epochs=1, validation_data=(X_val, Y_val), verbose=1)

# Plotting training & validation loss and accuracy
def plot_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

plot_history(history)
