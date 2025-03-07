# Define Dual Path U-Net with Attention, Residual Connections, PReLU, and Dropout
def ghost_module(x, filters, ratio=2):
    """ Lightweight Ghost Module to reduce redundant computation """
    conv = SeparableConv2D(filters // ratio, (3, 3), padding='same')(x)
    conv = PReLU()(conv)
    cheap_operation = DepthwiseConv2D((3, 3), padding='same')(conv)
    return Concatenate()([conv, cheap_operation])

def lightweight_dual_path_unet(input_shape=(224, 224, 3), dropout_rate=0.3):
    inputs = Input(input_shape)

    # Path 1: Global Features (Contextual Information)
    conv1_1 = ghost_module(inputs, 32)
    pool1_1 = MaxPooling2D(pool_size=(2, 2))(conv1_1)

    conv2_1 = ghost_module(pool1_1, 64)
    pool2_1 = MaxPooling2D(pool_size=(2, 2))(conv2_1)

    # Path 2: Local Features (Edges)
    conv1_2 = ghost_module(inputs, 32)
    pool1_2 = MaxPooling2D(pool_size=(2, 2))(conv1_2)

    conv2_2 = ghost_module(pool1_2, 64)
    pool2_2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)

    # Bottleneck with Squeeze-and-Excitation (SE) block
    merged = Concatenate()([pool2_1, pool2_2])
    gap = GlobalAveragePooling2D()(merged)
    squeeze = Dense(64, activation='relu')(gap)
    excite = Dense(128, activation='sigmoid')(squeeze)
    excite = Reshape((1, 1, 128))(excite)
    merged = Multiply()([merged, excite])  # SE attention

    # Decoder 1 (Global Features)
    up4_1 = UpSampling2D(size=(2, 2))(merged)
    up4_1 = Concatenate()([up4_1, conv2_1])
    conv4_1 = ghost_module(up4_1, 64)

    up5_1 = UpSampling2D(size=(2, 2))(conv4_1)
    up5_1 = Concatenate()([up5_1, conv1_1])
    conv5_1 = ghost_module(up5_1, 32)

    # Decoder 2 (Local Features)
    up4_2 = UpSampling2D(size=(2, 2))(merged)
    up4_2 = Concatenate()([up4_2, conv2_2])
    conv4_2 = ghost_module(up4_2, 64)

    up5_2 = UpSampling2D(size=(2, 2))(conv4_2)
    up5_2 = Concatenate()([up5_2, conv1_2])
    conv5_2 = ghost_module(up5_2, 32)

    # Merge Outputs from Both Decoders
    merged_output = Concatenate()([conv5_1, conv5_2])
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(merged_output)

    model = Model(inputs, outputs)
    return model


# Create the model
model = lightweight_dual_path_unet((224, 224, 3))
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
